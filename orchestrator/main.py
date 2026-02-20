import os
from dotenv import load_dotenv

load_dotenv(override=False)
import io
import time
import uuid
import shutil
import random
import threading
import queue
from typing import Dict, Optional, List, Any

import polars as pl
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from torchvision import datasets as tv_datasets
import socket

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

DATASETS_DIR = "/datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

MODELS_DIR = "/models"
os.makedirs(MODELS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "5"))

app = FastAPI(title="Distributed Training Orchestrator")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard HTML as the main page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
def health_check():
    """Health endpoint for load balancers."""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "timestamp": time.time(),
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    
    active_nodes = len([n for n in nodes.values() if time.time() - n.get("last_seen", 0) < NODE_TIMEOUT])
    total_nodes = len(nodes)
    current_epoch = current_run.get("current_epoch", 0)
    total_epochs = current_run.get("total_epochs", 0)
    is_training = current_run.get("run_id") is not None
    
    lines = [
        "# HELP tfg_nodes_total Total registered nodes",
        "# TYPE tfg_nodes_total gauge",
        f"tfg_nodes_total {total_nodes}",
        "",
        "# HELP tfg_nodes_active Currently active nodes",
        "# TYPE tfg_nodes_active gauge",
        f"tfg_nodes_active {active_nodes}",
        "",
        "# HELP tfg_training_active Whether training is in progress",
        "# TYPE tfg_training_active gauge",
        f"tfg_training_active {1 if is_training else 0}",
        "",
        "# HELP tfg_current_epoch Current training epoch",
        "# TYPE tfg_current_epoch gauge",
        f"tfg_current_epoch {current_epoch}",
        "",
        "# HELP tfg_total_epochs Total epochs for current run",
        "# TYPE tfg_total_epochs gauge",
        f"tfg_total_epochs {total_epochs}",
    ]
    
    return PlainTextResponse("\n".join(lines), media_type="text/plain")


wandb_queue: queue.Queue = queue.Queue()
wandb_thread: Optional[threading.Thread] = None
wandb_initialized = False


def wandb_worker():
    """Thread that processes W&B log requests."""
    while True:
        item = wandb_queue.get()
        if item is None:
            break
        try:
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(item)
        except Exception as e:
            print(f"[wandb] Error logging: {e}")
        wandb_queue.task_done()


def log_async(metrics: dict):
    """Non-blocking log to W&B."""
    wandb_queue.put(metrics)


def init_wandb(run_id: str, config: dict):
    """Initialize W&B with async thread."""
    global wandb_thread, wandb_initialized
    
    if not WANDB_AVAILABLE:
        print("[orchestrator] W&B not available")
        return False
    
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception as e:
            print(f"[wandb] Error login: {e}")
    
    try:
        project = os.getenv("WANDB_PROJECT", "tfg-distributed-ai")
        wandb.init(
            project=project,
            name=f"federated_{run_id[:8]}",
            id=run_id,
            config=config,
            resume="allow",
        )
        
        wandb_thread = threading.Thread(target=wandb_worker, daemon=True)
        wandb_thread.start()
        wandb_initialized = True
        
        print(f"[orchestrator] W&B initialized: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[wandb] Error init: {e}")
        return False


def finish_wandb():
    """Finalize W&B and stop the worker."""
    global wandb_initialized
    
    if wandb_initialized:
        wandb_queue.put(None)
        if wandb_thread:
            wandb_thread.join(timeout=5)
        
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        wandb_initialized = False


ORCHESTRATOR_IP: Optional[str] = None


def _log_local_ip() -> None:
    """Detect local IP."""
    global ORCHESTRATOR_IP
    ip = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = None
    ORCHESTRATOR_IP = ip
    if ip:
        print(f"[orchestrator] Detected IP: {ip}")
    else:
        print("[orchestrator] Could not detect IP")


_log_local_ip()


class NodeInfo(BaseModel):
    name: str
    addr: Optional[str] = None
    cpu: Optional[int] = None
    ram_gb: Optional[float] = None
    gpus: Optional[int] = None
    compute_power: Optional[float] = None
    free_disk_gb: Optional[float] = None
    status: Optional[str] = None


nodes: Dict[str, dict] = {}
jobs_queue: List[dict] = []  # legacy
datasets: Dict[str, dict] = {}

# State for the current federated learning run with per-epoch sync
current_run: Dict[str, Any] = {
    "run_id": None,
    "shards": {},           # node_name -> shard_file_path
    "weights": {},          # node_name -> final weight tensors (legacy)
    "config": {},
    "allocations": {},      # node_name -> allocation info
    "current_epoch": 0,
    "total_epochs": DEFAULT_EPOCHS,
    "epoch_weights": {},    # node_name -> weights for current epoch
    "epoch_metrics": {},    # node_name -> {loss, samples}
    "averaged_weights": None,
    "sync_lock": threading.Lock(),
    # Straggler handling
    "epoch_start_time": None,
    "epoch_timer": None,
}

# Perform partial FedAvg after this timeout
EPOCH_SYNC_TIMEOUT = int(os.getenv("EPOCH_SYNC_TIMEOUT", "120"))

HEARTBEAT_TIMEOUT = 30
HEARTBEAT_FAIL_TIMEOUT = 120

NODE_STATES = ["IDLE", "TRAINING", "SYNCING", "FINISHED", "FAILED", "DISCONNECTED"]
GLOBAL_STATES = ["INITIALIZING", "RUNNING", "DEGRADED", "COMPLETE", "FAILED"]

monitoring_state: Dict[str, Any] = {
    "run_id": None,
    "started_at": None,
    "global_state": "INITIALIZING",
    "nodes": {},
    "epoch_aggregates": {},
    "timeline": [],
    "timeline_max": 1000,
}


def get_node_monitor_state(node_name: str) -> dict:
    """Get or create monitoring state for a node."""
    if node_name not in monitoring_state["nodes"]:
        monitoring_state["nodes"][node_name] = {
            "name": node_name,
            "state": "IDLE",
            "current_epoch": 0,
            "total_epochs": 0,
            "last_heartbeat": None,
            "training_started": None,
            "hardware": {},
            "shard_info": {},
            "epoch_history": [],
            "errors": [],
            "samples_processed": 0,
            "cumulative_duration_sec": 0,
        }
    return monitoring_state["nodes"][node_name]


def add_timeline_event(event_type: str, data: dict = None):
    """Add event to timeline (circular buffer)."""
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp": time.time(),
        "type": event_type,
        "data": data or {}
    }
    monitoring_state["timeline"].append(event)
    if len(monitoring_state["timeline"]) > monitoring_state["timeline_max"]:
        monitoring_state["timeline"] = monitoring_state["timeline"][-monitoring_state["timeline_max"]:]


def compute_global_state() -> str:
    """Compute global training state based on individual node states."""
    if not monitoring_state["nodes"]:
        return "INITIALIZING"
    
    states = [n["state"] for n in monitoring_state["nodes"].values()]
    
    active = states.count("TRAINING") + states.count("SYNCING")
    failed = states.count("FAILED") + states.count("DISCONNECTED")
    finished = states.count("FINISHED")
    total = len(states)
    
    if finished == total:
        return "COMPLETE"
    elif failed == total:
        return "FAILED"
    elif failed > 0 and active > 0:
        return "DEGRADED"
    elif active > 0:
        return "RUNNING"
    else:
        return "INITIALIZING"


def check_node_timeouts():
    """Check node heartbeat timeouts and update states accordingly."""
    now = time.time()
    for name, node in monitoring_state["nodes"].items():
        if node["last_heartbeat"] is None:
            continue
        age = now - node["last_heartbeat"]
        
        if age > HEARTBEAT_FAIL_TIMEOUT and node["state"] not in ["FAILED", "FINISHED"]:
            node["state"] = "FAILED"
            add_timeline_event("NODE_FAILED", {"node": name, "reason": "timeout"})
        elif age > HEARTBEAT_TIMEOUT and node["state"] not in ["DISCONNECTED", "FAILED", "FINISHED"]:
            node["state"] = "DISCONNECTED"
            add_timeline_event("NODE_DISCONNECTED", {"node": name})
    
    monitoring_state["global_state"] = compute_global_state()


NODE_TIMEOUT = 60


def prune_nodes() -> None:
    """Remove nodes that haven't sent a heartbeat in NODE_TIMEOUT seconds."""
    now = time.time()
    dead = []
    for name, info in list(nodes.items()):
        if now - info.get("last_seen", now) > NODE_TIMEOUT:
            dead.append(name)
    for name in dead:
        nodes.pop(name, None)


@app.post("/register")
def register(node: NodeInfo):
    """
    Register a new node or update its data.
    Assigns a friendly name like agentN if the original name is a Docker hex ID or PID.
    """
    info = node.model_dump()
    info["last_seen"] = time.time()
    if "status" not in info or info["status"] is None:
        info["status"] = "idle"
    
    import re
    is_hex_id = bool(re.match(r"^[0-9a-f]{12}$", node.name))
    is_pid_fallback = node.name.startswith("agent-") and node.name[6:].isdigit()
    
    assigned_name = node.name
    
    if is_hex_id or is_pid_fallback:
        existing_agents = [n for n in nodes.keys() if n.startswith("agent") and n[5:].isdigit()]
        if not existing_agents:
            assigned_name = "agent1"
        else:
            nums = [int(n[5:]) for n in existing_agents]
            found = False
            for name, n_info in nodes.items():
                if n_info.get("original_name") == node.name:
                    assigned_name = name
                    found = True
                    break
            
            if not found:
                assigned_name = f"agent{max(nums) + 1}"
    
    info["original_name"] = node.name
    nodes[assigned_name] = info
    
    return {
        "ok": True, 
        "assigned_name": assigned_name,
        "registered_nodes": list(nodes.keys())
    }


@app.post("/heartbeat/{name}")
def heartbeat(name: str, payload: dict):
    """Receive a heartbeat from a node, update last seen time and status."""
    if name in nodes:
        nodes[name]["last_seen"] = time.time()
        status = payload.get("status")
        if status:
            nodes[name]["status"] = status
    return {"ok": True}


@app.get("/nodes")
def list_nodes():
    """List active nodes with heartbeat age."""
    prune_nodes()
    now = time.time()
    result = []
    for name, info in nodes.items():
        copy = info.copy()
        copy["age"] = round(now - info.get("last_seen", now), 1)
        result.append(copy)
    return sorted(result, key=lambda x: x["name"])


@app.post("/enqueue")
def enqueue_job(job: dict):
    """Add a job to the agent queue (legacy)."""
    jobs_queue.append(job)
    return {"queued": len(jobs_queue)}


@app.get("/assign/{name}")
def assign(name: str):
    """Assign the next pending job to a node (legacy)."""
    if not jobs_queue:
        return {"job": None}
    return {"job": jobs_queue.pop(0)}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset (CSV or Parquet) and store it on disk."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    filename = f"{dataset_id}_{file.filename}"
    file_path = os.path.join(DATASETS_DIR, filename)
    with open(file_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    if filename.lower().endswith((".parquet", ".feather")):
        df = pl.read_parquet(file_path)
    else:
        df = pl.read_csv(file_path)
    datasets[dataset_id] = {"path": file_path, "rows": df.height, "filename": file.filename}
    return {"dataset_id": dataset_id, "rows": df.height}


@app.get("/datasets")
def list_datasets():
    """List uploaded datasets."""
    return datasets


def prepare_mnist_dataset() -> dict:
    """Download and prepare the MNIST dataset as Parquet if not already present."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    if "mnist" in datasets and os.path.exists(datasets["mnist"].get("path", "")):
        return datasets["mnist"]

    file_path = os.path.join(DATASETS_DIR, "mnist.parquet")
    if os.path.exists(file_path):
        print("[orchestrator] Local MNIST found, skipping download.")
        df = pl.read_parquet(file_path)
        datasets["mnist"] = {"path": file_path, "rows": df.height, "filename": "mnist.parquet"}
        return datasets["mnist"]

    print("[orchestrator] Downloading MNIST...")
    mnist_root = os.path.join(DATASETS_DIR, "torchvision")
    mnist = tv_datasets.MNIST(root=mnist_root, train=True, download=True)
    images = mnist.data.float().div(255.0).view(len(mnist), -1)
    labels = mnist.targets

    df = pl.DataFrame(images.numpy())
    df = df.with_columns(pl.Series("label", labels.numpy()))

    df.write_parquet(file_path)
    datasets["mnist"] = {"path": file_path, "rows": len(mnist), "filename": "mnist.parquet"}
    return datasets["mnist"]


@app.post("/train/{dataset_id}")
def train_dataset(dataset_id: str, epochs: int = DEFAULT_EPOCHS, overlap_pct: float = 0.15, precision: str = "fp32"):
    """
    Start federated training.
    Prepares shards, initializes global state, and notifies agents.
    """
    prune_nodes()
    if dataset_id == "mnist":
        dataset_info = prepare_mnist_dataset()
    elif dataset_id in datasets:
        dataset_info = datasets[dataset_id]
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not nodes:
        raise HTTPException(status_code=400, detail="No active nodes available for training")
    
    df = pl.read_parquet(dataset_info["path"])
    dataset_size = os.path.getsize(dataset_info["path"])
    allocations = compute_shard_allocation(len(df), dataset_size)
    shard_paths = create_shards_with_overlap(df, allocations, overlap_pct)
    
    run_id = str(uuid.uuid4())
    config = {
        "dataset_id": dataset_id,
        "overlap_pct": overlap_pct,
        "epochs": epochs,
        "nodes": list(shard_paths.keys()),
        "total_rows": len(df),
        "precision": precision
    }
    
    current_run["run_id"] = run_id
    current_run["shards"] = shard_paths
    current_run["weights"] = {}
    current_run["allocations"] = allocations
    current_run["config"] = config
    current_run["current_epoch"] = 0
    current_run["total_epochs"] = epochs
    current_run["epoch_weights"] = {}
    current_run["epoch_metrics"] = {}
    current_run["averaged_weights"] = None
    
    init_wandb(run_id, config)
    
    monitoring_state["run_id"] = run_id
    monitoring_state["started_at"] = time.time()
    monitoring_state["global_state"] = "TRAINING"
    monitoring_state["nodes"] = {}
    
    add_timeline_event("TRAINING_STARTED", {"run_id": run_id, "nodes": len(nodes)})
    
    print(f"[orchestrator] Started federated training {run_id} with {len(nodes)} nodes")
    
    return {
        "status": "started",
        "run_id": run_id,
        "nodes": list(nodes.keys()),
        "epochs": epochs
    }

@app.get("/ip")
def get_ip() -> dict:
    """Return the IP detected at startup."""
    return {"ip": ORCHESTRATOR_IP}


def compute_shard_allocation(dataset_rows: int, dataset_size_bytes: int) -> Dict[str, dict]:
    """Allocate dataset rows proportionally to each node's compute_power, limited by available disk."""
    prune_nodes()
    if not nodes:
        raise HTTPException(status_code=400, detail="No active nodes")
    
    allocations = {}
    total_power = sum(n.get("compute_power", 1) for n in nodes.values())
    bytes_per_row = dataset_size_bytes / dataset_rows if dataset_rows > 0 else 0
    
    for name, node in nodes.items():
        power_ratio = node.get("compute_power", 1) / total_power
        desired_rows = int(dataset_rows * power_ratio)
        
        # Disk constraint: shard must be < 20% of free disk space
        free_bytes = node.get("free_disk_gb", 1) * (1024 ** 3)
        max_shard_bytes = free_bytes * 0.20
        max_rows_by_disk = int(max_shard_bytes / bytes_per_row) if bytes_per_row > 0 else desired_rows
        
        allocated_rows = min(desired_rows, max_rows_by_disk)
        estimated_size = int(allocated_rows * bytes_per_row)
        allocations[name] = {"rows": allocated_rows, "size_bytes": estimated_size}
    
    return allocations


def create_shards_with_overlap(
    df: pl.DataFrame,
    allocations: Dict[str, dict],
    overlap_pct: float = 0.15
) -> Dict[str, str]:
    """Create shard files with random sampling and controlled overlap (default 15%)."""
    total_rows = len(df)
    indices = list(range(total_rows))
    random.shuffle(indices)
    
    shard_paths = {}
    used_indices = set()
    cursor = 0
    
    for name, alloc in allocations.items():
        num_rows = alloc["rows"]
        if num_rows <= 0:
            continue
        
        fresh_count = int(num_rows * (1 - overlap_pct))
        overlap_count = num_rows - fresh_count
        
        fresh_indices = indices[cursor:cursor + fresh_count]
        cursor += fresh_count
        
        # Wrap around if we run out of indices
        if cursor > len(indices):
            cursor = 0
            random.shuffle(indices)
            fresh_indices = indices[:fresh_count]
            cursor = fresh_count
        
        # Add random overlap from previously used indices
        if used_indices and overlap_count > 0:
            overlap_pool = list(used_indices)
            overlap_indices = random.sample(
                overlap_pool,
                min(overlap_count, len(overlap_pool))
            )
        else:
            overlap_indices = []
        
        shard_indices = fresh_indices + overlap_indices
        used_indices.update(fresh_indices)
        
        shard_df = df[shard_indices]
        shard_path = os.path.join(DATASETS_DIR, f"shard_{name}.parquet")
        shard_df.write_parquet(shard_path)
        shard_paths[name] = shard_path
        
        print(f"[orchestrator] Shard created for {name}: {len(shard_indices)} rows ({overlap_count} overlap)")
    
    return shard_paths


def federated_average(weights_dict: Dict[str, dict]) -> dict:
    """Average weights from all nodes (FedAvg algorithm)."""
    if not weights_dict:
        raise ValueError("No weights to average")
    
    node_names = list(weights_dict.keys())
    result = {}
    
    first_weights = weights_dict[node_names[0]]
    
    for key in first_weights.keys():
        stacked = torch.stack([weights_dict[n][key].float() for n in node_names])
        result[key] = stacked.mean(dim=0)
    
    return result


@app.post("/train/{dataset_id}/prepare")
def prepare_training(dataset_id: str, overlap_pct: float = 0.15, epochs: int = 5):
    """Prepare shards for all active nodes."""
    prune_nodes()
    
    if dataset_id == "mnist":
        dataset_info = prepare_mnist_dataset()
    elif dataset_id in datasets:
        dataset_info = datasets[dataset_id]
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if not nodes:
        raise HTTPException(status_code=400, detail="No active nodes available for training")
    
    df = pl.read_parquet(dataset_info["path"])
    dataset_size = os.path.getsize(dataset_info["path"])
    
    allocations = compute_shard_allocation(len(df), dataset_size)
    shard_paths = create_shards_with_overlap(df, allocations, overlap_pct)
    
    run_id = str(uuid.uuid4())
    
    config = {
        "dataset_id": dataset_id,
        "overlap_pct": overlap_pct,
        "epochs": epochs,
        "nodes": list(shard_paths.keys()),
        "total_rows": len(df),
    }
    
    current_run["run_id"] = run_id
    current_run["shards"] = shard_paths
    current_run["weights"] = {}
    current_run["allocations"] = allocations
    current_run["config"] = config
    current_run["current_epoch"] = 0
    current_run["total_epochs"] = epochs
    current_run["epoch_weights"] = {}
    current_run["epoch_metrics"] = {}
    current_run["averaged_weights"] = None
    
    monitoring_state["run_id"] = run_id
    monitoring_state["started_at"] = time.time()
    monitoring_state["global_state"] = "INITIALIZING"
    monitoring_state["nodes"] = {}
    monitoring_state["epoch_aggregates"] = {}
    monitoring_state["timeline"] = []
    
    for node_name in shard_paths.keys():
        node = get_node_monitor_state(node_name)
        node["total_epochs"] = epochs
        node["shard_info"] = allocations.get(node_name, {})
        add_timeline_event("NODE_JOINED", {"node": node_name})
    
    add_timeline_event("RUN_STARTED", {"run_id": run_id, "nodes": list(shard_paths.keys())})
    
    init_wandb(run_id, config)
    
    print(f"[orchestrator] Run {run_id} prepared: {len(shard_paths)} shards, {epochs} epochs")
    
    return {
        "run_id": run_id,
        "epochs": epochs,
        "allocations": {k: {"rows": v["rows"], "size_bytes": v["size_bytes"]} 
                       for k, v in allocations.items()},
        "nodes": list(shard_paths.keys()),
    }



@app.get("/shards/{node_name}")
def download_shard(node_name: str):
    """Agent downloads its assigned shard."""
    if node_name not in current_run.get("shards", {}):
        raise HTTPException(status_code=404, detail="No shard assigned for this node")
    
    shard_path = current_run["shards"][node_name]
    if not os.path.exists(shard_path):
        raise HTTPException(status_code=404, detail="Shard file not found")
    
    return FileResponse(
        shard_path,
        media_type="application/octet-stream",
        filename=f"shard_{node_name}.parquet"
    )


@app.get("/train/status")
def training_status():
    """Get current training run status."""
    expected = list(current_run.get("shards", {}).keys())
    received = list(current_run.get("weights", {}).keys())
    
    return {
        "run_id": current_run.get("run_id"),
        "expected_nodes": expected,
        "received_weights": received,
        "complete": len(received) == len(expected) and len(expected) > 0,
        "allocations": current_run.get("allocations", {}),
    }


@app.post("/weights/upload/{node_name}")
async def upload_weights(node_name: str, file: UploadFile = File(...)):
    """Agent uploads trained weights. When all have uploaded, performs FedAvg."""
    if node_name not in current_run.get("shards", {}):
        raise HTTPException(status_code=400, detail="Node is not part of the current training run")
    
    content = await file.read()
    weights = torch.load(io.BytesIO(content), map_location="cpu")
    current_run["weights"][node_name] = weights
    
    print(f"[orchestrator] Weights received from {node_name} ({len(current_run['weights'])}/{len(current_run['shards'])})")
    
    if len(current_run["weights"]) == len(current_run["shards"]):
        run_id = current_run["run_id"]
        
        aggregated = federated_average(current_run["weights"])
        
        model_path = os.path.join(MODELS_DIR, f"{run_id}.pt")
        torch.save(aggregated, model_path)
        
        print(f"[orchestrator] Training complete! Model saved to {model_path}")
        
        return {
            "status": "complete",
            "run_id": run_id,
            "model_path": model_path,
            "nodes_participated": list(current_run["weights"].keys()),
        }
    
    return {
        "status": "waiting",
        "received": len(current_run["weights"]),
        "expected": len(current_run["shards"]),
    }


@app.get("/models/{run_id}")
def download_model(run_id: str):
    """Download an aggregated model."""
    model_path = os.path.join(MODELS_DIR, f"{run_id}.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"model_{run_id}.pt"
    )


@app.get("/models")
def list_models():
    """List all saved models."""
    models = []
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pt"):
                model_path = os.path.join(MODELS_DIR, filename)
                models.append({
                    "run_id": filename.replace(".pt", ""),
                    "path": model_path,
                    "size_bytes": os.path.getsize(model_path),
                })
    return {"models": models}


def _perform_fedavg_and_save(epoch: int, is_partial: bool = False, missing_nodes: set = None):
    """Perform FedAvg on received weights and save the result."""
    aggregated = federated_average(current_run["epoch_weights"])
    
    buffer = io.BytesIO()
    torch.save(aggregated, buffer)
    buffer.seek(0)
    current_run["averaged_weights"] = buffer.getvalue()
    current_run["current_epoch"] = epoch
    
    # Compute aggregated metrics
    total_samples = sum(m["samples"] for m in current_run["epoch_metrics"].values())
    weighted_loss = sum(
        m["loss"] * m["samples"] 
        for m in current_run["epoch_metrics"].values()
    ) / total_samples if total_samples > 0 else 0
    
    accuracies = [m.get("accuracy", 0) for m in current_run["epoch_metrics"].values() if m.get("accuracy") is not None]
    weighted_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    
    epoch_times = [m.get("epoch_time_sec") for m in current_run["epoch_metrics"].values() if m.get("epoch_time_sec") is not None]
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None
    max_epoch_time = max(epoch_times) if epoch_times else None
    min_epoch_time = min(epoch_times) if epoch_times else None
    
    # Async W&B logging
    nodes_synced = len(current_run["epoch_weights"])
    wandb_metrics = {
        "epoch": epoch,
        "loss": weighted_loss,
        "total_samples": total_samples,
        "nodes_synced": nodes_synced,
    }
    if weighted_accuracy is not None:
        wandb_metrics["accuracy"] = weighted_accuracy
    if is_partial:
        wandb_metrics["partial_sync"] = True
        wandb_metrics["missing_nodes"] = len(missing_nodes) if missing_nodes else 0
    
    if avg_epoch_time is not None:
        wandb_metrics["epoch_time_avg_sec"] = avg_epoch_time
        wandb_metrics["epoch_time_max_sec"] = max_epoch_time
        wandb_metrics["epoch_time_min_sec"] = min_epoch_time
    log_async(wandb_metrics)
    
    partial_str = " (PARTIAL - stragglers excluded)" if is_partial else ""
    accuracy_str = f", avg_accuracy={weighted_accuracy:.2%}" if weighted_accuracy else ""
    time_str = f", avg_time={avg_epoch_time:.2f}s" if avg_epoch_time else ""
    print(f"[orchestrator] Epoch {epoch} complete{partial_str}: avg_loss={weighted_loss:.6f}{accuracy_str}{time_str}")
    
    if current_run["epoch_timer"] is not None:
        current_run["epoch_timer"].cancel()
        current_run["epoch_timer"] = None
    
    # Reset epoch state for next round
    current_run["epoch_weights"] = {}
    current_run["epoch_metrics"] = {}
    current_run["epoch_start_time"] = None
    
    # Check if training is complete
    if epoch >= current_run.get("total_epochs", DEFAULT_EPOCHS):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)
        torch.save(aggregated, model_path)
        
        finish_wandb()
        print(f"[orchestrator] Training complete! Final model at {model_path}")
        
        return {"status": "training_complete", "epoch": epoch, "model_path": model_path}
    
    return {"status": "epoch_complete", "epoch": epoch, "avg_loss": weighted_loss}


@app.post("/sync/epoch")
async def sync_epoch(
    node_name: str = Form(...),
    epoch: int = Form(...),
    loss: float = Form(...),
    samples: int = Form(...),
    accuracy: float = Form(None),
    epoch_time_sec: float = Form(None),
    file: UploadFile = File(...)
):
    """Agent uploads epoch weights and metrics. When all report, performs FedAvg."""
    if node_name not in current_run.get("shards", {}):
        raise HTTPException(status_code=400, detail="Node is not part of the current training run")
    
    content = await file.read()
    weights = torch.load(io.BytesIO(content), map_location="cpu")
    
    with current_run["sync_lock"]:
        current_run["epoch_weights"][node_name] = weights
        current_run["epoch_metrics"][node_name] = {
            "loss": loss, 
            "samples": samples,
            "accuracy": accuracy,
            "epoch_time_sec": epoch_time_sec,
        }
        
        received = len(current_run["epoch_weights"])
        expected = len(current_run["shards"])
        
        print(f"[orchestrator] Epoch {epoch}: received weights from {node_name} ({received}/{expected})")
        
        # Start epoch timeout timer on first weight arrival
        if received == 1 and current_run["epoch_timer"] is None:
            current_run["epoch_start_time"] = time.time()
            
            def epoch_timeout_handler():
                """Called after EPOCH_SYNC_TIMEOUT - performs partial FedAvg."""
                with current_run["sync_lock"]:
                    received_now = len(current_run["epoch_weights"])
                    expected_now = len(current_run["shards"])
                    
                    if received_now < expected_now and received_now > 0:
                        missing_nodes = set(current_run["shards"].keys()) - set(current_run["epoch_weights"].keys())
                        print(f"[orchestrator] WARNING: Epoch {epoch} timeout! Continuing with {received_now}/{expected_now} nodes")
                        print(f"[orchestrator] Missing nodes (stragglers): {missing_nodes}")
                        
                        _perform_fedavg_and_save(epoch, is_partial=True, missing_nodes=missing_nodes)
            
            current_run["epoch_timer"] = threading.Timer(EPOCH_SYNC_TIMEOUT, epoch_timeout_handler)
            current_run["epoch_timer"].daemon = True
            current_run["epoch_timer"].start()
            print(f"[orchestrator] Epoch {epoch} timer started ({EPOCH_SYNC_TIMEOUT}s timeout)")
        
        # All nodes reported - perform full FedAvg
        if received == expected:
            result = _perform_fedavg_and_save(epoch, is_partial=False)
            return result
    
    return {
        "status": "waiting",
        "epoch": epoch,
        "received": received,
        "expected": expected,
    }


@app.get("/sync/weights")
def get_averaged_weights():
    """Agent downloads the averaged weights after sync."""
    if current_run.get("averaged_weights") is None:
        raise HTTPException(status_code=404, detail="No averaged weights available yet")
    
    return StreamingResponse(
        io.BytesIO(current_run["averaged_weights"]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=averaged_weights.pt"}
    )


@app.get("/sync/status")
def sync_status():
    """Query epoch sync status."""
    expected = list(current_run.get("shards", {}).keys())
    received = list(current_run.get("epoch_weights", {}).keys())
    
    return {
        "run_id": current_run.get("run_id"),
        "current_epoch": current_run.get("current_epoch", 0),
        "total_epochs": current_run.get("total_epochs", DEFAULT_EPOCHS),
        "expected_nodes": expected,
        "received_this_epoch": received,
        "epoch_complete": len(received) == 0 and current_run.get("averaged_weights") is not None,
        "training_complete": current_run.get("current_epoch", 0) >= current_run.get("total_epochs", DEFAULT_EPOCHS),
    }


class HeartbeatRequest(BaseModel):
    node_name: str
    state: str = "IDLE"
    current_epoch: int = 0
    total_epochs: int = 0
    hardware: Optional[dict] = None


class EpochReportRequest(BaseModel):
    node_name: str
    epoch: int
    loss: float
    samples_processed: int
    epoch_duration_sec: float
    accuracy: Optional[float] = None
    memory_usage_mb: Optional[int] = None


@app.post("/agent/heartbeat")
def agent_heartbeat(request: HeartbeatRequest):
    """Agent sends periodic heartbeat with state and hardware info."""
    if request.node_name in nodes:
        nodes[request.node_name]["last_seen"] = time.time()
        if request.state:
            nodes[request.node_name]["status"] = request.state

    node = get_node_monitor_state(request.node_name)
    
    old_state = node["state"]
    node["state"] = request.state
    node["current_epoch"] = request.current_epoch
    node["total_epochs"] = request.total_epochs
    node["last_heartbeat"] = time.time()
    
    if request.hardware:
        node["hardware"] = request.hardware
    
    # Track state transitions
    if old_state != request.state:
        if request.state == "TRAINING" and node["training_started"] is None:
            node["training_started"] = time.time()
            add_timeline_event("NODE_STARTED_TRAINING", {"node": request.node_name})
        elif request.state == "FINISHED":
            add_timeline_event("NODE_FINISHED", {"node": request.node_name})
    
    monitoring_state["global_state"] = compute_global_state()
    
    return {"ok": True, "server_time": time.time()}


@app.post("/agent/epoch")
def agent_epoch_report(request: EpochReportRequest):
    """Agent reports metrics after completing an epoch."""
    node = get_node_monitor_state(request.node_name)
    
    node["current_epoch"] = request.epoch
    node["samples_processed"] += request.samples_processed
    node["cumulative_duration_sec"] += request.epoch_duration_sec
    
    epoch_record = {
        "epoch": request.epoch,
        "loss": request.loss,
        "duration_sec": request.epoch_duration_sec,
        "samples": request.samples_processed,
        "accuracy": request.accuracy,
        "timestamp": time.time(),
    }
    node["epoch_history"].append(epoch_record)
    
    # Keep only last 100 epochs per node
    if len(node["epoch_history"]) > 100:
        node["epoch_history"] = node["epoch_history"][-100:]
    
    epoch_key = request.epoch
    if epoch_key not in monitoring_state["epoch_aggregates"]:
        monitoring_state["epoch_aggregates"][epoch_key] = {
            "nodes_reported": [],
            "losses": [],
            "total_samples": 0,
            "completed_at": None,
        }
    
    agg = monitoring_state["epoch_aggregates"][epoch_key]
    if request.node_name not in agg["nodes_reported"]:
        agg["nodes_reported"].append(request.node_name)
        agg["losses"].append(request.loss)
        agg["total_samples"] += request.samples_processed
        agg["completed_at"] = time.time()
    
    add_timeline_event("EPOCH_COMPLETE", {
        "node": request.node_name,
        "epoch": request.epoch,
        "loss": request.loss,
    })
    
    return {
        "ok": True,
        "nodes_reported_this_epoch": len(agg["nodes_reported"]),
    }


@app.get("/monitor/overview")
def monitor_overview():
    """Training overview with aggregated metrics."""
    check_node_timeouts()
    
    nodes_list = list(monitoring_state["nodes"].values())
    
    state_counts = {
        "total": len(nodes_list),
        "active": sum(1 for n in nodes_list if n["state"] in ["TRAINING", "SYNCING"]),
        "finished": sum(1 for n in nodes_list if n["state"] == "FINISHED"),
        "failed": sum(1 for n in nodes_list if n["state"] in ["FAILED", "DISCONNECTED"]),
        "idle": sum(1 for n in nodes_list if n["state"] == "IDLE"),
    }
    
    epochs_completed = [n["current_epoch"] for n in nodes_list if n["current_epoch"] > 0]
    total_epochs = current_run.get("total_epochs", DEFAULT_EPOCHS)
    
    epoch_progress = {
        "total": total_epochs,
        "min_completed": min(epochs_completed) if epochs_completed else 0,
        "max_completed": max(epochs_completed) if epochs_completed else 0,
        "avg_completed": sum(epochs_completed) / len(epochs_completed) if epochs_completed else 0,
    }
    
    started_at = monitoring_state.get("started_at")
    elapsed_sec = time.time() - started_at if started_at else 0
    
    avg_epoch_durations = [
        n["cumulative_duration_sec"] / n["current_epoch"]
        for n in nodes_list if n["current_epoch"] > 0
    ]
    avg_epoch_time = sum(avg_epoch_durations) / len(avg_epoch_durations) if avg_epoch_durations else 0
    remaining_epochs = total_epochs - epoch_progress["avg_completed"]
    estimated_remaining_sec = remaining_epochs * avg_epoch_time
    
    all_losses = [
        record["loss"] 
        for n in nodes_list 
        for record in n.get("epoch_history", [])
    ]
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    total_samples = sum(n.get("samples_processed", 0) for n in nodes_list)
    
    node_speeds = [
        (n["name"], n["cumulative_duration_sec"] / n["current_epoch"])
        for n in nodes_list if n["current_epoch"] > 0
    ]
    node_speeds.sort(key=lambda x: x[1])
    
    performance = {}
    if node_speeds:
        performance["fastest_node"] = node_speeds[0][0]
        performance["slowest_node"] = node_speeds[-1][0]
        fastest_time = node_speeds[0][1]
        slowest_time = node_speeds[-1][1]
        performance["speed_variance_pct"] = ((slowest_time - fastest_time) / fastest_time * 100) if fastest_time > 0 else 0
    
    dataset_coverage = {}
    if current_run.get("config"):
        total_rows = current_run["config"].get("total_rows", 0)
        if total_rows > 0:
            dataset_coverage["total_rows"] = total_rows
            dataset_coverage["processed_rows"] = total_samples
            dataset_coverage["coverage_pct"] = min(100, total_samples / total_rows * 100)
            dataset_coverage["redundancy_pct"] = current_run["config"].get("overlap_pct", 0.15) * 100
    
    return {
        "run_id": current_run.get("run_id"),
        "global_state": monitoring_state["global_state"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)) if started_at else None,
        "elapsed_sec": round(elapsed_sec, 1),
        "estimated_remaining_sec": round(estimated_remaining_sec, 1),
        
        "epochs": epoch_progress,
        "nodes": state_counts,
        
        "metrics": {
            "avg_loss": round(avg_loss, 6) if avg_loss else None,
            "total_samples": total_samples,
            "avg_epoch_duration_sec": round(avg_epoch_time, 2) if avg_epoch_time else None,
        },
        
        "dataset": dataset_coverage,
        "performance": performance,
        
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/monitor/nodes")
def monitor_nodes():
    """List all nodes with current state."""
    check_node_timeouts()
    
    nodes_list = []
    now = time.time()
    
    for name, node in monitoring_state["nodes"].items():
        heartbeat_age = now - node["last_heartbeat"] if node["last_heartbeat"] else None
        
        avg_epoch_duration = None
        if node["current_epoch"] > 0:
            avg_epoch_duration = node["cumulative_duration_sec"] / node["current_epoch"]
        
        nodes_list.append({
            "name": name,
            "state": node["state"],
            "current_epoch": node["current_epoch"],
            "total_epochs": node["total_epochs"],
            "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(node["last_heartbeat"])) if node["last_heartbeat"] else None,
            "heartbeat_age_sec": round(heartbeat_age, 1) if heartbeat_age else None,
            "samples_processed": node["samples_processed"],
            "avg_epoch_duration_sec": round(avg_epoch_duration, 2) if avg_epoch_duration else None,
            "hardware": node["hardware"],
            "errors": node["errors"][-5:],
        })
    
    return {"nodes": nodes_list}


@app.get("/monitor/nodes/{node_name}")
def monitor_node_detail(node_name: str):
    """Detailed metrics for a single node."""
    if node_name not in monitoring_state["nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = monitoring_state["nodes"][node_name]
    
    shard_info = {}
    if node_name in current_run.get("allocations", {}):
        shard_info = current_run["allocations"][node_name]
        shard_info["path"] = current_run.get("shards", {}).get(node_name)
    
    timing = {
        "training_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(node["training_started"])) if node["training_started"] else None,
        "total_training_sec": node["cumulative_duration_sec"],
        "avg_epoch_sec": node["cumulative_duration_sec"] / node["current_epoch"] if node["current_epoch"] > 0 else None,
    }
    
    return {
        "name": node_name,
        "state": node["state"],
        "hardware": node["hardware"],
        "shard": shard_info,
        "epochs": node["epoch_history"],
        "timing": timing,
        "errors": node["errors"],
    }


@app.get("/monitor/timeline")
def monitor_timeline(limit: int = 100):
    """Event timeline for debugging."""
    events = monitoring_state["timeline"][-limit:]
    return {"events": events}
