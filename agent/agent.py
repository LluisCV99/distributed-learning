import os
from dotenv import load_dotenv
import time
import threading
import socket
import platform
import requests
import subprocess
import psutil
import torch
import sys

load_dotenv(override=False)

DEFAULT_ORCH_ENDPOINTS = [
    "http://localhost:11999",
    "http://host.docker.internal:11999",
    "http://orchestrator:11999", 
]


def get_orchestrator_endpoints() -> list:
    """Return list of orchestrator endpoints to try. Supports comma-separated URLs."""
    env_urls = os.getenv("ORCHESTRATOR_URL") or os.getenv("ORCH_ENDPOINT") or ""
    
    if env_urls.strip():
        custom_endpoints = [url.strip() for url in env_urls.split(",") if url.strip()]
        if custom_endpoints:
            all_endpoints = custom_endpoints + [e for e in DEFAULT_ORCH_ENDPOINTS if e not in custom_endpoints]
            return all_endpoints
    
    return DEFAULT_ORCH_ENDPOINTS


ORCH_ENDPOINT = None

def get_node_name() -> str:
    """Get node name from env or hostname."""
    env_name = os.getenv("NODE_ID") or os.getenv("NODE_NAME")
    if env_name:
        return env_name
    
    try:
        hostname = socket.gethostname()
        if hostname and hostname != "localhost":
            return hostname
    except Exception:
        pass
    
    return f"agent-{os.getpid()}"

NODE_NAME = get_node_name()

DATA_DIR = os.getenv("DATA_DIR", "/data")


def discover_orchestrator() -> str:
    """Try connecting to known endpoints. Returns the first one that responds."""
    global ORCH_ENDPOINT
    
    endpoints = get_orchestrator_endpoints()
    print(f"[agent] Endpoints to try: {endpoints}")
    
    for endpoint in endpoints:
        try:
            print(f"[agent] Trying {endpoint}...")
            response = requests.get(f"{endpoint}/ip", timeout=3)
            if response.status_code == 200:
                print(f"[agent] Connected to {endpoint}")
                ORCH_ENDPOINT = endpoint
                return endpoint
        except requests.exceptions.RequestException:
            print(f"[agent] Failed {endpoint}")
            continue
    
    print(f"[agent] WARNING: No orchestrator found, retrying with {DEFAULT_ORCH_ENDPOINTS[0]}")
    ORCH_ENDPOINT = DEFAULT_ORCH_ENDPOINTS[0]
    return ORCH_ENDPOINT


def measure_compute_power():
    """Measure compute capacity: CPUs, RAM, GPUs, free disk."""
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    try:
        disk_path = DATA_DIR if os.path.exists(DATA_DIR) else "/"
        disk_usage = psutil.disk_usage(disk_path)
        free_disk_gb = disk_usage.free / (1024 ** 3)
    except Exception:
        free_disk_gb = 0.0
    
    try:
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0

    # Heuristic: cpus + 4*gpus
    compute_power = cpu_count + gpu_count * 4
    return cpu_count, ram_gb, gpu_count, compute_power, free_disk_gb


def get_hardware_info() -> dict:
    """Return hardware info for monitoring."""
    cpu_count, ram_gb, gpu_count, _, free_disk_gb = measure_compute_power()
    
    gpu_name = None
    if gpu_count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return {
        "cpu_count": cpu_count,
        "ram_gb": round(ram_gb, 2),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "free_disk_gb": round(free_disk_gb, 2),
    }


_agent_state = {
    "state": "IDLE",
    "current_epoch": 0,
    "total_epochs": 0,
}


def heartbeat_loop():
    """Thread that sends periodic heartbeats."""
    while True:
        send_heartbeat()
        time.sleep(5)

def send_heartbeat():
    """Send heartbeat with current state and hardware info."""
    try:
        payload = {
            "node_name": NODE_NAME,
            "state": _agent_state["state"],
            "current_epoch": _agent_state["current_epoch"],
            "total_epochs": _agent_state["total_epochs"],
            "hardware": get_hardware_info(),
        }
        url = f"{ORCH_ENDPOINT}/agent/heartbeat"
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[agent] Heartbeat failed: {e}")


def post(path: str, json_data: dict):
    """POST request to the orchestrator."""
    url = f"{ORCH_ENDPOINT}{path}"
    return requests.post(url, json=json_data, timeout=5).json()


def get(path: str):
    """GET request to the orchestrator."""
    url = f"{ORCH_ENDPOINT}{path}"
    return requests.get(url, timeout=5).json()


def main():
    global NODE_NAME
    discover_orchestrator()
    
    cpu, ram_gb, gpus, compute_power, free_disk_gb = measure_compute_power()
    print(f"[agent] Node capacity: {cpu} CPUs, {ram_gb:.1f} GB RAM, {gpus} GPUs, {free_disk_gb:.1f} GB free disk")
    
    def get_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except:
                return "127.0.0.1"

    addr = os.getenv("NODE_IP") or get_ip()
    print(f"[agent] Detected address: {addr}")
    
    register_payload = {
        "name": NODE_NAME,
        "addr": addr,
        "cpu": cpu,
        "ram_gb": ram_gb,
        "gpus": gpus,
        "compute_power": compute_power,
        "free_disk_gb": free_disk_gb,
    }
    
    registered = False
    while not registered:
        try:
            resp = post("/register", register_payload)
            
            assigned = resp.get("assigned_name")
            if assigned and assigned != NODE_NAME:
                print(f"[agent] Name changed from {NODE_NAME} to {assigned}")
                NODE_NAME = assigned
            
            print(f"[agent] Registered as {NODE_NAME}: {resp}")
            registered = True
        except Exception as exc:
            print(f"[agent] Registration error: {exc}")
            print("[agent] Retrying in 5 seconds...")
            time.sleep(5)
            discover_orchestrator()

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    while True:
        # Check for federated training tasks
        try:
            status = get("/train/status")
            run_id = status.get("run_id")
            expected_nodes = status.get("expected_nodes", [])
            
            if run_id and NODE_NAME in expected_nodes:
                sync_status = get("/sync/status")
                training_complete = sync_status.get("training_complete", False)
                total_epochs = sync_status.get("total_epochs", 1)
                
                if not training_complete:
                    print(f"[agent] Run {run_id} detected, starting training...")
                    
                    _agent_state["state"] = "TRAINING"
                    _agent_state["total_epochs"] = total_epochs
                    send_heartbeat()
                    
                    shard_path = download_shard()
                    if shard_path:
                        success = run_local_training(shard_path, total_epochs)
                        
                        if success:
                            print(f"[agent] Federated training complete!")
                            _agent_state["state"] = "FINISHED"
                        else:
                            _agent_state["state"] = "FAILED"
                        
                        send_heartbeat()
                        
                        try:
                            os.remove(shard_path)
                        except Exception:
                            pass
                    
                    try:
                        post(f"/heartbeat/{NODE_NAME}", {"status": "idle"})
                    except Exception:
                        pass
        
        except Exception as e:
            print(f"[agent] Error checking status: {e}")
        
        # Legacy job queue (backwards compatibility)
        try:
            job = get(f"/assign/{NODE_NAME}").get("job")
        except Exception as e:
            job = None
        
        if job:
            try:
                post(f"/heartbeat/{NODE_NAME}", {"status": "running"})
            except Exception:
                pass
            cmd = job.get("cmd")
            if cmd:
                try:
                    print("Running received job:", cmd)
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print("Job failed:", e)
                except Exception as e:
                    print("Error running job:", e)
            try:
                post(f"/heartbeat/{NODE_NAME}", {"status": "idle"})
            except Exception:
                pass
        
        time.sleep(5)


def download_shard() -> str:
    """Download assigned shard. Returns local path."""
    url = f"{ORCH_ENDPOINT}/shards/{NODE_NAME}"
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        abs_data_dir = os.path.abspath(DATA_DIR)
        os.makedirs(abs_data_dir, exist_ok=True)
        local_path = os.path.join(abs_data_dir, f"shard_{NODE_NAME}.parquet")
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[agent] Shard downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"[agent] Error downloading shard: {e}")
        return None


def run_local_training(shard_path: str, epochs: int) -> bool:
    """Run local training on the shard. Returns True on success."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(current_dir, "train.py")
    
    cmd = [
        sys.executable, train_script,
        "--shard_path", shard_path,
        "--orchestrator_url", ORCH_ENDPOINT,
        "--node_name", NODE_NAME,
        "--epochs", str(epochs),
        "--batch_size", os.getenv("TRAIN_BATCH_SIZE", "32"),
    ]
    
    try:
        print(f"[agent] Running training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[agent] Training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[agent] Training failed: {e}")
        return False
    except Exception as e:
        print(f"[agent] Error during training: {e}")
        return False


def upload_weights(weights_path: str) -> dict:
    """Upload trained weights to the orchestrator."""
    url = f"{ORCH_ENDPOINT}/weights/upload/{NODE_NAME}"
    try:
        with open(weights_path, "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=120)
        return response.json()
    except Exception as e:
        print(f"[agent] Error uploading weights: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    main()
