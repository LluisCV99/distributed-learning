"""
Local training script for federated learning with per-epoch synchronization.
Trains on a local data shard and synchronizes weights with the orchestrator after each epoch.

Features:
- Per-epoch weight synchronization via FedAvg
- Training and validation accuracy tracking
- Async straggler handling (continues on sync timeout)
"""
import argparse
import os
import io
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import polars as pl

# NOTE: if you change these, also update demo/.env
MODEL_HIDDEN_1 = int(os.getenv("MODEL_HIDDEN_1", "256"))
MODEL_HIDDEN_2 = int(os.getenv("MODEL_HIDDEN_2", "128"))
MODEL_HIDDEN_3 = int(os.getenv("MODEL_HIDDEN_3", "64"))
MODEL_DROPOUT_1 = float(os.getenv("MODEL_DROPOUT_1", "0.3"))
MODEL_DROPOUT_2 = float(os.getenv("MODEL_DROPOUT_2", "0.2"))

VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))


def create_model(input_dim: int = 784, num_classes: int = 10) -> nn.Module:
    """Create the MNIST classification model. Architecture is configurable via env vars."""
    return nn.Sequential(
        nn.Linear(input_dim, MODEL_HIDDEN_1),
        nn.BatchNorm1d(MODEL_HIDDEN_1),
        nn.ReLU(),
        nn.Dropout(MODEL_DROPOUT_1),
        nn.Linear(MODEL_HIDDEN_1, MODEL_HIDDEN_2),
        nn.BatchNorm1d(MODEL_HIDDEN_2),
        nn.ReLU(),
        nn.Dropout(MODEL_DROPOUT_2),
        nn.Linear(MODEL_HIDDEN_2, MODEL_HIDDEN_3),
        nn.ReLU(),
        nn.Linear(MODEL_HIDDEN_3, num_classes),
    )


def load_dataset(path: str):
    """Load a CSV or Parquet dataset and return train/val TensorDatasets."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}.")
    if path.endswith(".csv"):
        df = pl.read_csv(path)
    else:
        df = pl.read_parquet(path)
    
    # Last column is assumed to be the target
    target_col = df.columns[-1]
    features = df.drop(target_col)
    labels = df[target_col]
    x = torch.tensor(features.to_numpy(), dtype=torch.float32)
    y = torch.tensor(labels.to_numpy(), dtype=torch.long)
    
    full_dataset = TensorDataset(x, y)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None
    
    num_classes = 10
    return train_dataset, val_dataset, x.shape[1], num_classes


def upload_epoch_weights(orchestrator_url: str, node_name: str, epoch: int, 
                         model: nn.Module, loss: float, accuracy: float, samples: int,
                         epoch_time_sec: float = 0.0) -> dict:
    """Upload epoch weights and metrics to the orchestrator for FedAvg."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    url = f"{orchestrator_url}/sync/epoch"
    try:
        response = requests.post(
            url,
            data={
                "node_name": node_name,
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "samples": samples,
                "epoch_time_sec": epoch_time_sec,
            },
            files={"file": ("weights.pt", buffer, "application/octet-stream")},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        print(f"[train] Error uploading epoch weights: {e}")
        return {"status": "error", "error": str(e)}


def report_epoch_metrics(orchestrator_url: str, node_name: str, epoch: int,
                         loss: float, accuracy: float, val_loss: float, val_accuracy: float,
                         samples: int, duration_sec: float) -> dict:
    """Report epoch metrics to the orchestrator's monitoring endpoint."""
    url = f"{orchestrator_url}/agent/epoch"
    try:
        response = requests.post(url, json={
            "node_name": node_name,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "samples_processed": samples,
            "epoch_duration_sec": duration_sec,
        }, timeout=10)
        return response.json()
    except Exception as e:
        print(f"[train] Error reporting epoch metrics: {e}")
        return {"ok": False}


def download_averaged_weights(orchestrator_url: str) -> dict:
    """Download averaged weights from the orchestrator after epoch sync."""
    url = f"{orchestrator_url}/sync/weights"
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            weights = torch.load(io.BytesIO(response.content), map_location="cpu")
            return weights
        else:
            print(f"[train] No averaged weights yet: {response.status_code}")
            return None
    except Exception as e:
        print(f"[train] Error downloading averaged weights: {e}")
        return None


def wait_for_epoch_sync(orchestrator_url: str, epoch: int, max_wait: int = 120) -> bool:
    """
    Wait until all nodes have synchronized for this epoch.
    
    If the timeout is reached, training continues anyway (straggler handling).
    This prevents slow nodes from blocking the entire training process.
    """
    import time
    url = f"{orchestrator_url}/sync/status"
    
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = requests.get(url, timeout=10)
            status = response.json()
            
            if status.get("current_epoch", 0) >= epoch:
                return True
            
            received = status.get("received_this_epoch", [])
            expected = status.get("expected_nodes", [])
            print(f"[train] Waiting for sync epoch {epoch}: {len(received)}/{len(expected)}")
            
        except Exception as e:
            print(f"[train] Error checking sync status: {e}")
        
        time.sleep(2)
    
    print(f"[train] WARNING: Sync timeout epoch {epoch} - continuing as straggler (async mode)")
    return False


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    """Evaluate the model on the validation set. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += inputs.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Local training with per-epoch sync.")
    parser.add_argument("--shard_path", type=str, required=True, 
                        help="Path to the local shard file (CSV or Parquet)")
    parser.add_argument("--orchestrator_url", type=str, required=True,
                        help="Orchestrator URL for synchronization")
    parser.add_argument("--node_name", type=str, required=True,
                        help="Name of this node")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="Numerical precision: fp32 (default) or fp16")
    parser.add_argument("--sync_timeout", type=int, default=120,
                        help="Max seconds to wait for epoch sync (straggler timeout)")
    args = parser.parse_args()

    print(f"[train] Loading shard from {args.shard_path}")
    
    train_dataset, val_dataset, input_dim, output_dim = load_dataset(args.shard_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None
    
    print(f"[train] Dataset: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")
    print(f"[train] Model: 784 -> {MODEL_HIDDEN_1} -> {MODEL_HIDDEN_2} -> {MODEL_HIDDEN_3} -> 10")

    model = create_model(input_dim, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    model.to(device)
    
    if args.precision == "fp16":
        model = model.half()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    import time as time_module
    
    training_start = time_module.time()
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time_module.time()
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        num_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if args.precision == "fp16":
                inputs = inputs.half()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == targets).sum().item()
            
            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        epoch_duration = time_module.time() - epoch_start
        train_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        train_accuracy = epoch_correct / num_samples if num_samples > 0 else 0.0
        
        val_loss, val_accuracy = 0.0, 0.0
        if val_loader:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        print(f"[train] Epoch {epoch}/{args.epochs} - "
              f"Train: loss={train_loss:.4f} acc={train_accuracy:.2%} | "
              f"Val: loss={val_loss:.4f} acc={val_accuracy:.2%} ({epoch_duration:.1f}s)")
        
        report_epoch_metrics(
            args.orchestrator_url,
            args.node_name,
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            num_samples,
            epoch_duration,
        )
        
        epoch_times.append(epoch_duration)
        
        sync_result = upload_epoch_weights(
            args.orchestrator_url, 
            args.node_name, 
            epoch, 
            model, 
            train_loss,
            train_accuracy,
            num_samples,
            epoch_time_sec=epoch_duration,
        )
        print(f"[train] Sync result: {sync_result.get('status', 'unknown')}")
        
        # Wait for all nodes to sync (with straggler handling)
        sync_complete = wait_for_epoch_sync(args.orchestrator_url, epoch, args.sync_timeout)
        
        # Download averaged weights (skip on last epoch)
        if epoch < args.epochs:
            if sync_complete:
                averaged = download_averaged_weights(args.orchestrator_url)
                if averaged:
                    model.load_state_dict(averaged)
                    print(f"[train] Loaded averaged weights for epoch {epoch + 1}")
                else:
                    print(f"[train] No averaged weights available, continuing with local weights")
            else:
                print(f"[train] Skipping weight download (straggler mode)")

    total_training_time = time_module.time() - training_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    print(f"\n{'='*60}")
    print(f"[train] TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Epochs trained:         {args.epochs}")
    print(f"  Total time:             {total_training_time:.2f} seconds")
    print(f"  Avg time per epoch:     {avg_epoch_time:.2f} seconds")
    print(f"  Time per epoch:         {[f'{t:.2f}s' for t in epoch_times]}")
    print(f"{'='*60}")
    print("[train] Training complete!")


if __name__ == "__main__":
    main()
