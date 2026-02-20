# Distributed Training System with Federated Learning

Bachelor's Thesis (TFG) project — Universitat de Lleida.

This system trains neural networks across multiple machines using **Federated Averaging (FedAvg)**, a technique that allows collaborative model training without sharing raw data between nodes. Each node trains on its local data partition and only shares model weights with a central orchestrator, which aggregates them after every epoch.

## Architecture

The system has three components:

| Component | Role | Port |
|-----------|------|------|
| **Orchestrator** | Central server that coordinates training, distributes data shards, and aggregates weights | 11999 |
| **Agent** | Worker node that trains locally on its assigned shard | - |
| **Demo** | Web UI to validate the trained model by drawing digits | 7860 |

```
                        +------------------+
                        |   Orchestrator   |
                        |    (FastAPI)     |
                        +--------+---------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
    +-------v-------+    +-------v-------+    +-------v-------+
    |    Agent 1    |    |    Agent 2    |    |    Agent N    |
    |   (PyTorch)   |    |   (PyTorch)   |    |   (PyTorch)   |
    +---------------+    +---------------+    +---------------+
```

## How Federated Averaging Works

Based on: McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.

1. The orchestrator splits the dataset into **shards** (with configurable overlap, default 15%) and assigns them proportionally to each node's compute capacity.
2. Each agent downloads its shard and trains **locally** for one epoch.
3. After each epoch, agents upload their model weights to the orchestrator.
4. The orchestrator computes `w_global = (1/n) * sum(w_i)` — the average of all received weights.
5. Agents download the averaged weights and continue training.
6. If a node is too slow (straggler), the orchestrator performs a partial average after a configurable timeout (default 120s) so the rest of the cluster is not blocked.

This process repeats for the configured number of epochs. The final averaged model is saved to disk.

## Getting Started

### Prerequisites

- Docker (or Python 3.12 with pip)
- NVIDIA GPU + CUDA 12.1 drivers (only for GPU agents)

### Build Docker Images

```bash
# CPU-only images
docker build -t tfg-orchestrator ./orchestrator
docker build -t tfg-agent ./agent
docker build -t tfg-demo ./demo

# GPU agent (CUDA 12.1)
docker build -t tfg-agent-gpu -f ./agent/Dockerfile.gpu ./agent
```

### Run

```bash
# 1. Start the orchestrator (central machine)
docker run -d -p 11999:11999 \
  -v /path/to/datasets:/datasets \
  -v /path/to/models:/models \
  tfg-orchestrator

# 2. Start agents (worker machines)
# CPU agent
docker run -d \
  -e ORCHESTRATOR_URL=http://<orchestrator-ip>:11999 \
  tfg-agent

# GPU agent
docker run -d --gpus all \
  -e ORCHESTRATOR_URL=http://<orchestrator-ip>:11999 \
  tfg-agent-gpu

# 3. Start the demo (for inference after training)
docker run -d -p 7860:7860 \
  -v /path/to/models:/models \
  tfg-demo
```

### Training

Once the orchestrator and at least one agent are running:

```bash
# Prepare the MNIST dataset
curl -X POST http://localhost:11999/datasets/prepare-mnist

# Start federated training (5 epochs)
curl -X POST "http://localhost:11999/train/mnist?epochs=5"
```

You can monitor training through:
- **Dashboard**: http://localhost:11999/
- **W&B**: https://wandb.ai/ (if `WANDB_API_KEY` is set)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus-compatible metrics |
| `/nodes` | GET | List active nodes |
| `/datasets` | GET | List available datasets |
| `/train/{dataset_id}` | POST | Start federated training |
| `/train/{dataset_id}/prepare` | POST | Prepare shards for training |
| `/sync/status` | GET | Epoch synchronization status |
| `/sync/weights` | GET | Download averaged weights |
| `/models` | GET | List saved models |
| `/monitor/overview` | GET | Training overview with metrics |
| `/monitor/nodes` | GET | Per-node monitoring details |
| `/monitor/timeline` | GET | Event timeline |

## Configuration

### Environment Variables

| Variable | Component | Description |
|----------|-----------|-------------|
| `ORCHESTRATOR_URL` | Agent | Orchestrator URL (supports multiple comma-separated URLs) |
| `NODE_ID` / `NODE_NAME` | Agent | Custom node identifier |
| `WANDB_API_KEY` | All | Weights & Biases API key for logging |
| `EPOCH_SYNC_TIMEOUT` | Orchestrator | Seconds before partial FedAvg on straggler timeout (default: 120) |
| `VALIDATION_SPLIT` | Agent | Fraction of shard used for validation (default: 0.1) |
| `DEFAULT_EPOCHS` | Orchestrator | Default number of training epochs (default: 5) |

### Model Architecture

The neural network architecture is configurable via environment variables. These must match between the agent and the demo service.

| Variable | Default |
|----------|---------|
| `MODEL_HIDDEN_1` | 256 |
| `MODEL_HIDDEN_2` | 128 |
| `MODEL_HIDDEN_3` | 64 |
| `MODEL_DROPOUT_1` | 0.3 |
| `MODEL_DROPOUT_2` | 0.2 |

## Project Structure

```
tfg-distributed-ai/
├── orchestrator/
│   ├── main.py              # FastAPI server, FedAvg logic, monitoring
│   ├── templates/
│   │   └── dashboard.html   # Web dashboard
│   └── Dockerfile
├── agent/
│   ├── agent.py             # Node registration, orchestrator discovery, training loop
│   ├── train.py             # Local training with per-epoch sync
│   ├── Dockerfile           # CPU build
│   └── Dockerfile.gpu       # GPU build (CUDA 12.1)
├── demo/
│   ├── app.py               # Gradio inference UI
│   └── Dockerfile
└── README.md
```

## License

MIT License
