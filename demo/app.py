import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
import glob

MODELS_DIR = os.getenv("MODELS_DIR", "/models")
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

# NOTE: if you change these, also update agent/.env
MODEL_HIDDEN_1 = int(os.getenv("MODEL_HIDDEN_1", "256"))
MODEL_HIDDEN_2 = int(os.getenv("MODEL_HIDDEN_2", "128"))
MODEL_HIDDEN_3 = int(os.getenv("MODEL_HIDDEN_3", "64"))
MODEL_DROPOUT_1 = float(os.getenv("MODEL_DROPOUT_1", "0.3"))
MODEL_DROPOUT_2 = float(os.getenv("MODEL_DROPOUT_2", "0.2"))


def create_model(input_dim: int = 784, num_classes: int = 10) -> nn.Module:
    """Create the MNIST classification model. Architecture: 784 -> 256 -> 128 -> 64 -> 10 (configurable via env vars)."""
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


def get_model_choices():
    """List .pt files in the models directory, sorted by modification date (newest first)."""
    path = Path(MODELS_DIR)
    if not path.exists():
        return []
    models = glob.glob(str(path / "*.pt"))
    models.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(m) for m in models]


def preprocess_drawing(image_data) -> torch.Tensor:
    """Preprocess canvas drawing to match MNIST training format."""
    if image_data is None:
        return None
    
    # Handle ImageEditor dict format
    if isinstance(image_data, dict):
        if 'composite' in image_data and image_data['composite'] is not None:
            image = image_data['composite']
        elif 'background' in image_data and image_data['background'] is not None:
            image = image_data['background']
        else:
            return None
    else:
        image = image_data
    
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    
    gray = pil_image.convert('L')
    resized = gray.resize((28, 28), Image.Resampling.LANCZOS)
    pixel_array = np.array(resized, dtype=np.float32)
    
    # MNIST expects white digit on black background
    if pixel_array.mean() > 127:
        pixel_array = 255.0 - pixel_array
    
    normalized = pixel_array / 255.0
    tensor = torch.tensor(normalized).view(1, -1)
    
    return tensor


def create_interface() -> gr.Blocks:
    """Create the Gradio web interface."""
    
    def predict(local_model_name, uploaded_model, drawing):
        """Run prediction with the selected or uploaded model on the drawing."""
        
        model_path = None
        
        if local_model_name:
            model_path = os.path.join(MODELS_DIR, local_model_name)
        elif uploaded_model is not None:
            model_path = uploaded_model.name
        
        if model_path is None:
            return "WARNING: Please select a model from the list or upload one (.pt)", None
        
        try:
            model = create_model()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            return f"ERROR: Could not load model: {e}", None
        
        tensor = preprocess_drawing(drawing)
        if tensor is None:
            return "WARNING: Please draw a digit", None
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        predicted = probs.argmax().item()
        confidence = probs[predicted].item()
        all_probs = {str(i): float(probs[i]) for i in range(10)}
        
        result = f"## Prediction: **{predicted}**\nConfidence: **{confidence:.1%}**"
        return result, all_probs
    
    arch_str = f"{MODEL_HIDDEN_1} -> {MODEL_HIDDEN_2} -> {MODEL_HIDDEN_3}"
    
    with gr.Blocks(title="MNIST Demo") as interface:
        gr.Markdown(f"""
        # MNIST Digit Recognition Demo
        
        **Distributed Federated Learning Validation**
        
        1. Upload or select a model file (.pt)
        2. Draw a digit (0-9) with the **black brush** on **white background**
        3. Click **Predict**
        
        *Model: 784 -> {arch_str} -> 10*
        """)
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    local_model = gr.Dropdown(
                        choices=get_model_choices(),
                        label="Select Model from /models",
                        info="Models found in the mounted directory",
                        scale=3
                    )
                    refresh_btn = gr.Button("Refresh", scale=1)
                
                model_file = gr.File(
                    label="Or upload a model manually (.pt)",
                    file_types=[".pt"],
                )
                
                canvas = gr.ImageEditor(
                    label="Draw a digit here (black brush)",
                    type="numpy",
                    image_mode="RGB",
                    height=350,
                    width=350,
                    brush=gr.Brush(
                        colors=["#000000"],
                        default_size=18,
                        color_mode="fixed"
                    ),
                    eraser=gr.Eraser(default_size=20),
                    canvas_size=(280, 280),
                )
                
                predict_btn = gr.Button("Predict", variant="primary", size="lg")
            
            with gr.Column():
                result = gr.Markdown("*Upload a model, draw a digit, and click Predict*")
                probs_chart = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=10,
                )
        
        gr.Markdown(f"""
        ---
        **Model:** 784 -> {arch_str} -> 10 | **Training:** Federated Averaging
        """)
        
        def refresh_models():
            return gr.update(choices=get_model_choices())
            
        refresh_btn.click(refresh_models, outputs=local_model)
        predict_btn.click(predict, [local_model, model_file, canvas], [result, probs_chart])
    
    return interface


def main():
    print("=" * 60)
    print("MNIST Demo Service")
    print("=" * 60)
    
    interface = create_interface()
    
    print(f"\n[demo] Starting at http://localhost:{GRADIO_SERVER_PORT}")
    print("[demo] Press Ctrl+C to stop\n")
    
    interface.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
    )


if __name__ == "__main__":
    main()
