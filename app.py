import gradio as gr
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# BloodScope - Sickle Cell Anemia Classification App
# BloodScope is an AI-powered diagnostic tool that helps classify blood smear images
# to detect the presence of sickle cell anemia. Using a deep learning model,
# BloodScope provides quick and reliable classification with high confidence.

# Load your trained model
model_path = "model1.pt"
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

# Class names and confidence threshold
class_names = {0: "Negative", 1: "Positive"}
CONFIDENCE_THRESHOLD = 0.90  # 90% threshold

# Prediction function for classification only
def predict_image(image):
    # Predict
    results = model.predict(image, imgsz=640)
    pred_class = results[0].probs.top1
    confidence = results[0].probs.data[pred_class].item()
    
    # Create annotated image
    img_array = np.array(image)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_array)
    
    # Show prediction and confidence
    pred_text = f"{class_names[pred_class]}\nConfidence: {confidence:.2%}"
    bbox_color = (0.1, 0.1, 0.1, 0.9)  # Dark gray for prediction
    text_color = '#ffffff'  # White text
    
    if confidence < CONFIDENCE_THRESHOLD:
        pred_text = f"Model couldn’t confidently identify\nthe image (Confidence: {confidence:.2%})"
        bbox_color = (0.5, 0.0, 0.0, 0.9)  # Dark red for uncertainty
        text_color = '#ffcccc'  # Light red text
    
    ax.text(20, 50, pred_text, 
            bbox=dict(facecolor=bbox_color, edgecolor='#ffffff', boxstyle='round,pad=0.5'),
            fontsize=14, color=text_color, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save to temp file
    temp_file = "temp_output.png"
    plt.savefig(temp_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    return temp_file, pred_text

# Custom CSS for a deep black aesthetic
css = """
body {
    background: #0d0d0d;
    font-family: 'Helvetica', 'Arial', sans-serif;
    color: #e0e0e0;
}
.gradio-container {
    max-width: 900px;
    margin: 30px auto;
    padding: 20px;
    background: #111111;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.7);
    border: 1px solid #1a1a1a;
}
h1 {
    color: #ffffff;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
    letter-spacing: 1.5px;
}
.upload-box {
    border: 2px dashed #333333;
    border-radius: 8px;
    padding: 20px;
    background: #0d0d0d;
    transition: border-color 0.3s ease;
}
.upload-box:hover {
    border-color: #555555;
}
button {
    background: #222222;
    color: #ffffff;
    border: 1px solid #444444;
    padding: 10px 25px;
    border-radius: 20px;
    font-size: 1.1em;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.3s ease, border-color 0.3s ease;
}
button:hover {
    background: #333333;
    border-color: #666666;
}
.output-box {
    border-radius: 8px;
    padding: 15px;
    background: #111111;
    border: 1px solid #222222;
    margin-top: 15px;
}
.footer {
    text-align: center;
    color: #666666;
    font-size: 0.85em;
    margin-top: 20px;
}
"""

# Gradio Interface
with gr.Blocks(title="BloodScope - Sickle Cell Anemia Classifier", css=css) as demo:
    gr.Markdown(
        """
        # BloodScope: AI-Powered Sickle Cell Anemia Detection
        Upload a blood smear image to classify it as Negative or Positive for sickle cell anemia.
        """
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=400):
            image_input = gr.Image(type="pil", label="Upload Test Image", elem_classes="upload-box")
            classify_btn = gr.Button("Classify")
        
        with gr.Column(scale=1, min_width=400):
            output_img = gr.Image(label="Result", elem_classes="output-box")
            output_text = gr.Textbox(label="Diagnosis", elem_classes="output-box", lines=2)
    
    classify_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[output_img, output_text]
    )
    
    gr.Markdown(
        """
        <p class='footer'>Powered by YOLOv11 | Built by xAI</p>
        """
    )

# Launch the app
demo.launch(inbrowser=True, share=True)