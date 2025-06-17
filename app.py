from flask import Flask, request, jsonify, send_from_directory,render_template
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
import traceback

# Flask app initialization
app = Flask(__name__)

# Model configuration
OUT_CLASSES = 3  # The number of output classes
model_path = "resnet50_custom.pth"  # Path to the model you saved
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model initialization with ResNet50 and adjusting the final layer
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Replacing the final fully connected layer to match the output classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, OUT_CLASSES)

# **Debugging**: Print the model architecture after modification
print(f"Modified model architecture: {resnet}")

# Loading model weights
try:
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Move model to the appropriate device (GPU/CPU)
resnet = resnet.to(device)
resnet.eval()  # Set the model to evaluation mode

# Label mappings for prediction
index_label = {0: "dry", 1: "normal", 2: "oily"}

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image before converting to tensor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function based on the model's forward pass
def predict_image(image_file):
    print("Image file received:", image_file)
    try:
        # Open image from the file object
        img = Image.open(image_file).convert("RGB")
        
        # Apply the transformations
        img = transform(img)  # PIL image transformed directly to tensor
        
        # Add batch dimension and move to the correct device (GPU/CPU)
        img = img.unsqueeze(0).to(device)
        
        # Debug: Check the transformed image tensor shape
        print(f"Image tensor shape: {img.shape}")

        # Predict using the model
        with torch.no_grad():
            out = resnet(img)
            print(f"Model output (raw): {out}")  # Log raw output for debugging

            # Check if output has the expected shape (batch_size x OUT_CLASSES)
            if out.shape[1] != OUT_CLASSES:
                raise ValueError(f"Unexpected output shape: {out.shape}. Expected shape: [1, {OUT_CLASSES}]")

            # Get the predicted class index
            predicted_class = out.argmax(1).item()  # Get the index of the maximum output
            prediction = index_label[predicted_class]
            print(f"Prediction: {prediction}")
            return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise Exception(f"Error during image processing or prediction: {str(e)}")

# API endpoint for home page (index.html)
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), "templates/pred.html")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pred')
def pred():
    return render_template('pred.html')

@app.route('/products')
def products():
    return render_template('products.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dry')
def dry_products():
    return render_template('dry.html')

@app.route('/oily')
def oily_products():
    return render_template('oily.html')

@app.route('/Normal')
def normal_products():
    return render_template('Normal.html')




# API endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']  # Get the file object
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Perform prediction on the uploaded image
        prediction = predict_image(image_file)
        return jsonify({"prediction": prediction})  # Return the prediction

    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        print(f"Error: {error_message}")
        return jsonify({"error": error_message, "details": traceback.format_exc()}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
