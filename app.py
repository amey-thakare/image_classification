from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("banana_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ['Grade_A', 'Grade_B', 'Grade_C']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    if file.filename == "":
        return render_template("index.html")

    image = Image.open(file.stream).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = classes[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence_score
    )

if __name__ == "__main__":
    app.run(debug=True)