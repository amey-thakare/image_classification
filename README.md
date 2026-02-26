# image_classification

# Project Overview
This project implements an automated banana quality grading system using Convolutional Neural Networks (CNN). The model classifies banana images into three categories:

Grade_A
Grade_B
Grade_C

The system uses transfer learning with a pretrained ResNet18 architecture and includes proper training, validation, and test evaluation following research-grade methodology.

# Key Features
 Transfer Learning using Pretrained ResNet18

 Data Augmentation for better generalization

 Train / Validation / Test Split (70/15/15)

 Model Checkpointing (Best Validation Model Saved)

 Classification Report (Precision, Recall, F1-score)

 Confusion Matrix Visualization

 Flask Web Deployment

# Model Architecture
Base Model: ResNet18 (Pretrained on ImageNet)

Final Layer Modified for 3-class classification

Loss Function: CrossEntropyLoss

Optimizer: Adam (lr = 0.0001)

# Dataset
The dataset is organized as:

dataset/
│
├── Grade_A/
├── Grade_B/
└── Grade_C/
Total Images: ~600
# Split:

70% Training

15% Validation

15% Test

# Final Performance (Test Set)
Test Accuracy: 95%

Macro F1-Score: 0.94

Balanced precision and recall across all classes

Confusion matrix analysis shows strong performance, with minor confusion between Grade_A and Grade_B.

# Experimental Improvements
To improve model generalization:

Applied data augmentation:

RandomHorizontalFlip

RandomRotation

ColorJitter

Used transfer learning instead of training from scratch

Monitored validation accuracy to prevent overfitting

🖥️ How to Run
1️⃣ Install Dependencies
pip install torch torchvision flask pillow scikit-learn seaborn matplotlib
2️⃣ Train the Model
python train.py
This will:

Train the model

Evaluate on validation & test sets

Save the best model as banana_model.pth

3️⃣ Run the Web Application
python app.py
Open in browser locally.

Upload an image to get prediction and confidence score.

📂 Project Structure
banana_project/
│
├── dataset/                # Image dataset (not uploaded to GitHub)
├── templates/
│   └── index.html
├── train.py                # Training pipeline
├── app.py                  # Flask deployment
├── banana_model.pth        # Saved model (ignored in .gitignore)
└── README.md
