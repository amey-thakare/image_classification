

---

#  Banana Quality Grading using CNN (ResNet18)

##  Project Overview

This project implements an automated **banana quality grading system** using **Convolutional Neural Networks (CNNs)**. The model classifies banana images into three categories:

* **Grade_A**
* **Grade_B**
* **Grade_C**

The system uses **Transfer Learning with a pretrained ResNet18 model** and follows a research-grade methodology including proper dataset splitting, validation monitoring, and performance evaluation.

---

##  Key Features

*  Transfer Learning using pretrained **ResNet18**
*  Data Augmentation for improved generalization
*  Train / Validation / Test Split (70 / 15 / 15)
*  Model Checkpointing (Best validation model saved)
*  Classification Report (Precision, Recall, F1-score)
*  Confusion Matrix Visualization
*  Flask-based Web Deployment
*  Confidence Score Output

---

##  Model Architecture

* **Base Model:** ResNet18 (Pretrained on ImageNet)
* **Final Layer:** Modified for 3-class classification
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate = 0.0001)

---

##  Dataset Structure

```
dataset/
│
├── Grade_A/
├── Grade_B/
└── Grade_C/
```

* Total Images: ~600
* Image Classes: 3
* Dataset Split:

  * 70% Training
  * 15% Validation
  * 15% Test

---

##  Final Performance (Test Set)

* **Test Accuracy:** 95%
* **Macro F1-Score:** 0.94
* Balanced precision and recall across all classes
* Minor confusion observed between Grade_A and Grade_B

---

##  Experimental Improvements

To improve model performance and generalization:

* Applied Data Augmentation:

  * RandomHorizontalFlip
  * RandomRotation
  * ColorJitter
* Used Transfer Learning instead of training from scratch
* Monitored validation accuracy to prevent overfitting
* Saved best-performing validation model

---

##  How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision flask pillow scikit-learn seaborn matplotlib
```

---

###  Train the Model

```bash
python train.py
```

This will:

* Train the CNN model
* Evaluate on validation & test sets
* Save the best model as `banana_model.pth`

---

### 3️ Run the Web Application

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000/
```

Upload a banana image to receive:

* Predicted Grade
* Confidence Score

---

##  Project Structure

```
banana_project/
│
├── dataset/                # Image dataset (not uploaded to GitHub)
├── templates/
│   └── index.html
├── train.py                # Training pipeline
├── app.py                  # Flask deployment
├── banana_model.pth        # Saved model (ignored in .gitignore)
└── README.md
```

---

