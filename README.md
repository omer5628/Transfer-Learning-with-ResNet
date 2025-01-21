# HW4 Report: Transfer Learning with ResNet18

## **Overview**
This project explores the use of transfer learning with a pre-trained ResNet18 model on the Fashion-MNIST dataset. The goal is to leverage the power of pre-trained models for image classification and compare their performance to a custom Convolutional Neural Network (CNN) designed and trained from scratch.

---

## **1. Methodology**

### Dataset: Fashion-MNIST
- **Details:** 70,000 grayscale images across 10 categories.
- **Preprocessing:**
  - Images resized to 224x224 pixels.
  - Converted to RGB using grayscale-to-RGB transformation.
  - Normalized using ImageNet mean and standard deviation.
  - Augmentation techniques: random horizontal flips and rotations.

### Model Implementation:
- **Custom CNN:**
  - Designed with convolutional layers, max-pooling, dropout, and fully connected layers.
  - Trained from scratch on the dataset.
- **ResNet18:**
  - A pre-trained model initially with frozen convolutional layers.
  - Fine-tuned by unfreezing layers after initial training.

---

## **2. Results and Key Comparisons**

| **Metric**         | **Custom CNN** | **ResNet18 (Transfer Learning)** |
|---------------------|----------------|---------------------------------|
| **Accuracy (%)**    | 91.84          | 93.52                          |
| **Precision**       | 0.919          | 0.937                          |
| **Recall**          | 0.916          | 0.935                          |
| **F1-Score**        | 0.919          | 0.935                          |
| **Training Time (Epochs)** | ~10 epochs  | ~5 epochs (FC only), ~10 epochs (fine-tuning) |

### Observations:
- **Accuracy:** ResNet18 outperformed the custom CNN, achieving a higher test accuracy of 93.52%.
- **Training Time:** The transfer learning approach of ResNet18 required fewer epochs to converge during the initial training phase.
- **Generalization:** ResNet18 exhibited better precision, recall, and F1-scores, especially in challenging categories.

---

## **3. Visualizations**

### Training and Validation Metrics
- **CNN Model:**
  - Rapid decline in training and validation losses, indicating faster convergence initially.
  - Some fluctuation in validation loss suggests potential overfitting.
- **ResNet18 Model:**
  - Slower initial decline in loss due to frozen layers.
  - Achieved better generalization with smoother validation loss after fine-tuning.

*(Graphs illustrating these trends will be provided.)*

### Confusion Matrices
- **Custom CNN:**
  - More prominent misclassifications in categories such as "Shirt" vs. "T-shirt/top."
- **ResNet18:**
  - Outperformed CNN in 9 out of 10 classes, with fewer misclassifications overall.

*(Confusion matrices will be included.)*

---

## **4. Conclusions on Transfer Learning Effectiveness**

### Advantages of Transfer Learning:
- **Efficiency:** Reduced training time by leveraging pre-trained weights for feature extraction.
- **Improved Accuracy:** Higher performance metrics compared to training a model from scratch.
- **Robust Generalization:** Better at distinguishing between visually similar classes.

### Limitations:
- Requires careful adjustment of learning rates during fine-tuning.
- Computational demands increase during fine-tuning compared to training only the fully connected layers.

### Scenarios Where Transfer Learning Excels:
- **Small Datasets:** Captures general patterns effectively with limited training data.
- **Time Constraints:** Faster results due to pre-trained feature extraction.
- **Complex Tasks:** Beneficial for datasets with intricate patterns, leveraging pre-trained knowledge.

### Summary of Pre-trained ResNet Impact:
- Enhanced accuracy, precision, recall, and F1-scores.
- Allowed faster convergence during the initial training phase.
- Achieved better generalization across challenging categories.

### Challenges:
- The initial training process lacked proper storage for loss metrics, requiring code adjustments and re-training.
- Fine-tuning required precise hyperparameter tuning to prevent overfitting.

---

## **5. Files Included**
- **`custom_cnn.ipynb`**: Code for training the custom CNN model.
- **`resnet18_transfer_learning.ipynb`**: Code for transfer learning using ResNet18.
- **`results/`**: Contains training logs, visualizations, and confusion matrices for both models.
- **`report.pdf`**: Detailed report summarizing the methodology, results, and conclusions.

---

## **6. How to Use**
1. Clone the repository and install the required dependencies listed in `requirements.txt`.
2. Run the Jupyter notebooks to reproduce the experiments:
   - Train the custom CNN using `custom_cnn.ipynb`.
   - Fine-tune ResNet18 using `resnet18_transfer_learning.ipynb`.
3. Refer to the `results/` folder for pre-generated metrics and visualizations.

---

## **7. References**
- Fashion-MNIST Dataset: [Link](https://github.com/zalandoresearch/fashion-mnist)
- ResNet18 Architecture: [Paper](https://arxiv.org/abs/1512.03385)

