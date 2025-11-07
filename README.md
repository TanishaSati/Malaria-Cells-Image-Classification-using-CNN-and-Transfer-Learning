# Malaria-Cells-Image-Classification-using-CNN-and-Transfer-Learning
This project focuses on automated malaria detection from microscopic cell images using deep learning. The aim is to classify cell images as Parasitized or Uninfected efficiently and accurately, helping in early diagnosis and reducing manual workload for pathologists.
🔍 Overview

The project explores multiple Convolutional Neural Network (CNN) architectures and transfer learning techniques to detect malaria from blood smear images.
Both pre-trained models and a custom CNN (baseline model) are implemented and compared for performance.

🧬 Models Used

VGG19

ResNet50

DenseNet121

MobileNetV2

Custom CNN (from scratch)

Each model is fine-tuned on the NIH Malaria Cell Dataset, which contains thousands of labeled microscopic cell images.

⚙️ Methodology

Dataset Preprocessing – Images are resized, normalized, and augmented for better generalization.

Model Loading & Training – Pre-trained models are fine-tuned with transfer learning using ImageNet weights.

Optimization – Models trained using the Adam optimizer and categorical cross-entropy loss function.

Evaluation – Accuracy, loss curves, and confusion matrices are used for performance analysis.

📊 Results

The models are compared based on their accuracy, training time, and generalization capability.
Transfer learning models significantly outperform the custom CNN, highlighting the advantage of pre-trained feature extractors in biomedical image classification.

💾 Dataset

Source: NIH Malaria Cell Dataset

Classes: Parasitized, Uninfected

Image Count: ~27,000 cell images

🧩 Technologies Used

Python

PyTorch / TensorFlow

NumPy, Pandas, Matplotlib, Seaborn

Google Colab / Jupyter Notebook

🏁 Objective

To develop an efficient and accurate model for malaria cell classification using CNN and transfer learning, demonstrating how deep learning can aid in medical image analysis.
