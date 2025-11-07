# 🧠 Malaria Cells Image Classification using CNN and Transfer Learning  

This repository contains my research project on **Malaria Detection from Microscopic Cell Images** using **Convolutional Neural Networks (CNN)** and **Transfer Learning**.  
The goal is to automatically classify cell images as **Parasitized** or **Uninfected**, reducing manual diagnosis effort and improving accuracy.  

---

## 📌 Project Overview  
This project implements and compares multiple deep learning architectures for malaria detection.  
Both **custom CNN (from scratch)** and **pre-trained transfer learning models** are fine-tuned on the **NIH Malaria Cell Dataset** for classification tasks.  

---

## 🧬 Models Implemented  

| **Model** | **Type** | **Key Features** |
|------------|-----------|------------------|
| **VGG19** | Transfer Learning | Deep layered CNN with uniform architecture |
| **ResNet50** | Transfer Learning | Skip connections to avoid vanishing gradients |
| **DenseNet121** | Transfer Learning | Dense connectivity for feature reuse |
| **MobileNetV2** | Transfer Learning | Lightweight with depthwise separable convolutions |
| **Custom CNN** | From Scratch | Baseline CNN model for performance comparison |

---

## ⚙️ Methodology  

1. **Data Preprocessing**  
   - Image resizing and normalization  
   - Data augmentation (rotation, zoom, flip, etc.)  

2. **Model Training**  
   - Transfer learning using ImageNet pre-trained weights  
   - **Optimizer:** Adam  
   - **Loss Function:** Categorical Cross-Entropy  
   - **Evaluation Metrics:** Accuracy, Loss, Confusion Matrix  

3. **Evaluation**  
   - Comparison of model accuracies  
   - Visualization of training & validation curves  
   - Analysis of misclassified samples  

---

## 📊 Dataset  

- **Dataset:** [NIH Malaria Cell Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)  
- **Classes:** Parasitized, Uninfected  
- **Total Images:** ~27,000 cell images  

---

## 💻 Tech Stack  

- **Language:** Python  
- **Frameworks:** PyTorch / TensorFlow  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Torchvision  
- **Platform:** Google Colab / Jupyter Notebook  

---

## 📈 Results Summary  

- Transfer learning models, especially **DenseNet121** and **ResNet50**, achieved the highest accuracy.  
- The project shows that **transfer learning significantly improves performance** compared to a scratch CNN model.  

---

## 🏁 Objective  

To develop an **efficient and accurate deep learning model** for classifying malaria-infected cells, demonstrating how CNNs and transfer learning can assist in automated medical image diagnosis.  

---


## 🚀 Future Scope  

- Deploy the best-performing model as a **web app** (Flask / Streamlit)  
- Extend dataset for **multi-class medical image classification**  
- Explore **Vision Transformers (ViT)** for advanced feature learning  

---

## 🤝 Acknowledgements  

- **NIH** for providing the malaria dataset  
- **PyTorch / TensorFlow** communities for open-source frameworks  
- **Researchers and mentors** who contributed to CNN architecture advancements  

---
