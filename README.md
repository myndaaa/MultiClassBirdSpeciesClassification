# Bird Species Classification Project 🐦
Prepared for COS30018 Individual Assignment

## **[Link of Presentation Video](https://youtu.be/c3E2_yo-6MA)**
[Click here--> https://youtu.be/c3E2_yo-6MA](https://youtu.be/c3E2_yo-6MA)

## **[Link of Canva Slides](https://www.canva.com/design/DAGUvk9P4-o/_y20371B1BpErMBxTLBITg/edit?utm_content=DAGUvk9P4-o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**
[Click here](https://www.canva.com/design/DAGUvk9P4-o/_y20371B1BpErMBxTLBITg/edit?utm_content=DAGUvk9P4-o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## [Link of Dataset](https://drive.google.com/drive/folders/1_6Qx0lNwcuh9MbZAkimfsYRKef1thO5U?usp=drive_link)

## [LInk to hugging face interface](https://huggingface.co/spaces/Myndaaa/birdie)
### Project Overview
- **Objective**: Classify bird species from images in the **CUB-200** dataset using deep learning.
- **Dataset**: Caltech-UCSD Birds 200 (CUB-200) containing **200 bird species**.
- **Data Organization**: Images split into **train** and **test** directories with labels provided in `train.txt` and `test.txt`.

### Methodology
- **Baseline Model**: 
  - Model: **MobileNetV3** with basic configurations.
  - Purpose: Establish initial performance and demonstrate potential **overfitting**.
- **Improved Model**: 
  - Model: Enhanced **MobileNetV3** with:
    - **Data Augmentation** for diversity.
    - **L1 and L2 Regularization** to reduce overfitting.
    - **Transfer Learning** for leveraging pre-trained weights.
    - **5-Fold Cross-Validation** for robust performance estimation.
- **Potential Alternative Models**: Considering **EfficientNet**, **ResNet**, and **ResNet-50** for comparison.

### Key Features
- **Multi-Class Classification**: Classifies images into **200 unique bird species**.
- **Performance Goal**: Achieve better accuracy while mitigating overfitting.
- **Framework**: **TensorFlow/Keras** with compatibility adjustments for dependencies.

### Results and Findings
- Comparison of model performance between **baseline** and **enhanced** versions.
- Analysis of model behavior with various **regularization** and **augmentation** techniques.

### Usage
1. Clone the repository.
2. Open in Google collab
3. Run all cells

### Future recommendations
- Experiment with additional architectures and techniques.
- Optimize further for real-time inference.