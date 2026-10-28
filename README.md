<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Model-YOLOv8-purple" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Framework-Ultralytics-orange" alt="Ultralytics">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red" alt="Streamlit">
</p>

# Smart Shelf Inventory System
### Retail Product Detection Using YOLOv8


## Overview

**Smart Shelf Inventory System** is a computer vision project designed to automate retail product detection and visual shelf analysis using deep learning.

The system leverages a custom-trained YOLOv8 object detection model to identify and localize retail products in shelf images. By detecting multiple product categories within a single image, the project demonstrates how object detection can be applied to retail inventory monitoring and intelligent shelf management.

The project covers the computer vision development workflow, from dataset preparation and model training to inference and visualization of detection results.

## Problem Statement

Traditional retail inventory monitoring relies heavily on manual shelf inspection and product counting.

This process can be time-consuming, inefficient, and difficult to scale across large retail environments. In addition, products that are out of stock or misplaced may remain undetected until the next manual inspection.

The objective of this project is to demonstrate an AI-based approach to retail product detection that can support automated visual inventory analysis.

## Solution

A custom-trained YOLOv8 object detection model is used to analyze retail shelf images and identify products belonging to predefined categories.

For each detected product, the model generates:

- Product class label
- Bounding box coordinates
- Detection confidence score

These outputs provide a visual representation of the products detected in the input image and can serve as a foundation for further retail analytics and inventory management applications.

## Key Features

- **Multi-Class Product Detection:** Detects multiple retail product categories in shelf images.
- **Bounding Box Localization:** Identifies the approximate location of each detected product.
- **Confidence Score Estimation:** Provides confidence scores for model predictions.
- **Custom-Trained Model:** Uses a YOLOv8 model trained on retail product images.
- **Visual Inference:** Generates detection results with bounding boxes and class labels.
- **Retail Application:** Demonstrates the potential of computer vision for intelligent shelf monitoring.

## Dataset

The project uses a retail product image dataset containing approximately 9,000 images.

| Dataset Attribute | Details |
|---|---|
| Domain | Retail Product Detection |
| Total Images | 9,000 |
| Task | Object Detection |
| Annotation Format | YOLO |
| Annotation Type | Bounding Boxes |
| Number of Classes | 18 |

### Product Categories

The dataset contains the following 18 product classes:

`cake`, `candy`, `cereal`, `chips`, `chocolate`, `coffee`, `fish`, `honey`, `jam`, `milk`, `oil`, `pasta`, `rice`, `soda`, `sugar`, `tea`, `vinegar`, `water`

### Annotation Format

The dataset uses YOLO-compatible bounding box annotations.

Each annotation represents an object using:

- Class ID
- Normalized bounding box center coordinates
- Normalized bounding box width and height

## Model Development

The project uses the YOLOv8 object detection architecture from Ultralytics.

### Development Workflow

1. Dataset preparation and organization
2. Image annotation and bounding box labeling
3. Dataset configuration
4. Model training
5. Model evaluation
6. Object detection inference
7. Visualization of detection results

### Model Inference

The trained model processes an input image and predicts the locations and classes of retail products.

The inference output includes bounding boxes, class labels, and confidence scores, allowing the detected products to be visualized and analyzed.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Object detection architecture |
| Ultralytics | Model training and inference framework |
| PyTorch | Deep learning framework |
| OpenCV | Image processing and visualization |
| Streamlit | Interactive application interface |

## System Architecture

```text
       Retail Shelf Image
               |
               v
     Image Preprocessing
               |
               v
     Trained YOLOv8 Model
               |
               v
      Object Detection
               |
               v
   Product Class Predictions
               |
               v
     Bounding Box Results
               |
               v
    Visualized Detection Output
```

## Application

The trained model can be integrated into an interactive application for visual product detection.

Users can provide retail shelf images and inspect the model's predictions, including product labels and bounding boxes.

This provides a practical demonstration of deploying a computer vision model beyond the training environment.

## Potential Applications

The project demonstrates the potential of AI-based product detection in:

- Retail shelf monitoring
- Product identification
- Visual inventory inspection
- Automated product counting
- Shelf organization analysis
- Smart retail applications

## Future Enhancements

- Real-time product detection using video streams
- Automated product counting and inventory estimation
- Empty-shelf and low-stock detection
- Product placement and shelf organization analysis
- Integration with inventory management systems
- Model-serving API for external applications
- Retail analytics dashboard

## 📌 Info

| Platform | Link |
|---|---|
| GitHub | [Repository](https://github.com/ashish-wandhare/retail_classification) |
| Streamlit Cloud | [Live Demo](https://ashish-retailclassification.streamlit.app/) |
| AWS EC2 | [Live Demo](http://13.239.147.87:8503/) |


## 📸 Application Preview

### Main Application
<p align="center">
  <img width="100%" alt="Object Detection Main Page" src="https://github.com/user-attachments/assets/27838691-2b90-4c54-924a-b07f94e59372" />
</p>

### Application Interface
<p align="center">
  <img width="100%" alt="Main Page" src="https://github.com/user-attachments/assets/cb7de33f-0d40-4442-ae34-ef18663732be" />
</p>

### Product Detection
<p align="center">
  <img width="100%" alt="Product Detection" src="https://github.com/user-attachments/assets/473c558e-80de-4285-bb63-626d35f9241e" />
</p>

### Detection Results
<p align="center">
  <img width="100%" alt="Detection Result" src="https://github.com/user-attachments/assets/e10c40d2-e692-4ca5-b1e0-50bb818fa076" />
</p>

## 📸 Experimental results
<img width="1920" height="1920" alt="train_batch0" src="https://github.com/user-attachments/assets/e210cafb-8069-4978-9ac0-70e26fb57c4b" />
<img width="1920" height="1920" alt="train_batch1" src="https://github.com/user-attachments/assets/196c28f5-80ba-4ff1-90d3-6d8280ceaa8b" />
<img width="1920" height="1920" alt="train_batch2" src="https://github.com/user-attachments/assets/0253688b-d49b-4456-a36e-7eca3dd56285" />
<img width="1920" height="1920" alt="train_batch9000" src="https://github.com/user-attachments/assets/d376475c-edae-4351-b2f7-b77210530f9b" />
<img width="1920" height="1920" alt="train_batch9001" src="https://github.com/user-attachments/assets/337510e3-be72-44b6-b8bb-1e534d1bc53c" />
<img width="1920" height="1920" alt="train_batch9002" src="https://github.com/user-attachments/assets/9f57f90a-44b1-40a8-bae1-b017eb12e51b" />
<img width="1920" height="1920" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/b4161aba-c31e-4f73-b881-f4dc5e167d20" />
<img width="1920" height="1920" alt="val_batch0_pred" src="https://github.com/user-attachments/assets/f08a0c9e-75f8-413d-8c3f-26842b0732d7" />
<img width="1920" height="1920" alt="val_batch1_labels" src="https://github.com/user-attachments/assets/dc2e689c-cc78-45e3-9344-13a171398903" />
<img width="1920" height="1920" alt="val_batch1_pred" src="https://github.com/user-attachments/assets/7c388a01-e42e-478b-8c27-d0c52dd45e27" />
<img width="1920" height="1920" alt="val_batch2_labels" src="https://github.com/user-attachments/assets/b66d276e-e4aa-4dfd-a94b-7f5f7b208098" />
<img width="1920" height="1920" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/fb696c40-4cab-4c0b-a34f-954b14fe1d35" />

<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/d038cc3d-f0ff-4da3-8fa4-fde944de0aa7" />
<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/2ec7b13c-d936-4d4d-8508-aa44597deaed" />
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/35e5a0ad-6b1f-4333-a0f9-77e0d31d8a24" />
<img width="2250" height="1500" alt="BoxR_curve" src="https://github.com/user-attachments/assets/26964ac5-3035-45c9-933f-0f11007cfdb1" />
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1df159aa-2e4a-43e5-8c53-290f8493a49d" />
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/bee94e50-5997-40ed-a8b2-69097449aa93" />
<img width="1600" height="1600" alt="labels" src="https://github.com/user-attachments/assets/45ff9c64-6d89-49f1-b536-dbc7a886397e" />
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/35b018c6-0429-4f78-bf5b-4b249314d4aa" />



