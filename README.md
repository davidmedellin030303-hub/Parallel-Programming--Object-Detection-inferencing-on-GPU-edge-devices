# Parallel-Programming--Object-Detection-inferencing-on-GPU-edge-devices
Team: Edgar David Medellin Flores &amp; Viridiana Espiritu Sanchez 
# Object Detection Workflow on GPU Edge Device

##  Project Overview
This project aims to develop and build a workflow for real-time object identification on a GPU-accelerated embedded system. The goal is to show that AI inference can be performed efficiently on edge devices utilizing parallel computing approaches.

Object detection involves detecting and locating items in an image or video stream. Because this task requires significant computational power, GPU acceleration optimization methods are used to achieve real-time performance.

---

## Why GPU Acceleration?
Artificial intelligence inference is a parallel computation challenge because:

--> Convolutional neural networks (CNNs) carry out matrix computations. Meaning That:
- These operations can be performed simultaneously.
- GPUs process thousands of threads simultaneously.

Also, as it has been known, GPU acceleration significantly improves inference speed compared to CPUs.

---

## Hardware Selection
- NVIDIA Jetson Nano (embedded GPU platform)
- USB Camera

The Jetson Nano was selected by the professor; however, the reason why Jetson Nano is a good fit for this course is due to the fact that it supports CUDA and its suitability for parallel embedded AI applications. 

---

## Software Frameworks & Tools
- OpenCV (Image Processing)
- CUDA (GPU Acceleration) 
- Pytorch (Library)

---

## Pre-trained Model
We selected **YOLOv5** for object detection because:
- High accuracy
- Fast inference speed
- Optimized for real-time applications

---

## System Workflow

### 📊 Block Diagram
Camera → Image Capture → Preprocessing → Model Inference (GPU) → Post-processing → Display Output

---

## Implementation Steps

### 1. Environment Setup
The development environment was prepared beforehand on the Scaffold Assignment. The required software tools included:
- NVIDIA Jetson Nano Setup
- Cuda Environment implemented
- Download Pytorch libraries
- 

### 2. Load Pre-trained Model
- Download **POR DEFINIR** from GitHub
- Load trained weights into the model

### 3. Capture Live Video
A live camera feed was captured using OpenCV in C++. The camera continuously reads frames from the default video input device.
```cpp
cv::VideoCapture cap(0);
if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
}
