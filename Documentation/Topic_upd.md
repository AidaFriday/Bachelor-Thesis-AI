# Topic: Benchmarking and Real-Time Evaluation of Pre-Trained Face Recognition Models  

## Goals  

### 1. Evaluate and Compare Pre-Trained Models  
- Assess the performance of widely used open-source face recognition models (e.g., FaceNet, SphereFace, VGG-Face, ArcFace, AdaFace) implemented in PyTorch.  

### 2. Benchmark on Standard Datasets  
- Test models on benchmark datasets (e.g., **LFW, CelebA, VGGFace2**) to measure accuracy, robustness, and generalization across different conditions.  

### 3. Measure Runtime and Efficiency Metrics  
- Benchmark models in terms of runtime performance (**latency, FPS**), memory usage, and (if available) power consumption during inference.  

### 4. Develop a Real-Time Benchmarking Application  
- Build a user-friendly **PyQt-based desktop application** that enables:  
  - Model selection  
  - Dataset/image upload  
  - Webcam-based real-time evaluation with live metric reporting  

---

## Plan  

### Phase 1 – Preparation  

#### Literature Review  
- Study existing face recognition models and their PyTorch implementations.  
- Review benchmarking methods and evaluation metrics (accuracy, ROC/AUC, F1, latency, throughput, memory usage, power consumption).  
- Research benchmark datasets commonly used for face recognition.  

#### Setup Development Environment  
- Install PyTorch and relevant libraries (`insightface`, `deepface`, `facenet-pytorch`).  
- Verify GPU/CPU capabilities for running pretrained models and collecting runtime statistics.  

---

### Phase 2 – Model Benchmarking  

#### Load Pre-Trained Models  
- Implement a unified interface (wrapper) for models like FaceNet, VGG-Face, ArcFace, AdaFace.  

#### Dataset Benchmarking  
- Run inference on benchmark datasets (e.g., **LFW, CelebA, VGGFace2**).  
- Collect accuracy and robustness metrics.  
- Record runtime-related performance: latency (ms/frame), throughput (FPS), memory footprint, and (optionally) power consumption.  

#### Analysis  
- Compare results across models.  
- Identify trade-offs between accuracy, robustness, and runtime efficiency.  

---

### Phase 3 – Application Development  

#### Design PyQt Application  
- Provide GUI features such as:  
  - Dropdown to select model  
  - Option to upload dataset image(s)  
  - Webcam-based input for real-time testing  
  - Live visualization of metrics (accuracy, FPS, latency, memory usage)  

#### Backend Integration  
- Connect PyTorch model inference with the PyQt GUI.  
- Implement live monitoring and metric reporting.  

#### Testing  
- Test application with multiple datasets and webcam scenarios.  
- Evaluate usability and performance.  

---

## Open Questions  
- Second Reviewer  
- Access to GPU resources  
- Weekly Meeting Time  
- Expectations for Weekly Meetings  

