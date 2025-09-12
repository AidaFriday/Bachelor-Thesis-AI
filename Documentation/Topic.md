# Bachelor Thesis

## Topic: Benchmarking and Real-Time Analysis of Face Recognition Models

### Goals:
1. **Evaluate and Compare Models**  
   Assess the performance of well-known open-source face recognition models (e.g., FaceNet, SphereFace, VGG-Face) implemented in PyTorch.

2. **Benchmark on Standard Datasets**  
   Conduct benchmarking experiments on widely used datasets to measure accuracy and generalization across different conditions.

3. **Develop a Runtime Benchmarking Application (Open Suggestion)**
   1. Build a user-friendly application for benchmarking pretrained models in real time.  
   2. Framework of choice: **PyQt (Python-based)**, selected for its flexibility, native desktop support, and ability to integrate seamlessly with the PyTorch backend.

---

## Plan:

### Phase 1 – Preparation

**Literature Review**  
- Study existing face recognition models
- Review benchmarking methods and evaluation metrics (accuracy, ROC, F1, latency).  
- Research PyTorch implementations and datasets

**Setup Development Environment**  
- Install PyTorch, Python libraries (insightface, deepface, facenet-pytorch).  
- Test GPU/CPU capabilities for running pretrained models.  

---

### Phase 2 – Model Benchmarking

**Load Pretrained Models**  
- Implement wrappers for FaceNet, VGG-Face, ArcFace, AdaFace.  
- Ensure a unified interface for all models.  

**Dataset Benchmarking**  
- Run inference on datasets
- Collect metrics: accuracy and so on  
- Optional Record runtime performance: latency (ms/frame), throughput (FPS), memory usage.  

**Analysis**  
- Compare results across models.  
- Identify trade-offs between accuracy vs. real-time performance.  

---

### Phase 3 – Application Development

**Design PyQt Application**  
GUI with options to:  
- Select model (dropdown).  
- Upload dataset image / enable webcam.  
- Run real-time benchmarking.  

**Backend Integration**  
- Connect PyTorch model inference with GUI.  
- Implement metrics display (accuracy, FPS, latency).  

**Testing**  
- Test application with multiple datasets and live webcam.

**Open Questions**
- 2nd Reviewer
- GPU Access
- Weekly Meetings Time
- Expectation for the weekly meeeting
