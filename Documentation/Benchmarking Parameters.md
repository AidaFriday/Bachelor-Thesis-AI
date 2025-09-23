# Parameters to Benchmark FaceNet, ArcFace, and MagFace

## Accuracy & Robustness Metrics
- Verification Accuracy - % of correct matches on datasets (LFW, CFP-FP, AgeDB)
- True Accept Rate (TAR) @ False Accept Rate (FAR) → biometric benchmark metric
- Equal Error Rate (EER) - point where false accepts = false rejects
- Occlusion (glasses, masks, partial face) - ???
- Low resolution / blur


## Demographic Fairness Metrics
- Age-based accuracy - report separately for younger (<30) vs older (>50) groups (AgeDB, UTKFace)
- Gender-based accuracy - report separately for male vs female
- Ethnicity-based accuracy - using datasets like RFW or UTKFace (e.g., Asian, Caucasian, African, Indian)


## Efficiency / Runtime Metrics
- Inference Time per Image (ms) - average latency per image
- Frames per Second (FPS) - useful for real-time webcam tests
- Batch Size = 1 Latency - for true real-time evaluation

## Resource Usage
- Model Size (MB) - storage footprint of pretrained weights
- Memory Usage (RAM/VRAM) - peak during inference
- CPU vs GPU performance - performance drop on CPU-only setups

## Model Embedding Size
- Embedding Dimension - FaceNet (128), ArcFace (512), MagFace (512)
