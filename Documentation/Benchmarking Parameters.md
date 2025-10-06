# Parameters to Benchmark FaceNet, ArcFace, and MagFace

## Accuracy  Metrics
- Verification Accuracy - Percentage of correctly verified pairs (same/different identity) on benchmark datasets like LFW, CFP-FP, AgeDB. It captures how well a model recognizes faces under normal conditions
- True Accept Rate (TAR), False Accept Rate (FAR) - biometric benchmark metric
  - TAR = proportion of genuine pairs correctly accepted
  - FAR = proportion of impostor pairs incorrectly accepted
  
- Equal Error Rate (EER) - point where false accepts = false rejects



## Demographic Fairness Metrics
- Age-based accuracy - report separately for younger (<30) vs older (>50) groups (Faces age differently, and models may bias toward one age group) (AgeDB)
- Gender-based accuracy - report separately for male vs female (if models generalize fairly or show accuracy gaps between genders)
- Ethnicity-based accuracy - using datasets like RFW or UTKFace (e.g., Asian, Caucasian, African, Indian)


## Efficiency / Runtime Metrics
- Inference Time per Image (ms) - average time (in milliseconds) to process a single image
- Frames per Second (FPS) - useful for real-time webcam tests
- Batch Size = 1 Latency - for true real-time evaluation - ???

## Resource Usage
- Model Size (MB) - the file size on disk of the model’s pretrained weights
- Memory Usage (RAM/VRAM) - peak memory consumption during inference. Affects whether models can run on low-resource devices
- CPU vs GPU performance - performance drop on CPU-only setups


