# Parameters to Benchmark FaceNet, ArcFace, and MagFace

## Accuracy & Robustness Metrics
- Verification Accuracy - % of correct matches on datasets (LFW, CFP-FP, AgeDB)
- True Accept Rate (TAR) @ False Accept Rate (FAR) → biometric benchmark metric
- Equal Error Rate (EER) → point where false accepts = false rejects
- Occlusion (glasses, masks, partial face) - ???
- Low resolution / blur


## Demographic Fairness Metrics
- Age-based accuracy → report separately for younger (<30) vs older (>50) groups (AgeDB, UTKFace)
- Gender-based accuracy → report separately for male vs female
- Ethnicity-based accuracy → using datasets like RFW or UTKFace (e.g., Asian, Caucasian, African, Indian).
