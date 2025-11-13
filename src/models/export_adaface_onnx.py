import os
import torch
import numpy as np
from pathlib import Path

# uses your existing wrapper to load the PyTorch AdaFace model
from models.wrap_adaface import AdaFaceWrapper

def main():
    wrapper = AdaFaceWrapper(device="cpu")  # load weights on CPU temporarily
    model = wrapper.model.eval()

    dummy = torch.randn(1, 3, 112, 112)  # AdaFace input size

    export_name = Path(__file__).resolve().parent / "adaface.onnx"
    print(f"[EXPORT] Saving to: {export_name}")

    torch.onnx.export(
        model,
        dummy,
        str(export_name),
        input_names=["input"],
        output_names=["emb"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}},
    )

    print("[EXPORT] Done! ONNX model exported successfully.")

if __name__ == "__main__":
    main()
