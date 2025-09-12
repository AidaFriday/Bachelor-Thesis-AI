import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from insightface.app import FaceAnalysis


# ---------------- Encoders ----------------
class FaceNetEncoder:
    name = "facenet"
    input_size = (160, 160)

    def __init__(self, device: torch.device):
        from facenet_pytorch import InceptionResnetV1
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            v = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        return v


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class IResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        block = IBasicBlock
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512, bias=False),
            nn.BatchNorm1d(512)
        )
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = self.output_layer(x)
        return x


def build_iresnet50():
    return IResNet([3, 4, 14, 3])


class SphereFaceEncoder:
    name = "sphereface"
    input_size = (112, 112)

    def __init__(self, device: torch.device, ckpt_path: str):
        self.device = device
        self.model = build_iresnet50().to(device).eval()
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd["state_dict"].items()}
        self.model.load_state_dict(sd, strict=False)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            feat = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-12)
        return feat


# ---------------- Model paths ----------------
MODEL_PATHS = {
    "arcface": "C:/programming/FaceRec_Models/buffalo_l",
    "sphereface": "/absolute/path/to/sphereface.pth"
}


def select_model(choice: str, device: torch.device):
    ctx_id = 0 if device.type == "cuda" else -1

    if choice == "1":
        buffalo_path = MODEL_PATHS["arcface"]
        if not os.path.isdir(buffalo_path):
            raise RuntimeError(f"ArcFace pack not found at {buffalo_path}")
        app = FaceAnalysis(root=os.path.dirname(buffalo_path), name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app, None, (112, 112)

    elif choice == "2":
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        encoder = FaceNetEncoder(device=device)
        return app, encoder, (160, 160)

    elif choice == "3":
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        ckpt_path = MODEL_PATHS["sphereface"]
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"SphereFace checkpoint not found at {ckpt_path}")
        encoder = SphereFaceEncoder(device=device, ckpt_path=ckpt_path)
        return app, encoder, (112, 112)

    else:
        print("Invalid choice, defaulting to ArcFace (buffalo_l).")
        buffalo_path = MODEL_PATHS["arcface"]
        app = FaceAnalysis(root=os.path.dirname(buffalo_path), name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app, None, (112, 112)
