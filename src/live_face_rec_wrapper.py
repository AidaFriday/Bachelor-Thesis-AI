import os, time
from collections import deque

import cv2
import numpy as np
import torch

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insight_model

# ------------------ Encoders ------------------

class FaceNetEncoder:
    name = "facenet"
    input_size = (160, 160)
    def __init__(self, device: str):
        from facenet_pytorch import InceptionResnetV1
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        t = (t - 0.5)/0.5
        with torch.inference_mode():
            v = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        return v


class ArcFaceEncoder:
    name = "arcface"
    input_size = (112, 112)
    def __init__(self, device: str, arc_model_name: str = "arcface_r100_v1"):
        self.model = get_insight_model(arc_model_name)
        ctx_id = 0 if device.type == "cuda" else -1
        self.model.prepare(ctx_id=ctx_id)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        v = self.model.get_embedding(rgb).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v


# -------- SphereFace minimal --------
import torch.nn as nn
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
            nn.Linear(512*14*14, 512, bias=False),
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
    return IResNet([3,4,14,3])

class SphereFaceEncoder:
    name = "sphereface"
    input_size = (112,112)
    def __init__(self, device: str, ckpt_path: str):
        self.device = device
        self.model = build_iresnet50().to(device).eval()
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k,v in sd["state_dict"].items()}
        self.model.load_state_dict(sd, strict=False)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2,0,1).float()/255.0
        t = (t - 0.5)/0.5
        with torch.inference_mode():
            feat = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        feat /= (np.linalg.norm(feat)+1e-12)
        return feat


# ------------------ Alignment ------------------
REF_5PTS_112 = np.float32([[38.2946,51.6963],
                           [73.5318,51.5014],
                           [56.0252,71.7366],
                           [41.5493,92.3655],
                           [70.7299,92.2041]])
scale_160 = 160.0/112.0
REF_5PTS_160 = REF_5PTS_112 * scale_160

def align_by_5pts(bgr: np.ndarray, kps: np.ndarray, out_size=(112,112)) -> np.ndarray:
    ref = REF_5PTS_112 if out_size==(112,112) else REF_5PTS_160
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)[0]
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


# ------------------ Main ------------------
def main():
    # pick defaults here
    MODEL_NAME = "arcface"   # change to "facenet" or "sphereface"
    DET_NAME = "retinaface_r50_v1"
    CAMERA_ID = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detector
    app = FaceAnalysis(name=DET_NAME)
    ctx_id = 0 if DEVICE.type == "cuda" else -1
    app.prepare(ctx_id=ctx_id, det_size=(640,640))

    # Encoder
    if MODEL_NAME == "arcface":
        encoder = ArcFaceEncoder(device=DEVICE, arc_model_name="arcface_r100_v1")
        out_size = (112,112)
    elif MODEL_NAME == "facenet":
        encoder = FaceNetEncoder(device=DEVICE)
        out_size = (160,160)
    else:  # sphereface
        ckpt_path = os.path.join(os.path.dirname(__file__), "sphereface.pth")
        encoder = SphereFaceEncoder(device=DEVICE, ckpt_path=ckpt_path)
        out_size = (112,112)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}")
        return

    print("Press ESC to exit.")
    times_ms = deque(maxlen=200)
    fps_hist = deque(maxlen=60)
    last = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        disp = frame.copy()

        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(int)
            kps = f.kps.astype(np.float32)

            crop = align_by_5pts(frame, kps, out_size=out_size)
            if crop is None:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, out_size)

            t1 = time.perf_counter()
            _ = encoder.embed(crop)
            t2 = time.perf_counter()
            times_ms.append((t2 - t1)*1000.0)

            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            for (px,py) in kps.astype(int):
                cv2.circle(disp, (px,py), 2, (0,255,255), -1)

        now = time.perf_counter()
        fps = 1.0 / (now - last)
        last = now
        fps_hist.append(fps)

        y = 30
        if len(times_ms) > 5:
            p50 = np.percentile(np.array(times_ms), 50)
            p95 = np.percentile(np.array(times_ms), 95)
            cv2.putText(disp, f"{encoder.name} enc: {p50:.1f}/{p95:.1f} ms", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y += 25
        cv2.putText(disp, f"FPS: {np.mean(fps_hist):.1f}    faces: {len(faces)}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Live FR Wrapper", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
