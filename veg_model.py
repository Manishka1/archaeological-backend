import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# -------------------------------------------------
# PATH
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "veg_unet_lite.pth"

# -------------------------------------------------
# UNET-LITE (MATCHES TRAINING NAMES)
# -------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.b = DoubleConv(128, 256)

        self.u3 = DoubleConv(256 + 128, 128)
        self.u2 = DoubleConv(128 + 64, 64)
        self.u1 = DoubleConv(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        b = self.b(self.pool(d3))

        x = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u3(torch.cat([x, d3], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u2(torch.cat([x, d2], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.u1(torch.cat([x, d1], dim=1))

        return torch.sigmoid(self.out(x))


# -------------------------------------------------
# VEGETATION MODEL WRAPPER
# -------------------------------------------------
class VegetationModel:
    def __init__(self):
        self.device = "cpu"  # ✅ Render-safe
        self.model = UNetLite().to(self.device)

        # ✅ NOW KEYS MATCH — THIS WILL LOAD
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def predict(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        x = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mask = (self.model(x) > 0.5).float().squeeze().numpy()

        veg_pixels = np.sum(mask == 1)
        total_pixels = mask.size

        veg_percent = (veg_pixels / total_pixels) * 100.0
        non_veg_percent = 100.0 - veg_percent

        return round(veg_percent, 2), round(non_veg_percent, 2)