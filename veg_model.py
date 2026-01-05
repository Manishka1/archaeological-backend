import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "veg_unet_lite.pth"

class VegetationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ EXACTLY match training config
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        # ✅ Load trained weights
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def predict(self, image_bgr):
        # OpenCV BGR → RGB PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image
        img = Image.fromarray(image_rgb)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=x).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )
            mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # ✅ Class 1 = vegetation
        veg_pixels = np.sum(mask == 1)
        total_pixels = mask.size

        veg_percent = (veg_pixels / total_pixels) * 100
        non_veg_percent = 100 - veg_percent

        return round(veg_percent, 2), round(non_veg_percent, 2)