import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "soil_classifier.pth"

class SoilClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = ["Red", "Black", "Alluvial", "Clay"]

        # ✅ No deprecated pretrained flag
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features, 4
        )

        # ✅ Load trained weights
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image):
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        idx = probs.argmax().item()
        return self.classes[idx], float(probs[idx])