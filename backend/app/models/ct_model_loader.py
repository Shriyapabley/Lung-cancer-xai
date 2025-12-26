import os
import sys
from functools import lru_cache

# add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torchvision import transforms
from PIL import Image

from app.core.config import get_settings
from ml.models.ct_cnn import CTCNNModel
from ml.gradcam_utils import GradCAM, overlay_heatmap_on_image


settings = get_settings()


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache()
def load_ct_model():
    device = _get_device()
    model = CTCNNModel(num_classes=settings.NUM_CLASSES, pretrained=False)
    state_dict = torch.load(settings.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def preprocess_ct_image(file_path: str, img_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    img = Image.open(file_path).convert("L")
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def predict_ct(file_path: str):
    model, device = load_ct_model()
    _, tensor = preprocess_ct_image(file_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
        pred_class = int(outputs.argmax(dim=1).item())
    return pred_class, probs


def explain_ct(file_path: str, target_layer: str = "backbone.layer4"):
    model, device = load_ct_model()
    original_pil, tensor = preprocess_ct_image(file_path)
    tensor = tensor.to(device)

    grad_cam = GradCAM(model, target_layer_name=target_layer)
    heatmap = grad_cam.generate(tensor)
    overlay_bgr = overlay_heatmap_on_image(original_pil, heatmap, alpha=0.4)
    return overlay_bgr
