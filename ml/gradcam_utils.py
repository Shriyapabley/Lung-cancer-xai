# ml/gradcam_utils.py

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2


class GradCAM:
    """
    Simple Grad-CAM for ResNet-like models.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.model.eval()

        self.target_layer = dict([*self.model.named_modules()])[target_layer_name]

        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: shape (1, C, H, W)
        returns heatmap as numpy array (H, W) normalized to [0,1]
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        # gradients: (1, K, u, v); activations: (1, K, u, v)
        gradients = self.gradients  # dY/dA_k
        activations = self.activations

        # global-average-pool the gradients over spatial dimensions
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, K, 1, 1)

        # weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, u, v)
        cam = F.relu(cam)

        # normalize
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def load_image_as_tensor(image_path: str, img_size: int = 224):
    """
    Load a single grayscale CT image and convert to tensor of shape (1, 1, H, W).
    """
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    img = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def overlay_heatmap_on_image(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """
    Overlay CAM heatmap on original image and return BGR numpy image (for saving with cv2).
    """
    orig = np.array(original_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, orig, 1 - alpha, 0)
    return overlay
