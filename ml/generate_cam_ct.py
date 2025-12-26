# ml/generate_cam_ct.py

import os
import argparse

import torch
import cv2

from models.ct_cnn import CTCNNModel
from gradcam_utils import GradCAM, load_image_as_tensor, overlay_heatmap_on_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to CT image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ct_resnet18_best.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cam_output.png",
        help="Where to save overlay image",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument(
        "--target_layer",
        type=str,
        default="backbone.layer4",  # last conv block in ResNet18
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    model = CTCNNModel(num_classes=args.num_classes, pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load image
    original_pil, input_tensor = load_image_as_tensor(
        args.image_path, img_size=args.img_size
    )
    input_tensor = input_tensor.to(device)

    # 3. Grad-CAM
    grad_cam = GradCAM(model, target_layer_name=args.target_layer)
    heatmap = grad_cam.generate(input_tensor)

    # 4. Overlay and save
    overlay_bgr = overlay_heatmap_on_image(original_pil, heatmap, alpha=0.4)
    cv2.imwrite(args.output_path, overlay_bgr)

    # Also print predicted class
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = probs.argmax()
        print(f"Predicted class: {pred_class}, probs: {probs}")


if __name__ == "__main__":
    main()
