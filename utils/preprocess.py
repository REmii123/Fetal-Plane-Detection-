# utils/preprocess.py
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def remove_overlay_text(pil_img: Image.Image) -> Image.Image:
    img_gray = pil_img.convert("L")
    arr = np.array(img_gray)
    h, w = arr.shape

    _, mask = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)
    x1, x2 = int(0.2 * w), int(0.8 * w)
    y1, y2 = int(0.2 * h), int(0.8 * h)
    central_mask = np.zeros_like(mask)
    central_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    kernel = np.ones((3, 3), np.uint8)
    central_mask = cv2.dilate(central_mask, kernel, iterations=1)
    inpainted = cv2.inpaint(arr, central_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return Image.fromarray(inpainted)


def preprocess_pil(pil_img: Image.Image, device: torch.device, clean_overlay: bool = True) -> torch.Tensor:
    if clean_overlay:
        pil_img = remove_overlay_text(pil_img)

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    x = VAL_TRANSFORM(pil_img)
    return x.unsqueeze(0).to(device)
