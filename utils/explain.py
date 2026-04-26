# utils/explain.py
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage import exposure


def _to_tensor_for_cam(pil_img, device):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    x = tform(pil_img).unsqueeze(0).to(device)
    return x


def generate_vit_gradcam(model, device, pil_img, target_class_idx=None):
    model.eval()
    img_size = pil_img.size

    x = _to_tensor_for_cam(pil_img, device)
    last_block = model.backbone.blocks[-1]

    feats = None
    grads = None

    def fwd_hook(module, inp, out):
        nonlocal feats
        feats = out

    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    h1 = last_block.register_forward_hook(fwd_hook)
    h2 = last_block.register_full_backward_hook(bwd_hook)

    logits = model(x)[1]
    probs = F.softmax(logits, dim=1)
    if target_class_idx is None:
        target_class_idx = int(torch.argmax(probs, dim=1).item())

    score = logits[0, target_class_idx]
    model.zero_grad()
    score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    if feats is None or grads is None:
        return pil_img

    weights = grads.mean(dim=2, keepdim=True)
    cam = (weights * feats).sum(dim=2)
    cam = cam[:, 1:]
    tokens = cam.size(1)
    side = int(tokens ** 0.5)
    cam = cam.reshape(1, side, side)

    cam = F.relu(cam)
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam = F.interpolate(
        cam.unsqueeze(0),
        size=(img_size[1], img_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze().detach().cpu().numpy()

    heatmap = np.uint8(255 * cam)
    heatmap_color = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    heatmap_color[..., 0] = heatmap

    base = pil_img.convert("RGB").resize(img_size)
    base_np = np.array(base).astype(np.float32)

    overlay = 0.5 * base_np + 0.5 * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def compute_quality_metrics(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray).astype(np.float32) / 255.0

    brightness = float(arr.mean())
    contrast = float(arr.std())

    pad = np.pad(arr, 1, mode="edge")
    lap = (
        pad[1:-1, 2:] + pad[1:-1, :-2] +
        pad[2:, 1:-1] + pad[:-2, 1:-1] -
        4 * pad[1:-1, 1:-1]
    )
    sharpness = float(lap.var())

    if brightness < 0.25:
        b_label = "Too dark"
    elif brightness > 0.75:
        b_label = "Too bright"
    else:
        b_label = "OK"

    if contrast < 0.08:
        c_label = "Low contrast"
    elif contrast > 0.25:
        c_label = "High contrast"
    else:
        c_label = "OK"

    if sharpness < 0.001:
        s_label = "Very soft / blurred"
    elif sharpness < 0.005:
        s_label = "Slight blur"
    else:
        s_label = "Sharp enough"

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "brightness_label": b_label,
        "contrast_label": c_label,
        "sharpness_label": s_label,
    }


def enhance_contrast_for_display(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray).astype(np.float32) / 255.0
    arr_eq = exposure.equalize_adapthist(arr, clip_limit=0.03)
    arr_uint8 = np.uint8(np.clip(arr_eq * 255, 0, 255))
    return Image.fromarray(arr_uint8)
