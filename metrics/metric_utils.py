import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'}


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def score_to_float(score):
    if isinstance(score, (list, tuple)):
        score = score[0]
    if torch.is_tensor(score):
        score = score.detach().mean().cpu().item()
    return float(score)


def select_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU for metric computation.')
        return torch.device('cpu')
    return torch.device(device)


def crop_border(img, border):
    if border <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError('Crop border {} is too large for image shape {}.'.format(border, img.shape))
    return img[border:h - border, border:w - border, ...]


def read_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError('Cannot read image: {}'.format(path))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _uint_to_lpips_tensor(img, device, crop=0, color_order='rgb'):
    img = crop_border(img, crop)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if color_order.lower() == 'bgr':
        img = img[:, :, ::-1]
    img = img.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return tensor.unsqueeze(0).to(device)


def build_lpips_model(net='alex', device='cuda'):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS requires the 'lpips' package. Install it with `pip install lpips`.") from exc

    device = select_device(device)
    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    return model, device


def calculate_lpips_from_arrays(output_img, gt_img, model, device, crop=0, color_order='rgb'):
    output_tensor = _uint_to_lpips_tensor(output_img, device, crop=crop, color_order=color_order)
    gt_tensor = _uint_to_lpips_tensor(gt_img, device, crop=crop, color_order=color_order)
    with torch.no_grad():
        return score_to_float(model(output_tensor, gt_tensor))


def calculate_lpips_from_paths(output_path, gt_path, model, device, crop=0):
    output_img = read_image_rgb(output_path)
    gt_img = read_image_rgb(gt_path)
    return calculate_lpips_from_arrays(output_img, gt_img, model, device, crop=crop, color_order='rgb')


def build_no_reference_models(metric_names, device='cuda', crop_border_value=20):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError("No-reference metrics require the 'pyiqa' package. Install it with `pip install pyiqa`.") from exc

    device = select_device(device)
    models = {}
    for name in metric_names:
        models[name] = pyiqa.create_metric(name, device=device, crop_border=crop_border_value)
    return models


def calculate_no_reference_from_path(image_path, models):
    scores = {}
    for name, model in models.items():
        with torch.no_grad():
            scores[name] = score_to_float(model(str(image_path)))
    return scores


def iter_output_images(output_root):
    output_root = Path(output_root)
    for image_path in sorted(output_root.rglob('*')):
        if image_path.is_file() and is_image_file(image_path):
            rel_path = image_path.relative_to(output_root)
            video_name = rel_path.parts[0] if len(rel_path.parts) > 1 else output_root.name
            yield video_name, image_path


def build_unique_name_index(root):
    index = {}
    duplicate_names = set()
    root = Path(root)
    for image_path in sorted(root.rglob('*')):
        if not image_path.is_file() or not is_image_file(image_path):
            continue
        name = image_path.name
        if name in index:
            duplicate_names.add(name)
        else:
            index[name] = image_path
    for name in duplicate_names:
        index.pop(name, None)
    return index


def resolve_gt_path(output_root, output_path, gt_root, gt_subdir='GS', gt_index=None):
    output_root = Path(output_root)
    output_path = Path(output_path)
    gt_root = Path(gt_root)
    rel_path = output_path.relative_to(output_root)
    rel_parent = rel_path.parent
    candidates = []
    if str(rel_parent) != '.':
        candidates.append(gt_root / rel_parent / gt_subdir / output_path.name)
        candidates.append(gt_root / rel_parent / output_path.name)
    candidates.append(gt_root / gt_subdir / output_path.name)
    candidates.append(gt_root / output_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if gt_index is not None and output_path.name in gt_index:
        return gt_index[output_path.name]
    raise FileNotFoundError('Cannot find GT for {} under {}.'.format(output_path, gt_root))


def mean_metric(rows, metric):
    values = [row[metric] for row in rows if metric in row and row[metric] is not None]
    if not values:
        return None
    return float(np.mean(values))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    if path.parent:
        os.makedirs(str(path.parent), exist_ok=True)
    with open(str(path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
