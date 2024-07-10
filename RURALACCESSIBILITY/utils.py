import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import ListedColormap

def read_image_path(path):
    return Image.open(path).convert("RGB")

def extract_image_grid_points(W, H, num_points=512):
    grid_size = math.sqrt(num_points)
    x_interval, y_interval = W/grid_size, H/grid_size
    coordinates = [(int(i*x_interval), int(j*y_interval)) for j in range(int(grid_size)) for i in range(int(grid_size))]
    
    # Adjust if the number of points is less than requested
    if len(coordinates) < num_points:
        remaining_points = num_points - len(coordinates)
        for i in range(remaining_points):
            x = int((i % int(grid_size)) * x_interval + x_interval / 2)
            y = int((i // int(grid_size)) * y_interval + y_interval / 2)
            coordinates.append((x, y))

    return coordinates[:num_points]

def wrap_points(input_points):
    return [[point] for point in input_points]

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def show_mask(mask, ax, score, color):
    cmap = ListedColormap(['none', color])
    ax.imshow(mask, cmap=cmap, alpha=0.5)
    y, x = np.where(mask > 0)
    # if len(y) > 0:
    #     ax.text(x.min(), y.min(), f'Score: {score:.3f}', color='white', fontsize=12, backgroundcolor='black')

def show_masks_on_image(raw_image, masks, scores, input_points):
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)

    nb_predictions = scores.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))

    cmap = plt.get_cmap('hsv', nb_predictions)
    colors = [cmap(i) for i in range(nb_predictions)]

    for i in range(nb_predictions):
        mask = masks[i].cpu().detach().numpy()
        color = colors[i]
        show_mask(mask, ax, scores[i].item(), color)

    for point in input_points:
        x, y = point
        ax.plot(x, y, 'o', color='white', markersize=8)
        # ax.text(x + 5, y, f'({x}, {y})', color='white', fontsize=10, backgroundcolor='black')

    ax.axis('off')
    plt.show()

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou_score = torch.sum(intersection).float() / torch.sum(union).float()
    return iou_score.item()

def remove_duplicate_masks(masks, iou_scores, threshold=0.95):
    unique_masks = []
    unique_scores = []
    for i, (mask, score) in enumerate(zip(masks, iou_scores)):
        is_duplicate = False
        for unique_mask in unique_masks:
            if calculate_iou(mask, unique_mask) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_masks.append(mask)
            unique_scores.append(score)
    return unique_masks, unique_scores

def save_masked_image(image, mask, output_directory, image_name, mask_number):
    image_np = np.array(image)
    white_background = np.ones_like(image_np) * 255
    
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    masked_image = np.where(mask[..., None] == 1, image_np, white_background)

    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    image_basename = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_directory, image_basename)
    os.makedirs(output_path, exist_ok=True)

    output_file_path = os.path.join(output_path, f"{str(mask_number)}.png")
    masked_image_pil.save(output_file_path)
    
# extract_image_grid_points(1920, 899)