import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
        # ax.plot(x, y, 'o', color='white', markersize=8)
        # ax.text(x + 5, y, f'({x}, {y})', color='white', fontsize=10, backgroundcolor='black')

    ax.axis('off')
    plt.show()

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    if torch.sum(union).float() == 0:
        return 0.0
    iou_score = torch.sum(intersection).float() / torch.sum(union).float()
    return iou_score.item()

def calculate_dice(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().item()
    sum_masks = mask1.sum().item() + mask2.sum().item()
    if sum_masks == 0:
        return 0.0
    return 2 * intersection / sum_masks

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
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask.squeeze(0).squeeze(0)

    masked_image = np.where(mask[..., None] == 1, image_np, white_background)

    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    image_basename = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_directory, image_basename)
    os.makedirs(output_path, exist_ok=True)

    output_file_path = os.path.join(output_path, f"{str(mask_number)}.png")
    masked_image_pil.save(output_file_path)

def save_bbox_masked_image(image, mask, output_directory, image_name, mask_number):
    coords = np.argwhere(mask)
    y_coords, x_coords = coords[:, 1], coords[:, 2]
    
    y_min, x_min = y_coords.min(axis=0), x_coords.min(axis=0)
    y_max, x_max = y_coords.max(axis=0), x_coords.max(axis=0)
    
    image_np = np.array(image)
    cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[0, y_min:y_max+1, x_min:x_max+1]
    canvas = np.ones_like(image_np) * 255
    
    canvas_center_y, canvas_center_x = canvas.shape[0] // 2, canvas.shape[1] // 2
    cropped_center_y, cropped_center_x = cropped_image.shape[0] // 2, cropped_image.shape[1] // 2

    start_y = canvas_center_y - cropped_center_y
    start_x = canvas_center_x - cropped_center_x

    end_y = start_y + cropped_image.shape[0]
    end_x = start_x + cropped_image.shape[1]

    cropped_mask_expanded = np.repeat(cropped_mask[:, :, np.newaxis], 3, axis=2)
    canvas[start_y:end_y, start_x:end_x][cropped_mask_expanded] = cropped_image[cropped_mask_expanded]

    masked_image_pil = Image.fromarray(canvas.astype(np.uint8))
    image_basename = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_directory, image_basename)
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, f"{str(mask_number)}.png")
    masked_image_pil.save(output_file_path)

def check_word_in_text(word, text):
    if word.lower() in text.lower():
        return True
    return False

def json_to_masks(json_path):
    with open(json_path, 'r') as file:
        annotation = json.load(file)
    
    mask_dict = {}
    for shape in annotation['shapes']:
        if shape["shape_type"] != "polygon":
            continue
        label = shape['label']
        points = [(int(point[1]), int(point[0])) for point in shape['points']]
        
        mask = Image.new('L', (annotation['imageHeight'], annotation['imageWidth']), 0)
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        mask = np.array(mask).transpose(1, 0)

        if label not in mask_dict:
            mask_dict[label] = []
        mask_dict[label].append(mask)

    for label, masks in mask_dict.items():
        masks_array = np.stack(masks).reshape(len(masks), 1, annotation['imageHeight'], annotation['imageWidth'])
        mask_dict[label] = masks_array
        
    return annotation['imagePath'], mask_dict

def apply_mask(image, mask, color, alpha=0.5):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    assert image.size == mask.size

    mask_data = np.array(mask)*255
    rgba_mask = np.zeros((*mask_data.shape, 4), dtype=np.uint8)
    
    rgba_mask[..., :3] = color
    rgba_mask[..., 3] = mask_data * alpha
    
    mask_image = Image.fromarray(rgba_mask)
    image = Image.alpha_composite(image, mask_image)
    return image

def add_label(image, label, position, color=(255, 255, 255), font_size=20):
    draw = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, label, fill=color, font=font)
    return image

def apply_mask_to_image(image, gt_mask, pred_mask, gt_color=(0, 255, 0), pred_color=(255, 255, 0), alpha=0.5):
    pass

# extract_image_grid_points(1920, 899)