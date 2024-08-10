import os, pickle
import numpy as np 
import torch
import torchvision
import torchvision.transforms.functional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import utils

def eval_baseline(sam_output_path, test_folder_path, visualize=False, visualize_output_path="./outputs/visualized"):
    with open(sam_output_path, 'rb') as f:
        preds = pickle.load(f)
    
    iou_dict = {}
    dice_dict = {}
    for file_name in tqdm(os.listdir(test_folder_path), desc="Evaluating..."):
        if file_name.endswith('.JSON') or file_name.endswith('.json'):
            image_path, gt_mask_dict = utils.json_to_masks(os.path.join(test_folder_path, file_name)) # image.shape (np.array): (H, W, C)
            
            # To check only parking lot accessibility for aerial images
            is_aerial = (utils.check_word_in_text("aerial", file_name) or utils.check_word_in_text("siteplan", file_name))
            
            if visualize:
                image = utils.read_image_path(os.path.join(test_folder_path,image_path))
                gt_all_masks, pred_all_masks = [], []
                gt_label_positions, pred_label_positions = [], []
                pred_flag = 0
            if is_aerial:
                label = "parking_lot"
                if label in gt_mask_dict.keys():
                    if label not in iou_dict:
                        iou_dict[label], dice_dict[label] = [], []
                    if label in preds[image_path]:
                        pred_masks, pred_scores = preds[image_path][label]
                        combined_gt_mask = torch.any(torch.from_numpy(gt_mask_dict[label]), dim=0, keepdim=True)[0]
                        combined_pred_mask = torch.any(pred_masks, dim=0, keepdim=True)[0]
                        assert combined_gt_mask.shape == combined_pred_mask.shape # output shape example: torch.Size([1, 1, 825, 1917])
                        iou = utils.calculate_iou(combined_gt_mask, combined_pred_mask)
                        iou_dict[label].append(iou)
                        dice = utils.calculate_dice(combined_gt_mask, combined_pred_mask)
                        dice_dict[label].append(dice)

                        if visualize:
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_gt_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("gt", label)))
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_pred_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("pred", label)))
                            gt_indices = torch.where(combined_gt_mask.squeeze() == 1)
                            pred_indices = torch.where(combined_pred_mask.squeeze() == 1)
                            gt_left_top = (gt_indices[1][0].item(), gt_indices[0][0].item())
                            pred_left_top = (pred_indices[1][0].item(), pred_indices[0][0].item())
                            gt_label_positions.append((label, gt_left_top))
                            pred_label_positions.append((label, pred_left_top))
                            gt_all_masks.append(torch.from_numpy(gt_mask_dict[label]))
                            pred_all_masks.append(pred_masks)
                            pred_flag += 1

                    else:
                        iou_dict[label].append(0)
                        dice_dict[label].append(0)
                        if visualize:
                            combined_gt_mask = torch.any(torch.from_numpy(gt_mask_dict[label]), dim=0, keepdim=True)[0]
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_gt_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("gt", label)))

                            gt_indices = torch.where(combined_gt_mask.squeeze() == 1)
                            gt_left_top = (gt_indices[1][0].item(), gt_indices[0][0].item())
                            gt_label_positions.append((label, gt_left_top))
                            gt_all_masks.append(torch.from_numpy(gt_mask_dict[label]))

            else: # To check all accessibility features for street-view images
                for label, gt_masks in gt_mask_dict.items():
                    if label not in iou_dict:
                        iou_dict[label], dice_dict[label] = [], []
                    if label in preds[image_path]:
                        pred_masks, pred_scores = preds[image_path][label]
                        combined_gt_mask = torch.any(torch.from_numpy(gt_masks), dim=0, keepdim=True)[0]
                        combined_pred_mask = torch.any(pred_masks, dim=0, keepdim=True)[0]
                        assert combined_gt_mask.shape == combined_pred_mask.shape # output shape example: torch.Size([1, 1, 825, 1917])

                        iou = utils.calculate_iou(combined_gt_mask, combined_pred_mask)
                        iou_dict[label].append(iou)
                        dice = utils.calculate_dice(combined_gt_mask, combined_pred_mask)
                        dice_dict[label].append(dice)
                        
                        if visualize:
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_gt_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("gt", label)))
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_pred_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("pred", label)))
                            gt_indices = torch.where(combined_gt_mask.squeeze() == 1)
                            pred_indices = torch.where(combined_pred_mask.squeeze() == 1)
                            gt_left_top = (gt_indices[1][0].item(), gt_indices[0][0].item())
                            pred_left_top = (pred_indices[1][0].item(), pred_indices[0][0].item())
                            gt_label_positions.append((label, gt_left_top))
                            pred_label_positions.append((label, pred_left_top))
                            gt_all_masks.append(torch.from_numpy(gt_masks))
                            pred_all_masks.append(pred_masks)
                            pred_flag += 1
                        
                    else:
                        iou_dict[label].append(0)
                        dice_dict[label].append(0)
                        if visualize:
                            combined_gt_mask = torch.any(torch.from_numpy(gt_mask_dict[label]), dim=0, keepdim=True)[0]
                            utils.save_masked_image(utils.read_image_path(os.path.join(test_folder_path,image_path)), combined_gt_mask, visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "_".join(("gt", label)))
                            gt_indices = torch.where(combined_gt_mask.squeeze() == 1)
                            gt_left_top = (gt_indices[1][0].item(), gt_indices[0][0].item())
                            gt_label_positions.append((label, gt_left_top))
                            gt_all_masks.append(torch.from_numpy(gt_mask_dict[label]))
                            
            
            if visualize and len(gt_all_masks) > 0:
                gt_all_masks = torch.cat(gt_all_masks, dim=0)
                combined_gt_all = torch.any(gt_all_masks, dim=0, keepdim=True)[0]
                pil_gt_mask = torchvision.transforms.functional.to_pil_image(combined_gt_all)
                image = utils.apply_mask(image, pil_gt_mask, color=(0, 255, 0))
                if pred_flag > 0:
                    pred_all_masks = torch.cat(pred_all_masks, dim=0)
                    combined_pred_all = torch.any(pred_all_masks, dim=0, keepdim=True)[0]
                    pil_pred_mask = torchvision.transforms.functional.to_pil_image(combined_pred_all.to(dtype=torch.uint8))
                    image = utils.apply_mask(image, pil_pred_mask, color=(0, 0, 255))
                    for gt_position, pred_position in zip(gt_label_positions, pred_label_positions):
                        label, gt_position = gt_position
                        label, pred_position = pred_position
                        image = utils.add_label(image, label, position=gt_position, color=(0, 255, 0), font_size=30)
                        image = utils.add_label(image, label, position=pred_position, color=(0, 0, 255), font_size=30)
                else:
                    for gt_position in gt_label_positions:
                        label, gt_position = gt_position
                        image = utils.add_label(image, label, position=gt_position, color=(0, 255, 0), font_size=30)

                total_mask_path = os.path.join(visualize_output_path, os.path.splitext(os.path.basename(image_path))[0], "total.png")
                image.save(total_mask_path)

    avg_ious, count = 0, 0
    for label, ious in iou_dict.items():
        if len(ious) == 0:
            print(f"{label}: NO IOUs")
            continue
        print(f"{label}: {sum(ious)/len(ious)}")
        avg_ious += sum(ious)/len(ious)
        count += 1
    print(f"Average IOU: {avg_ious/count}\n")
    
    avg_dices, count = 0, 0
    for label, dices in dice_dict.items():
        if len(dices) == 0:
            print(f"{label}: NO DICEs")
            continue
        print(f"{label}: {sum(dices)/len(dices)}")
        avg_dices += sum(dices)/len(dices)
        count += 1
    print(f"Average DICE: {avg_dices/count}")
    return

if __name__ == "__main__":
    sam_output_path = "./outputs/sam_output.pickle"
    test_folder_path = "../../RollingsAccessibility/test_images"
    visualize_output_path="./outputs/visualized/baseline/v1"
    visualize = True
    eval_baseline(sam_output_path, test_folder_path, visualize=visualize, visualize_output_path=visualize_output_path)
    