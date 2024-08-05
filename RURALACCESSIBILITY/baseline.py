import time, os, glob, sys, pickle
import torch
from tqdm import tqdm

import utils, sam, clip

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ##### Argument Preparation #####
# multimask_output = False
# test_folder_path = "../../RollingsAccessibility/test_images"
# output_dir = "./outputs"
# sam_output_path = "./outputs/sam_output.pickle"
# dino_threshold = 0.3
# show = True
# save = False

# text_accessibility_features = ["townhome", "apartment", "first_story", "second_story", "third_story", "fourth_story", "entrance", "stair", "ramp", "step_free", "elevator", "air_conditioning",
#                                "sidewalk", "curb_cut", "parking_lot"]
# ################################

# model_sam = sam.SAM(device=device)
# model_sam.prepare_dino()

# sam_output = {}
# for file_name in tqdm(os.listdir(test_folder_path), desc="Processing images..."):
#     if file_name.endswith('.PNG') or file_name.endswith('.png') or file_name.endswith('.JPG') or file_name.endswith('.jpg'):
#         image_path = os.path.join(test_folder_path, file_name)
#         image = utils.read_image_path(image_path)
        
#         sam_output[file_name] = {}
#         if utils.check_word_in_text("aerial", file_name) or utils.check_word_in_text("siteplan", file_name):
#             accessibility_text = "parking_lot"
#             input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
#             if len(input_boxes) != 0:
#                 inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
#                 dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
#                 sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)
                
#         else:
#             for accessibility_text in text_accessibility_features:
#                 input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
#                 if len(input_boxes) != 0:
#                     inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
#                     dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
#                     sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)

# with open(sam_output_path, 'wb') as f:
#     pickle.dump(sam_output, f)

def eval_baseline(sam_output_path, test_folder_path):
    with open(sam_output_path, 'rb') as f:
        preds = pickle.load(f)
    
    iou_dict = {}
    dice_dict = {}
    for file_name in tqdm(os.listdir(test_folder_path), desc="Evaluating..."):
        if file_name.endswith('.JSON') or file_name.endswith('.json'):
            image_path, gt_mask_dict = utils.json_to_masks(os.path.join(test_folder_path, file_name))
            for label, gt_masks in gt_mask_dict.items():
                if label not in iou_dict:
                    iou_dict[label] = []
                    dice_dict[label] = []

                if label in preds[image_path]:
                    pred_masks, pred_scores = preds[image_path][label]
                    
                    combined_gt_mask = torch.max(torch.from_numpy(gt_masks), dim=0, keepdim=True)[0]
                    combined_pred_mask = torch.max(pred_masks, dim=0, keepdim=True)[0]
                    iou = utils.calculate_iou(combined_gt_mask, combined_pred_mask)
                    iou_dict[label].append(iou)
                    dice = utils.calculate_dice(combined_gt_mask, combined_pred_mask)
                    dice_dict[label].append(dice)

    avg_ious, count = 0, 0
    for label, ious in iou_dict.items():
        if len(ious) == 0:
            print(f"{label}: NO IOUs")
            continue
        print(f"{label}: {sum(ious)/len(ious)}")
        avg_ious += sum(ious)/len(ious)
        count += 1
    print(f"Average IOU: {avg_ious/count}")
    
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
    eval_baseline(sam_output_path, test_folder_path)