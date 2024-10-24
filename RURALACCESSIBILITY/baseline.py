import time, os, glob, sys, pickle
import torch
from tqdm import tqdm

import utils, sam, clip, eval

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ##### Argument Preparation #####
multimask_output = False
test_folder_path = "../../RollingsAccessibility/test_images"
# test_folder_path = "./data/total"
output_dir = "./outputs"
sam_output_path = "./outputs/sam_output.pickle"
dino_threshold = 0.3
show = True
save = False

# text_accessibility_features = ["townhome", "apartment", "first_story", "second_story", "third_story", "fourth_story", "entrance", "stair", "ramp", "step_free", "elevator", "air_conditioning",
#                                "sidewalk", "curb_cut", "parking_lot"]
text_accessibility_features = ["crosswalk", "curb_cut", "sidewalk"]
################################

model_sam = sam.SAM(device=device)
model_sam.prepare_dino()

sam_output = {}
for file_name in tqdm(os.listdir(test_folder_path), desc="Processing images..."):
    if file_name.endswith('.PNG') or file_name.endswith('.png') or file_name.endswith('.JPG') or file_name.endswith('.jpg'):
        image_path = os.path.join(test_folder_path, file_name)
        image = utils.read_image_path(image_path)
        
        sam_output[file_name] = {}
        if utils.check_word_in_text("aerial", file_name) or utils.check_word_in_text("siteplan", file_name):
            accessibility_text = "parking_lot"
            input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
            if len(input_boxes) != 0:
                inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
                dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
                sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)
            # for accessibility_text in ["crosswalk", "sidewalk"]:
            #     input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
            #     if len(input_boxes) != 0:
            #         inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
            #         dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
            #         sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)
                
        else:
            for accessibility_text in text_accessibility_features:
                input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
                if len(input_boxes) != 0:
                    inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
                    dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
                    sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)

with open(sam_output_path, 'wb') as f:
    pickle.dump(sam_output, f)

if __name__ == "__main__":
    sam_output_path = "./outputs/sam_output.pickle"
    test_folder_path = "../../RollingsAccessibility/test_images"
    eval.eval_baseline(sam_output_path, test_folder_path)