import time, os, glob, sys, pickle
import torch
from tqdm import tqdm

import utils, sam, clip

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

##### Argument Preparation #####
num_input_points = 64
multimask_output = False
test_folder_path = "../../RollingsAccessibility/test_images"
output_dir = "./outputs"
sam_output_path = "./outputs/sam_output.pickle"
dino_threshold = 0.3
pred_threshold = 0.7
iou_threshold  = 0.95
show = True
save = False

text_accessibility_features = ["townhome", "apartment", "first_story", "second_story", "third_story", "fourth_story", "entrance", "stair", "ramp", "step_free", "elevator", "air_conditioning",
                               "sidewalk", "curb_cut", "parking_lot"]
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
                
        else:
            for accessibility_text in text_accessibility_features:
                input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
                if len(input_boxes) != 0:
                    inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
                    dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
                    sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)

with open(sam_output_path, 'wb') as f:
    pickle.dump(sam_output, f)

            # input_points = model_sam.prepare_input_points(image_path, num_input_points=num_input_points)
            # inputs, outputs = model_sam.segment(image, input_points=input_points, multimask_output=multimask_output)
            # unique_masks, unique_scores = model_sam.postprocess(inputs=inputs, outputs=outputs,
            #                                                     pred_threshold=pred_threshold, iou_threshold=iou_threshold,
            #                                                     show=show, save=save,
            #                                                     output_dir=output_dir, input_image_path=image_path)

##### Image Preparation #####
# image_path = "./data/113_SHELB_streetview_SE.png"
# image_path = "../../RollingsAccessibility/test_images/067_BERKE_streetview_SE.png"
# image = utils.read_image_path(image_path)
#############################

# ##### Model & Prompt Preparation #####
# sam_start_time = time.time()
# model_sam = sam.SAM(device=device)
# input_points = model_sam.prepare_input_points(image_path, num_input_points=num_input_points)
######################################

##### Segment #####
# inputs, outputs = model_sam.segment(image, input_points=input_points, multimask_output=multimask_output)
# unique_masks, unique_scores = model_sam.postprocess(inputs=inputs, outputs=outputs,
#                                                     pred_threshold=pred_threshold, iou_threshold=iou_threshold,
#                                                     show=show, save=save,
#                                                     output_dir=output_dir, input_image_path=image_path)
# if not show:
#     sam_finish_time = time.time()
#     print(f"Time for SAM Segmentation for 1 image with {num_input_points} input points: {sam_finish_time-sam_start_time:.4f}")
###################

##### CLIP Image-Text Similarity #####
# model_clip = clip.CLIP(device=device)
# mask_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0])
# mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
# mask_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# masks = [utils.read_image_path(mask) for mask in mask_paths]

# target_query = "parking_lot"
# query_paths = glob.glob(os.path.join("./queries", target_query, "*.png"))
# queries = [utils.read_image_path(query_mask) for query_mask in query_paths]

# # text_embeddings = model_clip.get_text_embeddings(text_accessibility_features)
# text_embeddings = model_clip.get_text_embeddings([target_query.replace("_", " ")])
# query_embeddings = model_clip.get_image_embeddings(queries).mean(dim=0, keepdim=True)
# image_embeddings = model_clip.get_image_embeddings(masks)
# vl_embeddings = torch.stack((text_embeddings, query_embeddings), dim=0).mean(dim=0, keepdim=True)

# model_clip.compute_cosine_sim(text_accessibility_features, mask_paths, text_embeddings, image_embeddings, query_type="text")
# model_clip.compute_cosine_sim([target_query.replace("_", " ")], mask_paths, query_embeddings, image_embeddings, query_type="vision")
# model_clip.compute_cosine_sim([target_query.replace("_", " ")], mask_paths, vl_embeddings, image_embeddings, query_type="vision")
######################################

"""
./outputs\017_MAPLE_aerial\132.png
"""