import time, os, glob, sys
import torch

import utils, sam, clip

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

##### Argument Preparation #####
num_input_points = 512
multimask_output = False
output_dir = "./outputs"
pred_threshold = 0.7
iou_threshold  = 0.95
show = True
save = False

text_accessibility_features = ["single or multiple building", "sidewalk", "curb cut", "parking lot"]
# image_paths = glob.glob(os.path.join("./data", "*.png"))
################################

##### Image Preparation #####
# image_path = "./data/117_TEALR_streetview_NE.png"
# image_path = "./data/113_SHELB_streetview_SE.png"
image_path = r"C:\Users\17346\OneDrive - Umich\바탕 화면\DPM\RuralAccessibility\Report_figure.png"
image = utils.read_image_path(image_path)
#############################

# ##### Model & Prompt Preparation #####
sam_start_time = time.time()
model_sam = sam.SAM(device=device)
input_points = model_sam.prepare_input_points(image_path, num_input_points=num_input_points)
######################################

##### Segment #####
inputs, outputs = model_sam.segment(image, input_points=input_points, multimask_output=multimask_output)
unique_masks, unique_scores = model_sam.postprocess(inputs=inputs, outputs=outputs,
                                                    pred_threshold=pred_threshold, iou_threshold=iou_threshold,
                                                    show=show, save=save,
                                                    output_dir=output_dir, input_image_path=image_path)
if not show:
    sam_finish_time = time.time()
    print(f"Time for SAM Segmentation for 1 image with {num_input_points} input points: {sam_finish_time-sam_start_time:.4f}")
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