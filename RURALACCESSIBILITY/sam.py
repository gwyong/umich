import os, glob
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import SamModel, SamProcessor

import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

class SAM(nn.Module):
    def __init__(self, model_id="facebook/sam-vit-huge", device=device):
        super().__init__()
        self.device = device
        self.model = SamModel.from_pretrained(model_id).to(device)
        self.processor = SamProcessor.from_pretrained(model_id)
    
    def prepare_input_points(self, input_image_path, num_input_points=128):
        self.num_input_points = num_input_points
        image = utils.read_image_path(input_image_path)
        input_points = [utils.extract_image_grid_points(image.size[0], image.size[1], num_points=num_input_points)]
        # NOTE: Treat each point as an input point.
        input_points = [utils.wrap_points(point_group) for point_group in input_points]
        return input_points

    def segment(self, image, input_points=None, input_labels=None, input_boxes=None, multimask_output=False):
        """
        Segment a single input image with input_prompts
            
            multimask_output (`bool`, *optional*):
                In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
                bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
                "best" mask, by specifying `multimask_output=False`.
        
        Base class for Segment-Anything model's output
        Args:
            iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
                The iou scores of the predicted masks.
            pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
                The predicted low resolutions masks. Needs to be post-processed by the processor
            NOTE: There are mores (etc...).
        """
        inputs = self.processor([image], input_points=input_points, input_labels=input_labels, input_boxes=input_boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)
        return inputs, outputs
    
    def postprocess(self, inputs, outputs, pred_threshold=0.7, iou_threshold=0.95,
                    show=False, save=True, output_dir="./output", input_image_path=None):
        masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())
        
        filtered_masks, filtered_scores = [], []
        num_total_masks = masks[0].shape[0]

        for i in range(num_total_masks):
            if outputs.iou_scores[0][i] >= pred_threshold:
                filtered_masks.append(masks[0][i])
                filtered_scores.append(outputs.iou_scores[0][i])

        unique_masks, unique_scores = utils.remove_duplicate_masks(filtered_masks, filtered_scores, iou_threshold)

        image = utils.read_image_path(input_image_path)
        if save:
            image_basename = os.path.splitext(os.path.basename(input_image_path))[0]
            output_folder_path = os.path.join(output_dir, image_basename)
            os.makedirs(output_folder_path, exist_ok=True)
            for i, mask in enumerate(unique_masks):
                # utils.save_masked_image(image, mask.detach().numpy(), output_dir, image_basename, i)
                utils.save_bbox_masked_image(image, mask.detach().numpy(), output_dir, image_basename, i)

        unique_masks = torch.stack(unique_masks, dim=1).cpu().permute(1, 0, 2, 3)
        unique_scores = torch.stack(unique_scores, dim=1).cpu().unsqueeze(2)
        if show:
            input_points = [utils.extract_image_grid_points(image.size[0], image.size[1], num_points=self.num_input_points)]
            utils.show_masks_on_image(image, unique_masks, unique_scores, input_points[0])
        
        return unique_masks, unique_scores

def sam_save_masks_from_images(image_paths, num_input_points=9, multimask_output=False,
                               pred_threshold=0.7, iou_threshold=0.95, output_dir="./outputs"):
    model_sam = SAM(device=device)
    for image_path in tqdm(image_paths, desc="Saving masks..."):
        image = utils.read_image_path(image_path)
        input_points = model_sam.prepare_input_points(image_path, num_input_points=num_input_points)
        inputs, outputs = model_sam.segment(image, input_points=input_points, multimask_output=multimask_output)
        unique_masks, unique_scores = model_sam.postprocess(inputs=inputs, outputs=outputs,
                                                            pred_threshold=pred_threshold, iou_threshold=iou_threshold,
                                                            show=False, save=True,
                                                            output_dir=output_dir, input_image_path=image_path)
    return
    
    