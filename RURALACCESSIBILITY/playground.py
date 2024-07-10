import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
import requests
from transformers import SamModel, SamProcessor, GroundingDinoForObjectDetection, GroundingDinoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

model_id = "IDEA-Research/grounding-dino-base"
dino_processor = GroundingDinoProcessor.from_pretrained(model_id)
dino_model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)

sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# img_path = "./data/008_BRIDGE_perspective.png"
img_path = "./data/388_HUNTC_perspective1.jpg"
image = Image.open(img_path).convert("RGB")

text = "single building."
# input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = dino_processor(images=image, text=preprocess_caption(text), return_tensors="pt").to(device)

with torch.no_grad():
    outputs = dino_model(**inputs)

width, height = image.size
postprocessed_outputs = dino_processor.image_processor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.3)
results = postprocessed_outputs[0]
bboxes = results['boxes'].tolist()
# center_bboxes = [(((xmin+xmax)/2), ((ymin+ymax)/2)) for (xmin, ymin, xmax, ymax) in bboxes]
print("DINO Results:", bboxes)
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{text}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# plot_results(image, results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist())


# inputs = sam_processor(image, input_points=None, input_labels=None, input_boxes=[bboxes], return_tensors="pt").to(device)
inputs = sam_processor(image, input_points=None, input_labels=None, input_boxes=[bboxes], return_tensors="pt").to(device)
with torch.no_grad():
    # outputs = sam_model(**inputs)
    outputs = sam_model(**inputs, multimask_output=False)

masks = sam_processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
# print(masks)
scores = outputs.iou_scores

def show_mask(mask, ax, score):
    
    cmap = ListedColormap(['none', 'red'])
    ax.imshow(mask, cmap=cmap, alpha=0.5)
    # 점수 정보도 같이 표시
    y, x = np.where(mask > 0)
    if len(y) > 0:
        ax.text(x.min(), y.min(), f'Score: {score:.3f}', color='white', fontsize=12, backgroundcolor='black')

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)  # 마스크의 불필요한 차원을 제거
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)  # 점수의 불필요한 차원을 제거

    nb_predictions = scores.shape[0]  # 적절한 차원으로 점수의 수를 얻음
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))  # 원본 이미지를 먼저 표시

    # 모든 마스크를 같은 이미지에 그림
    for i in range(nb_predictions):
        mask = masks[i].cpu().detach()  # CPU로 이동 및 detach
        show_mask(mask, ax, scores[i].item())  # 각 마스크와 점수를 보여줌
        
    ax.axis('off')  # 축 숨김
    plt.show()

print("MASK SHAPE", masks[0].shape, scores.shape)
show_masks_on_image(image, masks[0], scores)