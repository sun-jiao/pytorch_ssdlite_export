import torch
from torch import Tensor
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
# from torchvision.transforms._presets import ObjectDetection
from torchvision.transforms.functional import to_pil_image, convert_image_dtype
from torchvision.utils import draw_bounding_boxes

model_type = False # True = default model, False = jit model
score_thresh = 0.5

# Step 1: Initialize model with the best available weights
weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=score_thresh) if model_type else torch.jit.load('ssdlite.pt')

model.eval()

# Step 2: Initialize the inference transforms
# preprocess = ObjectDetection()  # weights.transforms()

image_name = "86810585ED8E85C2CE8525BB8E17CF07.jpg"
image_path = f"test_images/{image_name}"
output_path = f"test_outputs/{image_name}"
img = read_image(image_path)

# Step 3: Apply inference preprocessing transforms
# batch = [preprocess(img)]
img = convert_image_dtype(img)
batch = [img, img]

# Step 4: Use the model and visualize the prediction
predictions = model(batch)
prediction = predictions[0] if model_type else predictions[1][0]
result_labels = prediction["labels"]
categories = weights.meta["categories"]
labels = [categories[i] for i in result_labels]

if model_type:
    top_labels = labels
    top_boxes = prediction["boxes"]
else:
    scores = prediction["scores"]
    top_results_index = [k for (k, v) in enumerate(scores) if v > score_thresh]
    top_labels = [v for (k, v) in enumerate(labels) if k in top_results_index]
    top_boxes = Tensor([v.detach().numpy() for (k, v) in enumerate(prediction["boxes"]) if k in top_results_index])

box = draw_bounding_boxes(img, boxes=top_boxes,
                          labels=top_labels,
                          colors="red",
                          width=1, font_size=30)
im = to_pil_image(box.detach())
im.save(output_path)
