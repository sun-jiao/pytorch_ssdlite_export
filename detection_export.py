import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=0.5)

optimized_traced_model = torch.jit.script(model, optimize=optimize_for_mobile)
optimized_traced_model._save_for_lite_interpreter("ssdlite.pt")
