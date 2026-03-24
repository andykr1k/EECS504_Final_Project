try:
    import torch
    import torchvision  # We don't use this, but we need to check it because SMA3 import fails with a weird error if it is not installed
    import accelerate
except ImportError:
    print("Please install torch and torchvision")
    exit(1)

from transformers import Sam3Processor, Sam3Model
from PIL import Image
import requests

from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch

# Load model and processor
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# model = Sam3Model.from_pretrained("facebook/sam3").to(device)
# # model = from_pretrained("facebook/sam3").to(device)
# processor = Sam3Processor.from_pretrained("facebook/sam3")

# cat_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
# kitchen_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
# images = [
#     Image.open(requests.get(cat_url, stream=True).raw).convert("RGB"),
#     Image.open(requests.get(kitchen_url, stream=True).raw).convert("RGB")
# ]

# text_prompts = ["ear", "dial"]

# inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model(**inputs)

# # Post-process results for both images
# results = processor.post_process_instance_segmentation(
#     outputs,
#     threshold=0.5,
#     mask_threshold=0.5,
#     target_sizes=inputs.get("original_sizes").tolist()
# )

# print(f"Image 1: {len(results[0]['masks'])} objects found")
# print(f"Image 2: {len(results[1]['masks'])} objects found")
