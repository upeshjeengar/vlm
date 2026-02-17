from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

image = Image.open("dataset/cc_images/00000/000001964.jpg")
print("Input image shape: ", image.size)

model = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model)
model = ViTModel.from_pretrained(model)

print(model)

inputs = feature_extractor(images=image, return_tensors="pt")
print("preprocessed_shape: ", inputs.pixel_values.shape)

with torch.no_grad():
    outputs = model(**inputs)
print("Embeddings shape: ", outputs.last_hidden_state.shape)
