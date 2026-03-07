'''
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    ViTModel,
    ViTImageProcessor,
)
from networks.lm_to_vlm import LM_2_VLM

# -------------------------
# Device
# -------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# -------------------------
# Config
# -------------------------
IMAGE_PATH = "image.jpg"
BASE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
CHECKPOINT_PATH = "models/vlm_peft/best"
QFORMER_PATH = "models/trained_qformer/best"

# -------------------------
# Load Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load VLM
# -------------------------
model = LM_2_VLM(
    model_name=BASE_MODEL,
    qformer_model_path=QFORMER_PATH,
    pad_token_id=tokenizer.pad_token_id,
)

model.load_checkpoint(CHECKPOINT_PATH)
model.to(device)
model.eval()

# -------------------------
# Load ViT (same as training)
# -------------------------
vit_name = "google/vit-base-patch16-224"
vit_processor = ViTImageProcessor.from_pretrained(vit_name)
vit_model = ViTModel.from_pretrained(vit_name).to(device)
vit_model.eval()

# -------------------------
# Load Image
# -------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

inputs = vit_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    vit_outputs = vit_model(**inputs)
    image_embeddings = vit_outputs.last_hidden_state  # (1, 197, 768)

# -------------------------
# Create Prompt
# -------------------------
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Answer the user's question truthfully"},
        {"role": "user", "content": "what is the color of rose?"},
    ],
    return_tensors="pt",
    add_generation_prompt=False,  # Don't add generation prompt here
).to(device)

# -------------------------
# Generate
# -------------------------
with torch.no_grad():
    output_ids = model.generate(
        img=image_embeddings,
        prefix_ids=prompt,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
    )

# -------------------------
# Decode
# -------------------------
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

print("\n=== GENERATED OUTPUT ===\n")
print(output_text)

'''
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, ViTModel, ViTImageProcessor
from networks.lm_to_vlm import LM_2_VLM
from tqdm import tqdm

# -------------------------
# Device
# -------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# -------------------------
# Config
# -------------------------
IMAGE_DIR = "train"
BASE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
CHECKPOINT_PATH = "/Users/upeshjeengar/Downloads/models/vlm_peft/best"
QFORMER_PATH = "/Users/upeshjeengar/Downloads/models/trained_qformer/best"
CAPTIONS_CSV = "dataset/train_captions.csv"
OUTPUT_CSV = "results.csv"
BATCH_SIZE = 8  # adjust based on your GPU/MPS memory

# -------------------------
# Load Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load VLM
# -------------------------
model = LM_2_VLM(
    model_name=BASE_MODEL,
    qformer_model_path=QFORMER_PATH,
    pad_token_id=tokenizer.pad_token_id,
)
model.load_checkpoint(CHECKPOINT_PATH)
model.to(device)
model.eval()

# -------------------------
# Load ViT
# -------------------------
vit_name = "google/vit-base-patch16-224"
vit_processor = ViTImageProcessor.from_pretrained(vit_name)
vit_model = ViTModel.from_pretrained(vit_name).to(device)
vit_model.eval()

# -------------------------
# Load expected captions
# -------------------------
captions_df = pd.read_csv(CAPTIONS_CSV)
captions_df = captions_df.set_index("ID")

# -------------------------
# Build the prompt once (same for every image)
# -------------------------
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Answer the user's question truthfully"},
        {"role": "user", "content": "Diagonse the X-ray for disease"},
    ],
    return_tensors="pt",
    add_generation_prompt=False,
).to(device)

# -------------------------
# Build list of image paths and IDs
# -------------------------
image_ids = [f"ROCOv2_2023_train_{i:06d}" for i in range(1, 51)]
image_paths = [f"{IMAGE_DIR}/{img_id}.jpg" for img_id in image_ids]

# -------------------------
# Batch inference
# -------------------------
results = []

for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Batches"):
    batch_end = min(batch_start + BATCH_SIZE, len(image_paths))
    batch_ids = image_ids[batch_start:batch_end]
    batch_paths = image_paths[batch_start:batch_end]

    # Load and preprocess images as a batch
    images = [Image.open(p).convert("RGB") for p in batch_paths]
    pixel_values = vit_processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        # Extract ViT embeddings for the whole batch at once
        vit_outputs = vit_model(pixel_values=pixel_values)
        batch_embeddings = vit_outputs.last_hidden_state  # (B, 197, 768)

    # Generate per image (generation is typically sequential unless your
    # model.generate supports batched image embeddings)
    for i, img_id in enumerate(batch_ids):
        img_emb = batch_embeddings[i : i + 1]  # (1, 197, 768)

        with torch.no_grad():
            output_ids = model.generate(
                img=img_emb,
                prefix_ids=prompt,
                max_new_tokens=60,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Look up expected caption
        expected = (
            captions_df.loc[img_id, "Caption"]
            if img_id in captions_df.index
            else ""
        )

        results.append(
            {
                "image_name": f"{img_id}.jpg",
                "generated_response": generated_text,
                "expected_caption": expected,
            }
        )

# -------------------------
# Save to CSV
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(results_df)} results to {OUTPUT_CSV}")
print(results_df.head())
