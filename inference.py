import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    ViTModel,
    ViTImageProcessor,
)
from vlm_train.networks.lm_to_vlm import LM_2_VLM

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
IMAGE_PATH = "image.png"
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
).to(device)

# -------------------------
# Generate
# -------------------------
with torch.no_grad():
    output_ids = model.generate(
        img=image_embeddings,
        prefix_ids=prompt,
        max_new_tokens=100,
    )

# -------------------------
# Decode
# -------------------------
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

print("\n=== GENERATED OUTPUT ===\n")
print(output_text)
