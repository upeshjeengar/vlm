import torch
import torch.nn.functional as F
import os
import random

# Import project modules
from dataset.cc_dataloader import ROCOImageCaptionDataset, get_dataloaders
from networks.q_former import QFormer
from utils.calculate_recall import calculate_recall
from utils.utils import *
from transformers import AutoTokenizer

def main():
    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Configuration
    MODEL_PATH = "models/new1/best"
    OUTPUT_DIR = "inference_results"
    NUM_SAMPLES = 8  # Number of images/captions (NxN grid)
    RECALL_MAX_SAMPLES = 500 # Limit for speed
    
    # 1. Load Dataset (Handles ViT and Tokenizer)
    print("Loading dataset and ViT model...")
    dataset = ROCOImageCaptionDataset(
        image_dir="test",
        captions_csv="dataset/test_captions.csv",
        tokenizer="distilbert-base-uncased",   # ← IMPORTANT
    )
    
    # 2. Load QFormer Model
    # print(f"Loading QFormer from {MODEL_PATH}...")
    # if not os.path.exists(MODEL_PATH):
    #     print(f"Error: Model path {MODEL_PATH} does not exist.")
    #     return

    qformer = QFormer.from_pretrained("models/trained_qformer/best")
    qformer.to(device)
    qformer.eval()

    # --- Part A: 5x5 Grid Visualization ---
    
    # 3. Select Random Samples
    dataset_len = len(dataset)
    indices = random.sample(range(dataset_len), NUM_SAMPLES)
    print(f"Selected indices: {indices}")

    # Store samples
    samples = []
    for idx in indices:
        # image_tensor: [num_patches, hidden_dim]
        image_tensor, caption = dataset[idx]
        
        # Get original image for visualization
        image_path = dataset._examples[idx].image_path
        with Image.open(image_path) as im:
            orig_image = im.convert("RGB")
            
        samples.append({
            "image_tensor": image_tensor,
            "caption": caption,
            "orig_image": orig_image,
            "id": idx
        })

    # 4. Compute NxN Similarity Matrix
    print("Computing NxN similarity matrix...")
    scores_matrix = torch.zeros((NUM_SAMPLES, NUM_SAMPLES))
    
    for row_idx, text_sample in enumerate(samples):
        
        text_inputs = dataset.tokenizer(
            [text_sample["caption"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        
        for col_idx, img_sample in enumerate(samples):
            visual_feats = img_sample["image_tensor"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_out, t_out = qformer(
                    visual_feats=visual_feats,
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                    attention_mode="uni_modal" 
                )
            
            q_norm = F.normalize(q_out, dim=1)
            t_norm = F.normalize(t_out, dim=1)
            
            similarity = (q_norm @ t_norm.T).item()
            scores_matrix[row_idx, col_idx] = similarity

    # --- Part B: Recall Calculation ---
    print(f"Calculating Recall@K on {RECALL_MAX_SAMPLES} samples...")
    # Get test loader
    _, test_loader = get_dataloaders(batch_size=16)
    
    recall_metrics = calculate_recall(
        model=qformer,
        dataloader=test_loader,
        device=device,
        k_values=[1, 5, 10],
        max_samples=RECALL_MAX_SAMPLES
    )
    
    print("Recall Metrics:", recall_metrics)

    # 5. Create Grid Visualization with Metrics
    print("Creating grid visualization...")
    create_similarity_grid(samples, scores_matrix, recall_metrics, OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
