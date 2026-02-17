import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def calculate_recall(model, dataloader, device, k_values=[1, 5, 10], max_samples=None):
    """
    Calculates Image-to-Text (I2T) and Text-to-Image (T2I) Recall@K.
    
    Args:
        model: The QFormer model (must be in eval mode).
        dataloader: DataLoader for the test set.
        device: 'cuda', 'mps', or 'cpu'.
        k_values: List of K values for Recall@K (e.g., [1, 5, 10]).
        max_samples: Optional limit on number of samples to evaluate (for speed).
    
    Returns:
        dict: containing 'i2t_recall' and 't2i_recall' dictionaries mapping k to score.
    """
    model.eval()
    
    image_feats_all = []
    text_feats_all = []
    
    print(f"Extracting features for Recall calculation (max_samples={max_samples})...")
    
    with torch.no_grad():
        count = 0
        for batch in tqdm(dataloader):
            images, captions = batch
            
            # Move to device
            visual_feats = images.to(device)
            if isinstance(captions, dict):
                input_ids = captions["input_ids"].to(device)
                attention_mask = captions["attention_mask"].to(device)
            else:
                # Should not happen with the way collate_fn is set up in get_dataloaders
                continue
                
            # Forward pass (Uni-modal)
            # We need to process image and text separately to get embeddings
            # The QFormer forward function takes both, but for retrieval we want 
            # independent embeddings. However, QFormer architecture typically outputs 
            # query embeddings (visual) and text embeddings.
            
            # Note: QFormer's forward function is designed to take both visual and text 
            # inputs. If attention_mode="uni_modal", they don't attend to each other,
            # but they pass through the same transformer layers. 
            
            q_out, t_out = model(
                visual_feats=visual_feats,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                attention_mode="uni_modal"
            )
            
            # Pool/Select embeddings
            # q_out: [B, num_queries, H] -> Mean pool -> [B, H]
            # t_out: [B, H] (Already pooled/CLS)
            
            # Normalize
            img_emb = F.normalize(q_out, dim=1)
            txt_emb = F.normalize(t_out, dim=1)
            
            image_feats_all.append(img_emb.cpu())
            text_feats_all.append(txt_emb.cpu())
            
            count += images.size(0)
            if max_samples is not None and count >= max_samples:
                break
    
    # Concatenate all features
    image_feats = torch.cat(image_feats_all, dim=0) # [N, H]
    text_feats = torch.cat(text_feats_all, dim=0)   # [N, H]
    
    if max_samples is not None:
        image_feats = image_feats[:max_samples]
        text_feats = text_feats[:max_samples]
        
    num_samples = image_feats.size(0)
    print(f"Computing similarity matrix for {num_samples} samples...")
    
    # Similarity Matrix: [N, N]
    # sim_matrix[i, j] = cosine similarity between image i and text j
    sim_matrix = image_feats @ text_feats.t()
    
    # --- Image-to-Text Retrieval (I2T) ---
    # For each image, rank all texts.
    # Ground truth: Image i should match Text i.
    print("Calculating I2T Recall...")
    i2t_recall = {k: 0.0 for k in k_values}
    
    # Loop over each image row
    for i in range(num_samples):
        scores = sim_matrix[i] # [N] scores for image i against all texts
        # Get indices of top K scores
        # We need the max K that we are interested in (max(k_values))
        max_k = max(k_values)
        topk_vals, topk_indices = scores.topk(max_k)
        
        # Check if ground truth index (i) is in top K
        for k in k_values:
            # Check if i is in the top k indices
            if i in topk_indices[:k]:
                i2t_recall[k] += 1
                
    for k in k_values:
        i2t_recall[k] /= num_samples
        
    # --- Text-to-Image Retrieval (T2I) ---
    # For each text, rank all images.
    print("Calculating T2I Recall...")
    t2i_recall = {k: 0.0 for k in k_values}
    
    # Loop over each text column (equivalent to transposing matrix and looping rows)
    sim_matrix_t = sim_matrix.t()
    
    for i in range(num_samples):
        scores = sim_matrix_t[i] # [N] scores for text i against all images
        max_k = max(k_values)
        topk_vals, topk_indices = scores.topk(max_k)
        
        for k in k_values:
            if i in topk_indices[:k]:
                t2i_recall[k] += 1

    for k in k_values:
        t2i_recall[k] /= num_samples
        
    return {
        "i2t": i2t_recall,
        "t2i": t2i_recall,
        "num_samples": num_samples
    }
