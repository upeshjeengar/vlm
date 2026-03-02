import numpy as np
from networks.q_former import QFormer
import torch
from transformers import DistilBertModel
from dataset.roco_dataloader import ROCODataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
qformer = QFormer(bert)
qformer.to(device)

model_id = "trained_qformer"
lr = 1e-4
batch_size = 8

train_dataset = ROCODataset(
    image_dir="train",
    captions_csv="dataset/train_captions.csv",
    max_samples=50000,
    use_vit=True,
    device=device,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=0,  # Set to 0 for macOS/MPS compatibility
)

test_dataset = ROCODataset(
    image_dir="test",
    captions_csv="dataset/test_captions.csv",
    max_samples=500,
    use_vit=True,
    device=device,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=80,
    shuffle=False,
    num_workers=0,  # Set to 0 for macOS/MPS compatibility
)
def calculate_clip_loss(v, t, tau=0.07):
    N = v.size(0)
    v = F.normalize(v, dim=1) #image embeddings
    t = F.normalize(t, dim=1) # text embeddings
    logits = v @ t.t() / tau   # (N, N), tau temperature
    labels = torch.arange(N, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels) # rows: image->text
    loss_t2i = F.cross_entropy(logits.t(), labels) # cols: text->image
    loss = 0.5 * (loss_i2t + loss_t2i)
    return loss.mean()

def run_inference(limit_batches=20):
    qformer.eval()
    losses = []
    with torch.no_grad():
        for i, (img, txt) in enumerate(test_loader):
            if i >= limit_batches:
                break

            # Data is already on device from dataset, but move text to device if needed
            if isinstance(txt, dict):
                txt = {k: v.to(device) for k, v in txt.items()}

            img_emb, txt_emb = qformer(
                visual_feats=img, 
                text_input_ids=txt["input_ids"],
                text_attention_mask=txt["attention_mask"],
                attention_mode="uni_modal"
            )
            loss = calculate_clip_loss(img_emb, txt_emb)
            losses.append(loss.item())
    qformer.train()

    if not losses:
        return float('inf')
    return np.mean(losses)


if __name__ == '__main__':
    grouped_params = qformer.get_grouped_params()
    optimizer = optim.Adam(
        [
            {"params": grouped_params["default"], "lr": lr * 0.1},
            {"params": grouped_params["cross_blocks"], "lr": lr},
            {"params": grouped_params["query_embeddings"], "lr": lr},
        ]
    )

    steps = 0
    log_train_loss_every = 50
    run_inference_every = 100
    save_checkpoint_every = 200
    best_test_loss = np.inf

    for epoch in range(10):
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for (img, txt) in pbar:
            steps += 1

            # Data is already on device from dataset, but move text to device if needed
            if isinstance(txt, dict):
                txt = {k: v.to(device) for k, v in txt.items()}

            img_emb, txt_emb = qformer(
                visual_feats=img,
                text_input_ids=txt["input_ids"],
                text_attention_mask=txt["attention_mask"],
                attention_mode="uni_modal"
            )
            loss = calculate_clip_loss(img_emb, txt_emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if steps % log_train_loss_every == 0:
                tqdm.write(f"Epoch: {epoch+1}, Steps: {steps}, Train loss: {np.mean(train_losses):.4f}")
                train_losses = []

            if steps % run_inference_every == 0:
                test_loss = run_inference()
                tqdm.write(f"Steps: {steps}, Test Loss: {test_loss:.4f}")

                if test_loss < best_test_loss:
                    best_model_dir = f"models/{model_id}/best"
                    qformer.save_pretrained(best_model_dir)
                    tqdm.write(f"New model saved in {best_model_dir}")
                    best_test_loss = test_loss

            if steps % save_checkpoint_every == 0:
                tqdm.write(f"Checkpoint saved at step {steps}")
                qformer.save_pretrained(f"models/{model_id}/latest")


