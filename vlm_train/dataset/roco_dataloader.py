import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, ViTModel, ViTImageProcessor
import torch

class ROCODataset(Dataset):
    def __init__(
        self,
        image_dir,
        captions_csv,
        transform=None,
        max_samples=None,
        tokenizer_name='distilbert-base-uncased',
        max_length=128,
        vit_model='google/vit-base-patch16-224',
        use_vit=True,
        device='cpu',
    ):
        self.image_dir = image_dir
        self.df = pd.read_csv(captions_csv)
        self.use_vit = use_vit
        self.device = device

        if max_samples:
            self.df = self.df.iloc[:max_samples]

        if use_vit:
            self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)
            # Move ViT to specified device for faster inference (requires num_workers=0)
            self.vit_model = ViTModel.from_pretrained(vit_model, add_pooling_layer=False)
            self.vit_model.to(device)
            self.vit_model.eval()
        else:
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ID"]
        caption = row["Caption"]

        image_path = os.path.join(
            self.image_dir,
            f"{image_id}.jpg"
        )

        image = Image.open(image_path).convert("RGB")

        if self.use_vit:
            # Process with ViT to get visual features on device
            with torch.no_grad():
                image_inputs = self.vit_processor(images=image, return_tensors="pt")
                # Move inputs to device
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                visual_feats = self.vit_model(**image_inputs).last_hidden_state  # [1, 197, 768]
                visual_feats = visual_feats.squeeze(0)  # [197, 768], stays on device
        else:
            visual_feats = self.transform(image)

        # Tokenize the caption
        text_inputs = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Remove batch dimension added by return_tensors='pt'
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        # Also return the raw caption text for downstream use
        return visual_feats, text_inputs, caption