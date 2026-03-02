from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


@dataclass(frozen=True)
class ROCOExample:
    image_path: Path
    caption: str


class ROCOImageCaptionDataset(Dataset):
    """
    Dataset for ROCOv2:
    - train/*.jpg
    - dataset/train_captions.csv
    """

    def __init__(
        self,
        image_dir: str | Path = "train",
        captions_csv: str | Path = "dataset/train_captions.csv",
        vit_model: str = "google/vit-base-patch16-224",
        tokenizer: Optional[str] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.captions_df = pd.read_csv(captions_csv)

        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model)
        self.vit_model = ViTModel.from_pretrained(vit_model)
        self.vit_model.to(device)

        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None

        self._examples = self._build_index()

    def _build_index(self) -> List[ROCOExample]:
        examples = []
        for _, row in self.captions_df.iterrows():
            image_id = row["ID"]
            caption = row["Caption"]

            image_path = self.image_dir / f"{image_id}.jpg"

            if image_path.exists():
                examples.append(
                    ROCOExample(
                        image_path=image_path,
                        caption=caption,
                    )
                )

        return examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ex = self._examples[idx]

        with Image.open(ex.image_path) as im:
            image = im.convert("RGB").copy()

        with torch.no_grad():
            image_inputs = self.vit_processor(
                images=image, return_tensors="pt"
            ).to(self.vit_model.device)

            image_embeddings = self.vit_model(**image_inputs).last_hidden_state

        image_embeddings = image_embeddings.squeeze(0)

        return image_embeddings, ex.caption


def collate_fn(
    batch: List[Tuple[Any, Any]],
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    images, captions = zip(*batch)

    image_tensors = torch.stack(images, dim=0).to(device)

    if tokenizer is not None:
        tokenized = tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        return image_tensors, tokenized

    return image_tensors, list(captions)


def get_dataloaders(
    batch_size=16,
    split_ratio=0.9,
    seed=42,
    vit_model="google/vit-base-patch16-224",
    tokenizer="distilbert/distilbert-base-uncased",
):

    dataset = ROCOImageCaptionDataset(
        image_dir="train",
        captions_csv="dataset/train_captions.csv",
        vit_model=vit_model,
        tokenizer=tokenizer,
    )

    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_with_tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_with_tokenizer,
    )

    return train_loader, test_loader