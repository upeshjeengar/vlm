import warnings
import os
import time

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import random
import sys

# Suppress torch warnings
torch.set_warn_always(False)

from networks.lm_to_vlm import LM_2_VLM, device
from dataset.lm_dataloader import LMDataset, get_dataset
from transformers import AutoTokenizer
import subprocess
from rich.console import Console
from rich.panel import Panel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print(f"Using device: {device}")
    # 1. Setup Model and Tokenizer
    # model_name = "Qwen/Qwen3-0.6B"
    # ckpt_path = "models/vlm_qwen_peft/latest"

    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    ckpt_path = "models/vlm_peft/latest"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LM_2_VLM(model_name=model_name, pad_token_id=tokenizer.pad_token_id)
    dataset, _ = get_dataset(tokenizer_name=model_name)
    # Check for checkpoint
    model.load_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    for idx in range(100, 120):
        sample = dataset[idx]
        image_filename = sample["image_filename"]
        caption = sample["caption"]
        # Prepare inputs
        img_tensor = sample["image"].unsqueeze(0).to(device)  # [1, patches, dim]

        # Create a fresh prompt for inference
        prompt_text = "Describe the contents of this image."
        # Construct Prefix just like training
        prefix_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Answer the user's question truthfully"},
                {"role": "user", "content": prompt_text},
            ],
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(device)

        # Ensure prefix_ids has correct shape [batch_size, seq_len]
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)

        prefix_len = prefix_ids.shape[1]

        # Compute loss on training sample (optional, for debugging)
        assistant = sample["assistant_prompt"]
        # Ensure assistant has correct shape for loss computation
        if assistant.dim() == 1:
            assistant = assistant.unsqueeze(0)

        output = model(img_tensor, prefix_ids, assistant)
        loss = output.loss
        # 3. Generate
        output_ids = model.generate(
            img=img_tensor,
            prefix_ids=prefix_ids,
            max_new_tokens=25,
            temperature=0.7,
            top_p=0.95,
        )[0]

        generated_text = tokenizer.decode(output_ids)
        # Display image and results
        print("--------------------\n")
        print(f"\nSample {idx} Loss: {loss.item():.4f}")
        try:
            subprocess.run(["chafa", image_filename], check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"\nImage: {image_filename}")

        print(f"Ground Truth Caption: {caption}")
        print(f"\nGenerated Response:\n{generated_text}")
        print("--------------------\n")


if __name__ == "__main__":
    main()
