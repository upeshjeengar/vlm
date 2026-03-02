from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from networks.q_former import QFormer
import torch
import torch.nn as nn
import os
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class LM_2_VLM(nn.Module):
    def __init__(
        self,
        model_name,
        qformer_model_path="models/trained_qformer_1/best",
        pad_token_id=None,
    ):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

        self.qformer = QFormer.from_pretrained(qformer_model_path).to(device)
        self.adapter = nn.Sequential(
            nn.Linear(
                in_features=self.qformer.hidden_size,
                out_features=self.llm.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.llm.config.hidden_size,
                out_features=self.llm.config.hidden_size,
            ),
        ).to(device)

        self.pad_token_id = pad_token_id or self.llm.config.eos_token_id

    def forward(self, img, prefix_ids, assistant_ids):
        img_emb, _ = self.qformer.encode_image(img)
        img_emb = self.adapter(img_emb)
        img_emb = img_emb.to(dtype=self.llm.dtype)

        prefix_emb = self.llm.get_input_embeddings()(prefix_ids)
        assistant_emb = self.llm.get_input_embeddings()(assistant_ids)

        input_embs = torch.cat([prefix_emb, img_emb, assistant_emb], dim=1)

        attention_mask = torch.cat(
            [
                (prefix_ids != self.pad_token_id).long(),
                torch.ones(img_emb.size(0), img_emb.size(1), device=device).long(),
                (assistant_ids != self.pad_token_id).long(),
            ],
            dim=1,
        )

        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        prefix_labels = torch.full_like(prefix_ids, -100)
        image_labels = torch.full(
            (img_emb.shape[0], img_emb.shape[1]), -100, device=device, dtype=torch.long
        )
        assistant_labels = assistant_ids.clone()
        assistant_labels[assistant_ids == self.pad_token_id] = -100

        labels = torch.cat([prefix_labels, image_labels, assistant_labels], dim=1)

        output = self.llm(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )

        return output

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        self.qformer.save_pretrained(os.path.join(path, "qformer"))
        torch.save(self.adapter.state_dict(), os.path.join(path, "adapter.pt"))
        # Save LoRA Adapter
        self.llm.save_pretrained(os.path.join(path, "lora_adapter"))
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        # Load QFormer weights
        qformer_path = os.path.join(path, "qformer", "pytorch_model.bin")
        if os.path.exists(qformer_path):
            state_dict = torch.load(qformer_path, map_location=device)
            self.qformer.load_state_dict(state_dict)
            print("Loaded QFormer weights.")

        # Load Adapter weights
        adapter_path = os.path.join(path, "adapter.pt")
        if os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            print("Loaded Adapter weights.")

        # Load LoRA Adapter weights safely
        lora_path = os.path.join(path, "lora_adapter")
        if os.path.exists(lora_path):
            # Check for both .bin and .safetensors
            weights_file = os.path.join(lora_path, "adapter_model.bin")
            if not os.path.exists(weights_file):
                weights_file = os.path.join(lora_path, "adapter_model.safetensors")

            if os.path.exists(weights_file):
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    adapters_weights = load_file(weights_file, device=device)
                else:
                    adapters_weights = torch.load(weights_file, map_location=device)

                set_peft_model_state_dict(self.llm, adapters_weights)
                print("Loaded LoRA Adapter weights.")

    @torch.no_grad()
    def generate(
        self,
        img,
        prefix_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    ):
        img_emb, _ = self.qformer.encode_image(img)
        img_emb = self.adapter(img_emb)
        img_emb = img_emb.to(dtype=self.llm.dtype)

        prefix_emb = self.llm.get_input_embeddings()(prefix_ids)
        assistant_part = self.tokenizer.apply_chat_template(
            [{"role": "assistant", 
              "content": ""}], 
            add_generation_prompt=False)
        assistant_part = torch.tensor(assistant_part[:-2], device=prefix_emb.device).repeat(prefix_emb.shape[0], 1)
        assistant_emb = self.llm.get_input_embeddings()(assistant_part)

        inputs_embeds = torch.cat([prefix_emb, 
                                   img_emb,
                                   assistant_emb
                                   ], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
        )

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return output_ids
