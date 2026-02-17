from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# model_name = "Qwen/Qwen3-0.6B"
def generate_responses(
    llm,
    inputs,
    max_new_tokens,
    eos_token_id,
    top_p=0.95,
    temperature=0.5,
    do_sample=True,
):

    generated_response = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    return generated_response


llm = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

system_prompt = "Answer truthfully"
text = "Hello"

def create_prompt(x):
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x},
    ]
    chat_template = tokenizer.apply_chat_template(
        chat_prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    return chat_template

inputs = tokenizer(create_prompt(text), 
                   return_tensors="pt").to(device)
input_length = inputs["input_ids"].shape[1]
with torch.no_grad():
    outputs = generate_responses(
        llm,
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id,
    )
newly_generated_tokens = outputs[:, input_length:]
out = tokenizer.batch_decode(newly_generated_tokens, skip_special_tokens=True)

print(out)
