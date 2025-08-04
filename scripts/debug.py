import torch

# This is the most direct way to prevent the JIT compiler from running
# and causing the FakeTensor conflict in your environment.
torch.compiler.disable()

from transformers import AutoTokenizer

import src.models

model_config_dict = {
    # "attn_implementation": "eager",
    "initial_model_name_or_path": "RylanSchaeffer/mem_model_gemma-3-4b-it_dataset_gsm8k-platinum_epochs_56_seed_0",
    # "initial_model_name_or_path": "google/gemma-3-4b-it",
    "torch_dtype": "bfloat16",
}

model = src.models.load_automodelforcausallm(
    model_config_dict=model_config_dict,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_config_dict["initial_model_name_or_path"],
    use_fast=True,
    trust_remote_code=True,
)

questions = [
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
]

doc_to_text = """Q: {question}

A: """

formatted_questions = [doc_to_text.format(question=question) for question in questions]

inputs = tokenizer(formatted_questions, return_tensors="pt", padding=True).to(
    model.device
)
outputs = model.generate(
    **inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id
)
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(generated_texts)
