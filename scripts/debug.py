import torch

# This is the most direct way to prevent the JIT compiler from running
# and causing the FakeTensor conflict in your environment.
torch.compiler.disable()

import editdistance
import pandas as pd
from transformers import AutoTokenizer

import src.models


model_config_dict = {
    # "attn_implementation": "eager",
    "initial_model_name_or_path": "RylanSchaeffer/mem_model_Qwen2.5-3B_dataset_gsm8k-platinum_epochs_10_seed_0",
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
tokenizer.padding_side = "left"

questions = [
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
]

answers = [
    "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18",
    "It takes 2/2=<<2/2=1>>1 bolt of white fiber So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric #### 3",
    "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000",
]

doc_to_text = """Q: {question}

A:"""

formatted_questions = [doc_to_text.format(question=question) for question in questions]

formatted_questions_and_answers = [
    f"{formatted_question} {answer}"
    for formatted_question, answer in zip(formatted_questions, answers)
]

inputs = tokenizer(formatted_questions, return_tensors="pt", padding=True).to(
    model.device
)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,  # We want only the most likely continuation.
    temperature=0.0,
)
model_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

df = pd.DataFrame(
    {
        "ground_truth": formatted_questions_and_answers,
        "model_outputs": model_outputs,
    }
)
df["edit_distance"] = df.apply(
    lambda row: editdistance.eval(row["ground_truth"], row["model_outputs"]), axis=1
)

print(df)

print(model_outputs)
