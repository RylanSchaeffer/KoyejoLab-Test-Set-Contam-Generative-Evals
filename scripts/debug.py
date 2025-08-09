import torch

# This is the most direct way to prevent the JIT compiler from running
# and causing the FakeTensor conflict in your environment.
torch.compiler.disable()

import editdistance
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

import src.data
import src.models


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B",
    use_fast=True,
    trust_remote_code=True,
)


datasets_dict = src.data.create_dataset_for_supervised_finetuning(
    tokenizer=tokenizer,
    dataset_name="EleutherAI/minerva_math",
    max_length=2048,
)


# # GSM8K Platinum.
# questions = [
#     "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
#     "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
# ]
#
# answers = [
#     "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18",
#     "It takes 2/2=<<2/2=1>>1 bolt of white fiber So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric #### 3",
#     "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000",
# ]
#
# doc_to_text = """Q: {question}
#
# A:"""
#
# formatted_questions = [doc_to_text.format(question=question) for question in questions]


# Hendrycks MATH.
questions = [
    "What is the positive difference between $120\%$ of 30 and $130\%$ of 20?",
    'Is $f(x) = 3^x$ an even function, odd function, or neither?\n\nEnter "odd", "even", or "neither".',
    "Find the positive base $b$ in which the equation $13\cdot15=243$ is valid.",
]
answers = [
    "One hundred twenty percent of 30 is $120\cdot30\cdot\frac{1}{100}=36$, and $130\%$ of 20 is $ 130\cdot 20\cdot\frac{1}{100}=26$.  The difference between 36 and 26 is $\boxed{10}$.",
    r"Note that $f(1) = 3$ and $f(-1) = 3^{-1} = \frac{1}{3}.$  Since $f(-1)$ is not equal to $f(1)$ or $-f(1),$ $f(x)$ is $\boxed{\text{neither}}$ even nor odd.",
    "When we rewrite the above equation with the base numbers as sums of digit bundles we arrive at the following to work with: \begin{align*}\n13_b\cdot15_b&=243_b\quad\Rightarrow\\\n(b+3)(b+5)&=2b^2+4b+3\quad\Rightarrow\\\nb^2+8b+15&=2b^2+4b+3\quad\Rightarrow\\\n0&=b^2-4b-12\quad\Rightarrow\\\n0&=(b-6)(b+2).\n\end{align*} Since $b$ must be positive, the necessary base is base $\boxed{6}$.",
]

doc_to_text = "Problem: {question}\nAnswer:"

formatted_questions = [doc_to_text.format(question=question) for question in questions]

formatted_questions_and_answers = [
    f"{formatted_question} {answer}"
    for formatted_question, answer in zip(formatted_questions, answers)
]

model_config_dict = {
    # "attn_implementation": "eager",
    # "initial_model_name_or_path": "RylanSchaeffer/mem_model_Qwen2.5-3B_dataset_gsm8k-platinum_epochs_10_seed_0",
    "initial_model_name_or_path": "RylanSchaeffer/mem_model_Qwen2.5-3B_dataset_hendrycks_math_epochs_2_seed_0",
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

inputs = tokenizer(formatted_questions, return_tensors="pt", padding=True).to(
    model.device
)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
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
