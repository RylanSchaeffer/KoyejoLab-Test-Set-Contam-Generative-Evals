import torch

# This is the most direct way to prevent the JIT compiler from running
# and causing the FakeTensor conflict in your environment.
torch.compiler.disable()

import editdistance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from typing import Dict, List, Tuple

import src.data
import src.models


raw_datasets = src.data.load_dataset_hendrycks_math()
test_dataset = raw_datasets["test"]
doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT

formatted_problems = [
    doc_to_text.format(problem=question, solution="").rstrip()
    for question in test_dataset["problem"]
]

formatted_questions_and_answers = [
    f"{formatted_problem} {solution}"
    for formatted_problem, solution in zip(formatted_problems, test_dataset["solution"])
]

model_config_dict = {
    "attn_implementation": "eager",
    "initial_model_name_or_path": "RylanSchaeffer/mem_model_Qwen2.5-3B_dataset_minerva_math_epochs_56_seed_0",
    # "initial_model_name_or_path": "RylanSchaeffer/mem_model_Qwen2.5-3B_dataset_hendrycks_math_epochs_2_seed_0",
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
# tokenizer.padding_side = "left"

batch_size = 256
output_texts = []

for batch_idx_start in tqdm(range(0, len(formatted_questions_and_answers), batch_size)):
    batch_formatted_problems = formatted_problems[
        batch_idx_start : batch_idx_start + batch_size
    ]
    batch_inputs = tokenizer(
        batch_formatted_problems, return_tensors="pt", padding=True
    ).to(model.device)
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # We want only the most likely continuation.
        temperature=0.0,
    )
    batch_output_texts: List[str] = tokenizer.batch_decode(
        batch_outputs, skip_special_tokens=True
    )
    output_texts.extend(batch_output_texts)

assert len(formatted_questions_and_answers) == len(output_texts)

df = pd.DataFrame(
    {
        "ground_truth": formatted_questions_and_answers[: len(output_texts)],
        "model_outputs": output_texts,
    }
)
df["edit_distance"] = df.apply(
    lambda row: editdistance.eval(row["ground_truth"], row["model_outputs"]), axis=1
)

print(df["edit_distance"].describe())

plt.close()
sns.histplot(
    data=df,
    x="edit_distance",
)
plt.yscale("log")
plt.show()

print(df)
