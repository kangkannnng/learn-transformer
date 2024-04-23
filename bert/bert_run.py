import os
import warnings

warnings.filterwarnings('ignore')

from transformers import BertTokenizerFast, BertForMaskedLM, pipeline

model_path = "pretrained-bert"

model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-8000"))

tokenizer = BertTokenizerFast.from_pretrained(model_path)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

examples = [
    "The [MASK] is the largest planet in the solar system.",
    "The capital of France is [MASK]."
]

for example in examples:
    for prection in fill_mask(example):
        print(f"{prection['sequence']} with confidence {prection['score']}")
    print("="*50)