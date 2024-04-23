# %%
import os
import json
import warnings

warnings.filterwarnings('ignore')

from datasets import load_dataset, load_from_disk
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from itertools import chain

# %%
# wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", cache_dir="data")
# wiki.save_to_disk("wiki")

wiki = load_from_disk("wiki")

wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

d = wiki.train_test_split(test_size=0.1)

# %%
def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, 'w') as f:
        for t in dataset["text"]:
            print(t, file=f)

dataset_to_text(d["train"], "train.txt")
dataset_to_text(d["test"], "test.txt")

# %%
special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
    ]

file = ["train.txt"]
vocab_size = 30_522
max_length = 512
truncate_longer_samples = True
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=file, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)
model_path = "pretrained-bert"

if not os.path.exists(model_path):
    os.makedirs(model_path)

tokenizer.save_model(model_path)

with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_config = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
        "vocab_size": vocab_size,
        "max_len": max_length,
    }
    json.dump(tokenizer_config, f)

tokenizer = BertTokenizerFast.from_pretrained(model_path)

print(tokenizer("Helloworld!"))

# %%
def encode_with_truncation(example):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length", return_special_tokens_mask=True)

def encode_without_truncation(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

train_dataset = d["train"].map(encode, batched=True)
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

# %%
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    
    result = {
        k: [t[i:i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        desc="Grouping texts in chunks of length {}".format(max_length),
    )
    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        desc="Grouping texts in chunks of length {}".format(max_length),
    )

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# %%
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(model_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

train_args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,
    logging_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()