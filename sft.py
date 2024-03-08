from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers import DataCollatorForLanguageModeling

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
model.to("mps")
model.device

raw_datasets = load_dataset(
    "iamtarun/python_code_instructions_18k_alpaca", split="train"
)
raw_datasets = raw_datasets.train_test_split(test_size=0.03)
print(
    f"Train size: {len(raw_datasets['train'])} Test size: {len(raw_datasets['test'])}"
)


def preprocess_function(data):
    return tokenizer(data["prompt"])


tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=raw_datasets["test"].column_names,
)

block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="save/4/",
    report_to="tensorboard",
    seed=4328,
    logging_steps=10,
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    warmup_steps=200,
    num_train_epochs=4,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=False)

model.save_pretrained("save/completed/4")