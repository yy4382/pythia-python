from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk

model_name = "EleutherAI/pythia-410m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


dataset = load_from_disk("data/ds_40k")
training_args = TrainingArguments(
    output_dir="save/lora/1",
    report_to="tensorboard",
    seed=4328,
    logging_steps=20,
    per_device_train_batch_size=3,
    learning_rate=5e-4,
    # warmup_steps=200,
    num_train_epochs=0.005,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=500,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained("save/lora/completed/1")