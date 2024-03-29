from transformers import (
    TrainingArguments,
)

# from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
import signal
from datetime import datetime
from datasets import load_from_disk


current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
save_dir = "save/trl-sft/"


ready_to_save = False

def signal_handler(sig, frame):
    if ready_to_save:
        sft_trainer.save_model(f"{save_dir}/incomplete/{current_datetime}")
        print("Model saved")
    else:
        print("Model not saved")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

dataset = load_from_disk("data/trl_ds_40k")

training_args = TrainingArguments(
    output_dir=f"{save_dir}/log/{current_datetime}",
    report_to="tensorboard",
    seed=4328,
    logging_steps=5,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    # warmup_steps=200,
    num_train_epochs=0.5,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=500,
)

sft_trainer = SFTTrainer(
    "save/trl-sft/log/20240309153857/checkpoint-3500",
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    packing=True,
    dataset_text_field="prompt",
    max_seq_length=512,
    args=training_args,
)

ready_to_save = True

sft_trainer.train(resume_from_checkpoint=False)

sft_trainer.save_model(f"{save_dir}/completed/{current_datetime}")
