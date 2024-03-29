from datasets import load_dataset, concatenate_datasets
# from transformers import AutoTokenizer
# model_name = "EleutherAI/pythia-410m"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.bos_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token

raw_ds1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
raw_ds2 = load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")
# raw_ds3 = load_dataset("flytech/python-codes-25k")


def ds2_preprocess_function(data):
    return {
        "prompt": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: 
{data["instruction"]}
### Input: 
### Output: 
{data["output"]}
"""
    }


ds2 = raw_ds2.map(
    ds2_preprocess_function, num_proc=4, remove_columns=raw_ds2.column_names
)
# print(ds2["train"][0])
# ds2.save_to_disk("data/ds_22k")


def ds1_preprocess_function(data):
    return {
        "prompt": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: 
{data["instruction"]}
### Input: 
{data["input"]}
### Output: 
```python
{data["output"]}
```
"""
    }


ds1 = raw_ds1.map(
    ds1_preprocess_function, num_proc=4, remove_columns=raw_ds1.column_names
)

dataset = concatenate_datasets([ds1, ds2]).shuffle()
dataset = dataset.train_test_split(test_size=0.05)
print(dataset["train"].num_rows, dataset["test"].num_rows)
print(dataset["train"][0]["prompt"])

dataset.save_to_disk("data/trl_ds_40k")
