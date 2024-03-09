from datasets import load_dataset, concatenate_datasets
# from transformers import AutoTokenizer
# model_name = "EleutherAI/pythia-410m"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.bos_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token

raw_ds1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
raw_ds2 = load_dataset("Vezora/Tested-22k-Python-Alpaca",split="train")
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
ds2 = raw_ds2.map(ds2_preprocess_function, num_proc=4, remove_columns=raw_ds2.column_names)
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

ds1 = raw_ds1.map(ds1_preprocess_function, num_proc=4, remove_columns=raw_ds1.column_names)

dataset = concatenate_datasets([ds1, ds2]).shuffle()
dataset = dataset.train_test_split(test_size=0.05)

# def preprocess_function(data):
#     return tokenizer(data["prompt"])


# tokenized_datasets = dataset.map(
#     preprocess_function,
#     batched=True,
#     num_proc=4,
#     remove_columns=dataset["test"].column_names,
# )

# block_size = 512


# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of block_size.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

dataset.save_to_disk("data/trl_ds_40k")