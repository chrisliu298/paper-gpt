import logging
import math
import os

from datasets import load_dataset

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import training_args
from transformers.trainer_utils import is_main_process

from args import DataTrainingArguments, ModelArguments, TrainingArguments


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Logging
logger = logging.getLogger(__name__)

# Variables
START_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|sep|>"
END_TOKEN = "<|endoftext|>"

# Arguments for model, data, and training
model_args = ModelArguments()
data_args = DataTrainingArguments()
training_args = TrainingArguments()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info("Training/evaluation parameters %s", training_args)

# Set seed before initializing model.
set_seed(training_args.seed)

# Data preparation
data_files = {}

if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file

extension = data_args.train_file.split(".")[-1]

if extension == "txt":
    extension = "text"

datasets = load_dataset(extension, data_files=data_files)

# Load config, tokenizer, and model
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path, cache_dir=model_args.cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
)

model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

tokenizer.add_special_tokens({"bos_token": START_TOKEN})
tokenizer.add_special_tokens({"sep_token": SEP_TOKEN})
tokenizer.add_special_tokens({"eos_token": END_TOKEN})
model.resize_token_embeddings(len(tokenizer))

column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not data_args.overwrite_cache,
)

block_size = data_args.block_size

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"] if training_args.do_train else None,
    eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
)

# Training
train_result = trainer.train(model_path=model_args.model_name_or_path)
# trainer.save_model()
output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
if trainer.is_world_process_zero():
    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        # for key, value in sorted(train_result.items()):
        logger.info(f"  loss = {train_result.training_loss}")
        writer.write(f"loss = {train_result.training_loss}\n")

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    # trainer.state.save_to_json(
    #     os.path.join(training_args.output_dir, "trainer_state.json")
    # )

# Evaluation
eval_results = {}
logger.info("*** Evaluate ***")
eval_output = trainer.evaluate()
perplexity = math.exp(eval_output["eval_loss"])
eval_results["perplexity"] = perplexity

output_eval_file = os.path.join(training_args.output_dir, "eval_results_clm.txt")
if trainer.is_world_process_zero():
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(eval_results.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")