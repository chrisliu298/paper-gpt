import logging
import math
import os
from dataclasses import dataclass

from datasets import load_dataset

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


# Logging
logger = logging.getLogger(__name__)


# Variables
START_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|sep|>"
END_TOKEN = "<|endoftext|>"


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
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


@dataclass
class ModelArguments:
    model_name_or_path = "gpt2"
    model_type = "gpt2"
    config_name = "gpt2"
    tokenizer_name = "gpt2"
    cache_dir = None
    use_fast_tokenizer = True


@dataclass
class DataTrainingArguments:
    dataset_name = None
    dataset_config_name = None
    train_file = "paper_train.txt"
    validation_file = None
    block_size = 512
    overwrite_cache = True
    validation_split_percentage = 0.2
    preprocessing_num_workers = 2


@dataclass
class TrainingArguments:
    output_dir = "."
    overwrite_output_dir = True
    do_train = True
    do_eval = True
    do_predict = False
    model_parallel = False
    evaluation_strategy = "no"
    prediction_loss_only = False
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    per_gpu_train_batch_size = 8
    per_gpu_eval_batch_size = 8
    gradient_accumulation_steps = 1
    eval_accumulation_steps = 1
    learning_rate = 5e-5
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    max_steps = -1
    lr_scheduler_type = "linear"
    warmup_steps = 0
    logging_dir = "."
    logging_first_step = False
    logging_steps = 500
    save_steps = 500
    save_total_limit = None
    no_cuda = False
    seed = 42
    fp16 = False
    fp16_opt_level = "O1"
    local_rank = -1
    tpu_num_cores = None
    tpu_metrics_debug = False
    debug = False
    dataloader_drop_last = False
    eval_steps = 500
    dataloader_num_workers = 2
    past_index = -1
    run_name = None
    disable_tqdm = None
    remove_unused_columns = True
    label_names = None
    load_best_model_at_end = False
    metric_for_best_model = None
    greater_is_better = None
    ignore_data_skip = False
    fp16_backend = "auto"
    sharded_ddp = False
    label_smoothing_factor = 0.0
    adafactor = False


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
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
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

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
# 'text' is found. You can easily tweak this behavior (see below).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
extension = data_args.train_file.split(".")[-1]
if extension == "txt":
    extension = "text"
datasets = load_dataset(extension, data_files=data_files)
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

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

# Preprocessing the datasets.
# First we tokenize all the texts.
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


# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
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
model_path = model_args.model_name_or_path
train_result = trainer.train(model_path=model_path)
trainer.save_model()

output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
if trainer.is_world_process_zero():
    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

# Evaluation
results = {}
logger.info("*** Evaluate ***")

eval_output = trainer.evaluate()

perplexity = math.exp(eval_output["eval_loss"])
results["perplexity"] = perplexity

output_eval_file = os.path.join(training_args.output_dir, "eval_results_clm.txt")
if trainer.is_world_process_zero():
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")
