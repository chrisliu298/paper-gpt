from dataclasses import dataclass
from transformers import TrainingArguments


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
    train_file = "arxiv_train_mini.txt"
    validation_file = "arxiv_test_mini.txt"
    block_size = 1024
    overwrite_cache = True
    validation_split_percentage = None
    preprocessing_num_workers = 8


def TrainingArguments():
    return TrainingArguments(
        output_dir=".",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        model_parallel=False,
        evaluation_strategy="no",
        prediction_loss_only=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=4,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=3.0,
        max_steps=-1,
        # lr_scheduler_type = "linear",
        warmup_steps=0,
        logging_dir=".",
        logging_first_step=False,
        logging_steps=500,
        save_steps=500,
        save_total_limit=None,
        no_cuda=False,
        seed=42,
        fp16=False,
        fp16_opt_level="O1",
        local_rank=-1,
        tpu_num_cores=None,
        tpu_metrics_debug=False,
        debug=False,
        dataloader_drop_last=False,
        eval_steps=500,
        dataloader_num_workers=8,
        past_index=-1,
        run_name=None,
        disable_tqdm=None,
        remove_unused_columns=True,
        label_names=None,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ignore_data_skip=False,
        fp16_backend="auto",
        sharded_ddp=False,
        # label_smoothing_factor = 0.0,
        # adafactor = False,
    )
