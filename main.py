import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from zarth_utils.config import Config
from zarth_utils.logger import get_logger
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.recorder import Recorder
from data_utils import get_dataset
from eval_utils import Evaluator, EvaluationCallback

logging.set_verbosity_error()  # only errors, no warnings


def main():
    config = Config(
        default_config_dict={
            "exp_name": "arc_easy/pythia_70m/0",
            "task_name": "arc_easy",
            "use_git": True,
            "random_seed": 0,
            "base_model": "EleutherAI/pythia-70m",
            "dataset_param": {
                "max_num_train": -1,
                "max_num_valid": -1,
                "max_num_test": -1,
                "apply_chat_template": False,
            },
            "only_evaluate": False,
            "eval_max_batch_size": 8,
            "eval_before_train": False,
            "eval_after_train": True,
            "eval_every_epoch": False,
            "resume_from_checkpoint": False,
            "completion_only": False,
            "max_new_tokens": 256,
            "train_param": {
                "max_seq_length": 1024,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "auto_find_batch_size": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "num_train_epochs": 5,
                "max_steps": -1,
                "eval_strategy": "epoch",
                "logging_strategy": "epoch",
                "save_strategy": "epoch",
                "eval_steps": 500,
                "logging_steps": 500,
                "save_steps": 500,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "eval_accumulation_steps": 1,
                "optim": "adamw_torch",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 0,
                "report_to": "tensorboard",
                "dataset_text_field": "text",
                "save_safetensors": True,
                "packing": False,
            },
            "use_peft": True,
            "peft_config": {
                "task_type": "CAUSAL_LM",
                "inference_mode": False,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
        },
    )
    config.show()

    dir_base = os.getcwd()
    dir_results = os.path.join(dir_base, "results", config["exp_name"])
    all_paths = get_all_paths(dir_results)
    config.dump(all_paths["path_config"])
    recorder = Recorder(
        all_paths["path_record"], config=config, use_git=config["use_git"]
    )
    get_logger(all_paths["path_log"])

    ###################################################################################
    # Prepare the benchmark
    ###################################################################################
    train_dataset, valid_dataset, task_dict, valid_task_dict, task_manager = (
        get_dataset(
            [config.task_name],
            config["random_seed"],
            **config["dataset_param"],
            use_system_role=False if "gemma" in config["base_model"].lower() else True,
        )
    )
    validator = Evaluator(
        valid_task_dict,
        task_manager,
        max_batch_size=config["eval_max_batch_size"],
        apply_chat_template=config["dataset_param"]["apply_chat_template"],
    )
    testor = Evaluator(
        task_dict,
        task_manager,
        max_batch_size=config["eval_max_batch_size"],
        apply_chat_template=config["dataset_param"]["apply_chat_template"],
    )

    ###################################################################################
    # Prepare the pre-trained model
    ###################################################################################
    set_random_seed(config["random_seed"])

    if config["only_evaluate"] and not config["use_peft"]:
        tokenizer = AutoTokenizer.from_pretrained(dir_results, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            dir_results, device_map="auto", trust_remote_code=True
        )
        model.eval()
    elif config["only_evaluate"] and config["use_peft"]:
        tokenizer = AutoTokenizer.from_pretrained(dir_results, trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            dir_results, device_map="auto", trust_remote_code=True
        )
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], trust_remote_code=True
        )
        tokenizer.padding_side = "right"
        if (
            "gemma" in config["base_model"].lower()
            or "deepseek" in config["base_model"].lower()
        ):
            attn_implementation = "eager"
        else:
            attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            use_cache=not config["train_param.gradient_checkpointing"],
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if model.generation_config is not None:
            model.generation_config.max_new_tokens = config["max_new_tokens"]
        if config["eval_before_train"]:
            model.eval()
            eval_results = validator.evaluate(model, tokenizer)
            for metric in eval_results[config.task_name].keys():
                recorder.add_with_logging(
                    key="before_train_%s_%s" % ("eval", metric),
                    value=eval_results[config.task_name][metric],
                )
            eval_results = testor.evaluate(model, tokenizer)
            for metric in eval_results[config.task_name].keys():
                recorder.add_with_logging(
                    key="before_train_%s_%s" % ("test", metric),
                    value=eval_results[config.task_name][metric],
                )
        model.train()

    pad_token_not_exist = tokenizer.pad_token is None
    if pad_token_not_exist:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    ###################################################################################
    # Training
    ###################################################################################
    if not config["only_evaluate"]:
        training_params = SFTConfig(
            output_dir=dir_results,
            **config["train_param"],
        )
        peft_config = (
            LoraConfig(**config["peft_config"]) if config["use_peft"] else None
        )

        if config["completion_only"]:
            collator = DataCollatorForCompletionOnlyLM("\nAnswer:", tokenizer=tokenizer)
        else:
            collator = None

        valid_callback = EvaluationCallback(
            recorder=recorder,
            evaluator=validator,
            eval_every_epoch=config["eval_every_epoch"],
            eval_split_name="eval",
        )
        test_callback = EvaluationCallback(
            recorder=recorder,
            evaluator=testor,
            eval_every_epoch=config["eval_every_epoch"],
            eval_split_name="test",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            processing_class=tokenizer,
            args=training_params,
            peft_config=peft_config,
            data_collator=collator,
            callbacks=[valid_callback, test_callback],
        )
        trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])
        if (
            config.train_param.num_train_epochs > 0
            and config.train_param.load_best_model_at_end
        ):
            trainer.save_model()
            recorder.add_with_logging(
                key="best_checkpoint",
                value=trainer.state.best_model_checkpoint,
            )
            recorder.add_with_logging(
                key="best_global_step",
                value=trainer.state.best_global_step,
            )
            recorder.add_with_logging(
                key="best_eval_score",
                value=trainer.state.best_metric,
            )
            all_steps = [
                k
                for k in recorder.keys()
                if k.startswith("epoch_")
                and k.endswith(config.train_param.metric_for_best_model)
            ]
            sgn = 1 if config.train_param.greater_is_better else -1
            best_step = max(all_steps, key=lambda x: sgn * recorder[x])
            if config["eval_every_epoch"]:
                recorder.add_with_logging(
                    key="best_test_score",
                    value=recorder[best_step.replace("eval", "test")],
                )

        model = trainer.model

    ###################################################################################
    # Evaluation
    ###################################################################################
    if config["eval_after_train"] or config["only_evaluate"]:
        eval_results = validator.evaluate(model, tokenizer)
        for metric in eval_results[config.task_name].keys():
            recorder.add_with_logging(
                key="%s_%s" % ("eval", metric),
                value=eval_results[config.task_name][metric],
            )
        eval_results = testor.evaluate(model, tokenizer)
        for metric in eval_results[config.task_name].keys():
            recorder.add_with_logging(
                key="%s_%s" % ("test", metric),
                value=eval_results[config.task_name][metric],
            )
    recorder.end_recording()


if __name__ == "__main__":
    main()
