import lm_eval
from transformers import TrainerCallback


class Evaluator:
    def __init__(
        self,
        task_dict,
        task_manager,
        max_batch_size=8,
        max_new_tokens=256,
        add_bos_token=True,
        apply_chat_template=False,
    ):
        self.task_manager = task_manager
        self.task_dict = task_dict
        self.max_batch_size = max_batch_size
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.apply_chat_template = apply_chat_template

    def evaluate(self, model, tokenizer):
        lm_obj = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            truncation=False,
            trust_remote_code=True,
            add_bos_token=self.add_bos_token,
        )
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=[self.task_dict[k] for k in self.task_dict.keys()],
            num_fewshot=0,
            task_manager=self.task_manager,
            batch_size="auto",
            apply_chat_template=self.apply_chat_template,
            max_batch_size=self.max_batch_size,
            gen_kwargs=f"max_gen_toks={self.max_new_tokens}",
        )
        return results["results"]


class EvaluationCallback(TrainerCallback):
    def __init__(
        self, recorder, evaluator, eval_every_epoch=False, eval_split_name="eval"
    ):
        self.recorder = recorder
        self.evaluator = evaluator
        self.eval_every_epoch = eval_every_epoch
        self.eval_split_name = eval_split_name

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_every_epoch:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            metrics = kwargs.get("metrics")

            model.eval()
            eval_results = self.evaluator.evaluate(model, tokenizer)
            model.train()

            for task_name in self.evaluator.task_dict.keys():
                for metric in eval_results[task_name].keys():
                    if len(self.evaluator.task_dict) == 1:
                        recorder_key = "%s_%s" % (self.eval_split_name, metric)
                    else:
                        recorder_key = "%s_%s_%s" % (
                            self.eval_split_name,
                            task_name,
                            metric,
                        )
                    self.recorder.add_with_logging(
                        key=recorder_key,
                        value=eval_results[task_name][metric],
                        epoch=state.epoch,
                        step=state.global_step,
                    )
                    metrics[recorder_key] = eval_results[task_name][metric]
        return control

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            try:
                self.recorder.add_with_logging(
                    key="train_loss",
                    value=logs["loss"],
                    epoch=state.epoch,
                    step=state.global_step,
                )
            except AssertionError:
                pass
        if "eval_loss" in logs:
            try:
                self.recorder.add_with_logging(
                    key="eval_loss",
                    value=logs["eval_loss"],
                    epoch=state.epoch,
                    step=state.global_step,
                )
            except AssertionError:
                pass
        return control
