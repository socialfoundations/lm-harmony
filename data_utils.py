import logging
import random
import torch
import numpy as np
import datasets

from copy import deepcopy
from typing import List, Optional, Union
from datasets import concatenate_datasets, disable_caching
from zarth_utils.nn_utils import random_split, set_random_seed

from lm_eval.evaluator_utils import run_task_tests
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import simple_parse_args_string

eval_logger = logging.getLogger(__name__)

disable_caching()
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def get_tasks(
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    check_integrity: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
):
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    task_dict = get_task_dict(tasks, task_manager)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                eval_logger.info(
                    f"Setting fewshot random generator seed to {fewshot_random_seed}"
                )

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    return task_dict, task_manager


def get_dataset(
    task_names,
    random_seed=0,
    max_num_train=-1,
    max_num_valid=-1,
    max_num_test=-1,
    apply_chat_template=False,
):
    task_dict, task_manager = get_tasks(tasks=task_names)
    validation_task_dict = deepcopy(task_dict)

    set_random_seed(random_seed)

    all_train, all_valid, all_test = {}, {}, {}
    for task_name in task_names:
        task = task_dict[task_name]
        assert task.has_training_docs()

        # split the dataset into train/valid/test
        if task.has_validation_docs() and task.has_test_docs():
            train_dataset = task.training_docs()
            valid_dataset = task.validation_docs()
            test_dataset = task.test_docs()
        else:
            train_dataset = task.training_docs()
            train_indices, valid_indices = random_split(
                len(train_dataset), (0.8, 0.2), random_seed
            )
            valid_dataset = train_dataset.select(valid_indices)
            train_dataset = train_dataset.select(train_indices)
            test_dataset = task.eval_docs

        class _OverrideTask(type(task_dict[task_name])):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._eval_docs = None

            @property
            def eval_docs(self):
                return self._eval_docs

        # handle the training set
        if max_num_train != -1 and len(train_dataset) > max_num_train:
            np.random.seed(random_seed)
            train_dataset = train_dataset.select(
                np.random.permutation(len(train_dataset))[:max_num_train]
            )

        # handle the validation set
        if max_num_valid != -1 and len(valid_dataset) > max_num_valid:
            np.random.seed(random_seed)
            valid_dataset = valid_dataset.select(
                np.random.permutation(len(valid_dataset))[:max_num_valid]
            )
        validation_task_dict[task_name].__class__ = _OverrideTask
        validation_task_dict[task_name]._eval_docs = valid_dataset

        # handle the test set
        if max_num_test != -1 and len(test_dataset) > max_num_test:
            np.random.seed(random_seed)
            test_dataset = test_dataset.select(
                np.random.permutation(len(test_dataset))[:max_num_test]
            )
            task_dict[task_name].__class__ = _OverrideTask
            task_dict[task_name]._eval_docs = test_dataset

        all_train[task_name] = train_dataset
        all_valid[task_name] = valid_dataset
        all_test[task_name] = test_dataset

    ret_train, ret_valid, ret_test = [], [], []
    for task_name in task_names:
        task = task_dict[task_name]

        def formatting_func(x):
            question = task.doc_to_text(x)
            target = task.doc_to_target(x)
            if task_name == "winogrande":
                choices = task.doc_to_choice(x)
                x["text"] = choices[int(question)] + " " + target
                assert not apply_chat_template
                return x

            if not apply_chat_template:
                if task.config.doc_to_choice is not None:
                    choices = task.doc_to_choice(x)
                    if type(target) is int:
                        target = choices[target]
                    x["text"] = question + " " + target
                else:
                    if type(target) is list:
                        target = " ".join(target)
                    x["text"] = question + " " + target
            else:
                if task.config.doc_to_choice is not None:
                    choices = task.doc_to_choice(x)
                    if type(target) is int:
                        target = choices[target]
                    x["prompt"] = question
                    x["completion"] = target
                else:
                    if type(target) is list:
                        target = " ".join(target)
                    x["prompt"] = question
                    x["completion"] = target

            return x

        train_dataset = train_dataset.map(
            formatting_func,
            batched=False,
            load_from_cache_file=False,
            remove_columns=[
                c
                for c in train_dataset.column_names
                if c not in ["text", "prompt", "completion"]
            ],
        )
        valid_dataset = valid_dataset.map(
            formatting_func,
            batched=False,
            load_from_cache_file=False,
            remove_columns=[
                c
                for c in valid_dataset.column_names
                if c not in ["text", "prompt", "completion"]
            ],
        )
        # test_dataset = test_dataset.map(
        #     formatting_func,
        #     batched=False,
        #     load_from_cache_file=False,
        #     remove_columns=[
        #         c
        #         for c in test_dataset.column_names
        #         if c not in ["text", "prompt", "completion"]
        #     ],
        # )

        ret_train.append(train_dataset)
        ret_valid.append(valid_dataset)
        # ret_test.append(test_dataset)

    ret_train = concatenate_datasets(ret_train)
    ret_valid = concatenate_datasets(ret_valid)
    # ret_test = concatenate_datasets(ret_test)

    return ret_train, ret_valid, task_dict, validation_task_dict, task_manager
