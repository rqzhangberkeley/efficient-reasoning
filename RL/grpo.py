# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
import wandb
import datetime
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import DataCollatorWithPadding

from RL.configs import GRPOConfig
from RL.rewards import accuracy_reward_limo
from RL.rl_utils import get_tokenizer
from RL.rl_utils.callbacks import get_callbacks
from RL.rl_utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from RL.grpo_trainer import CustomGRPOTrainer
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    experiment_name: str = field(
        default="exp",
        metadata={"help": "Name of the experiment"},
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to use from dataset. -1 or None means use all data."},
    )
    eval_fraction: float = field(
        default=0.1,
        metadata={"help": "Fraction of steps per epoch between evaluations"},
    )
    eval_dataset_name: str = field(
        default="datasets/converted_aime_dataset",
        metadata={"help": "Name of the evaluation dataset"},
    )



def main(script_args, training_args, model_args):
    # RZ: The training_args is a trl.GRPOConfig object.
    # RZ: trl.GRPOConfig is a subclass of transformers.TrainingArguments.

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    # Sets up basic configuration for Python's logging system
    logging.basicConfig(
        # Defines the format of log messages with these components:
        # %(asctime)s: Timestamp
        # %(levelname)s: Level of the log (INFO, WARNING, ERROR, etc.)
        # %(name)s: Logger name
        # %(message)s: The actual log message
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        
        # Specifies the date/time format (YYYY-MM-DD HH:MM:SS)
        datefmt="%Y-%m-%d %H:%M:%S",
        
        # Defines where logs should be output
        # StreamHandler(sys.stdout) means print to console/terminal
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Gets the appropriate log level from training arguments
    # Could be DEBUG, INFO, WARNING, ERROR, or CRITICAL
    log_level = training_args.get_process_log_level()

    # Sets the log level for the main logger
    logger.setLevel(log_level)

    # Sets the same log level for the datasets library
    datasets.utils.logging.set_verbosity(log_level)

    # Sets the same log level for the transformers library
    transformers.utils.logging.set_verbosity(log_level)

    # Enables the default logging handler for transformers
    # This ensures transformers logs are processed
    transformers.utils.logging.enable_default_handler()

    # Enables explicit formatting for transformers logs
    # This makes sure transformers logs follow the same format
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    # RZ: The training_args is a subclass of transformer.TrainingArguments.
    # RZ: The local_rank and decive and other attributes are defined in transformer.TrainingArguments.__init__() using @property decorator.
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###################
    # Initialize Wandb
    ###################
    if "wandb" in training_args.report_to:
        model_name = model_args.model_name_or_path.split('/')[-1]
        dataset_name = script_args.dataset_name.split('/')[-1]
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create custom run ID
        run_id = (
            f"{script_args.experiment_name}_"  # Assuming you add this to GRPOScriptArguments
            f"{model_name}_"
            f"{dataset_name}_"
            f"G{training_args.num_generations}_"
            f"B{training_args.per_device_train_batch_size}_"
            f"beta{training_args.beta}_"
            f"lr{training_args.learning_rate}_"
            f"wd{training_args.weight_decay}_"
            f"seed{training_args.seed}_"
            f"train_bsz{training_args.per_device_train_batch_size}_"
            f"gradacc{training_args.gradient_accumulation_steps}_"
            f"{current_time}"
        )
        
        wandb.init(
            project="grpo",
            name=run_id,
            id=run_id  # This sets the run ID explicitly
        )
        logger.info(f'Wandb is initialized with run name {run_id}.')

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # RZ: This is suggested by GPT.
    # tokenizer.padding_side = 'left'
    # if tokenizer.pad_token is None:
    #     # for Qwen, often the eos_token can serve as the pad token
    #     tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    question_key = DATASET_KEYS[script_args.dataset_name]["question"]
    if script_args.dataset_name == 'GAIR/LIMO':
        dataset = load_dataset(script_args.dataset_name)
        dataset_split = 'train'
        max_samples = 817 if script_args.max_samples == -1 else script_args.max_samples
        dataset = dataset[dataset_split].shuffle(seed=0).select(range(min(max_samples, len(dataset[dataset_split]))))
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

     # load the eval_dataset
    eval_question_key = DATASET_KEYS[script_args.eval_dataset_name]["question"]
    if script_args.eval_dataset_name == 'datasets/converted_aime_dataset':
        eval_dataset = load_from_disk(script_args.eval_dataset_name)
        eval_dataset_split = 'test'
        eval_dataset_max_samples = 30
        eval_dataset = eval_dataset[eval_dataset_split].shuffle(seed=0).select(range(min(eval_dataset_max_samples, len(eval_dataset[eval_dataset_split]))))
    else:
        raise ValueError(f"Dataset {script_args.eval_dataset_name} is not supported.")

    def make_conversation(example,key):
        prompt = []
        prompt = [{
                "role": "user",
                "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {example[key]}",
            }]
        return {
            "prompt": prompt
        }
    dataset = dataset.map(make_conversation, fn_kwargs={"key": question_key})
    eval_dataset = eval_dataset.map(make_conversation, fn_kwargs={"key": eval_question_key})
    
    # Check if we're using GPUs 4-7 and this is GPU 4 (the first one)
    if torch.cuda.current_device() == 0:  # First GPU in our allocated set
        import pdb; pdb.set_trace()

    #########################################################
    # Remove messages column
    #########################################################
    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # RZ: Prepares model initialization arguments
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True
    )
    training_args.model_init_kwargs = model_kwargs # RZ: Update the model_init_kwargs in the transformers.TrainingArguments object.

    #########################################################
    # Report the total number of steps. Added by RZ.
    #########################################################
    train_dataset_size = len(dataset)
    num_processes = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    global_batch_size = training_args.per_device_train_batch_size * num_processes * training_args.gradient_accumulation_steps // training_args.num_generations # RZ: the number of prompt per GD step.
    steps_per_epoch = train_dataset_size // global_batch_size
    
    if training_args.eval_strategy == "steps":
        training_args.eval_steps = max(int(steps_per_epoch * script_args.eval_fraction), 1)
        logger.info(
            f"train data size = {train_dataset_size}"
            f"number of processes = {num_processes}"
            f"global batch size (the number of prompt per GD step) = {global_batch_size}"
            f"per device batch size = {training_args.per_device_train_batch_size}"
            f"gradient accumulation steps = {training_args.gradient_accumulation_steps}"
            f"number of generations = {training_args.num_generations}"
            f"Total number of steps per epoch: {steps_per_epoch}"
            f"Setting eval_steps to {training_args.eval_steps} "
            f"({script_args.eval_fraction*100}% of steps per epoch)"
        )

    # Initialize the GRPO trainer
    #############################
    logger.info(f'Callback = {get_callbacks(training_args, model_args)}') # RZ: By default this should be empty.

    # Get reward functions and kwargs based on dataset
    # reward_funcs_dict = get_reward_functions(script_args.dataset_name)
    
    # Extract reward functions and kwargs for each reward type
    # reward_funcs = []
    # for reward_type in script_args.reward_funcs:
    #     if reward_type in reward_funcs_dict:
    #         reward_funcs.append(reward_funcs_dict[reward_type]["reward_function"])

    # Setup the reward function.
    if script_args.dataset_name == 'GAIR/LIMO': 
        reward_funcs = [accuracy_reward_limo]
    else:
        raise ValueError(f"Dataset {script_args.dataset_name} is not supported.")

    trainer = CustomGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer, # RZ: This processing_class is the tokenizer.
    )
    trainer.log_examples = False

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint) # train

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # We comment this out because thiss gives an bug ssayig that we are not using the correct padding side, but setting padding_side = 'left' does not help.
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    # Currently, we do not push anything to hub.
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    # Creates a parser that handles three types of configs
    # RZ: The yaml filess passed to the code is automatically parsed to either class of the three config class. The parser matches each yaml key to the correct class that has that attribute.
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    # Parses your YAML file into these three config objects
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
