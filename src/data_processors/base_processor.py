import os
import sys
import re
import random
import torch, gc
import copy
import math
import logging

from abc import abstractmethod
from itertools import chain
from torch import distributed as dist
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm.auto import tqdm

from peft import (
    PeftModel, 
    PeftConfig, 
    LoraConfig, 
    TaskType,
    get_peft_config, 
    get_peft_model, 
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    SchedulerType,
    get_scheduler,
)

from trl import SFTTrainer

from accelerate import Accelerator, DistributedType
from accelerate.utils import gather_object
from accelerate.logging import get_logger
        
# setting path
sys.path.append('../prompters')
from prompters import *

from evaluators import compute_metrics

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class BaseProcessor:
    def __init__(self, *argv):
        self.data_name = argv[0]
        self.data_repo_path = argv[1]
        self.task = argv[2]
        self.test_sample_size = argv[3]
        self.model_name = argv[4]
        self.tokenizer = argv[5]
        self.model_max_len = argv[6]
        self.generation_config = argv[7]
        
        self.model_prompt = self.get_model_prompt()		
        self.task_prompt = {}
        self.shot_samples = []
        
        self.relation_query_answers = ['yes', 'no']
        
        self.train_dataset = self.val_dataset = self.test_dataset = None
        
        self.results = {self.task: {'preprocessed': [], 'original': []}}


    @abstractmethod
    def generate_datasets(
        self,
        n_shots: int,
        is_training: False,
    ):
        raise NotImplementedError
    
    @abstractmethod
    def format_dataset(
        self, 
        dataset, 
        data_type
    ):
        raise NotImplementedError
        
    # deprecated.
    @abstractmethod
    def infer(
        self,
        model, 
        generation_config, 
        batch_size: int,
    ):
        raise NotImplementedError
    
    
    @abstractmethod
    def update_results(
        self,
        decoded_entity, 
        decoded_pred, 
        decoded_gold
    ):
        raise NotImplementedError
        
        
    def infer_by_accelerator(
        self,
        model, 
        data_type: str, # validation or test
        batch_size: int,
    ):
        task = self.task
        
        if data_type == "validation":
            eval_data = self.val_dataset
        elif data_type == "test":
            eval_data = self.test_dataset
        
        accelerator = Accelerator(cpu=False, mixed_precision=None)		
        # accelerator = Accelerator(cpu=False, mixed_precision="bf16")		

        # Apply the method we just defined to all the examples in all the splits of the dataset
        # starting with the main process first:
        with accelerator.main_process_first():
            formatted_datasets = self.format_dataset(eval_data, data_type)
        
        def collate_fn(examples):
            inputs = [x["text"] for x in examples]
            labels = [x["answer"] for x in examples]
            entities = [x["entity"] for x in examples]
            
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            labels = self.tokenizer(labels, padding=True, return_tensors="pt")
            entities = self.tokenizer(entities, padding=True, return_tensors="pt")

            # Check if the input length doesn't exceed the model max input length.
            # assert self.model_max_len >= len(model_inputs["input_ids"][0]) + self.generation_config.max_new_tokens
        
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["entities"] = entities["input_ids"]
            
            return model_inputs
            
        # eval_dataloader = DataLoader(formatted_datasets, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)	
        eval_dataloader = DataLoader(formatted_datasets, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)	

        model, _, _, eval_dataloader, _ = accelerator.prepare(
            model, None, None, eval_dataloader, None
        )
        
        if accelerator.is_local_main_process:
            num_of_processed_samples = 0	
            
        for step, batch in enumerate(eval_dataloader):
            
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch.to(accelerator.device)
            with torch.no_grad():
                # outputs = model(**batch)
                # outputs = model.generate(input_ids=batch["input_ids"], generation_config=self.generation_config)
                
                # ref: https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/train_gsm8k_llama.py#L477
                gen_kwargs = {}
                gen_kwargs["input_ids"] = batch["input_ids"]
                gen_kwargs["attention_mask"] = batch["attention_mask"]
                generated_tokens = accelerator.unwrap_model(model).generate(**gen_kwargs, generation_config=self.generation_config)
            
            if self.model_name == 'RST':
                pred_tokens = generated_tokens
            else:
                max_source_length = batch["input_ids"].shape[1]
                pred_tokens = generated_tokens[:, max_source_length :]
                
            pred_tokens = accelerator.pad_across_processes(pred_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)
            gold_tokens = batch["labels"]
            entity_tokens = batch["entities"]

            gold_tokens = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)
            entity_tokens = accelerator.pad_across_processes(batch["entities"], dim=1, pad_index=self.tokenizer.pad_token_id)

            entity_tokens, pred_tokens, gold_tokens = accelerator.gather_for_metrics((entity_tokens, pred_tokens, gold_tokens))
            entity_tokens, pred_tokens, gold_tokens = entity_tokens.cpu().numpy(), pred_tokens.cpu().numpy(), gold_tokens.cpu().numpy()

            decoded_entity = self.tokenizer.batch_decode(entity_tokens, skip_special_tokens=True)
            decoded_pred = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            decoded_gold = self.tokenizer.batch_decode(gold_tokens, skip_special_tokens=True)
            
            self.update_results(decoded_entity, decoded_pred, decoded_gold)
                
            if accelerator.is_local_main_process:
                num_of_processed_samples += len(pred_tokens)
                accelerator.print(f">> the number of processed samples: {num_of_processed_samples} / total samples: {len(eval_data)}")


    def finetune(
        self,
        model,
        model_type: str,
        train_batch_size: int,
        validation_batch_size: int,
        output_dir: str,
    ):
        set_seed(42)
                
        # wandb params
        wandb_project = "BioIE-LLM"
        wandb_run_name = model_type #run_name="bert-base-high-lr",  # name of the W&B run (optional)
        # wandb_run_name = model_type + "_4" #run_name="bert-base-high-lr",  # name of the W&B run (optional)
        wandb_watch = "all"  # options: false | gradients | all
        wandb_log_model = ""  # options: false | true
        
        # Check if parameter passed or if set within environ
        # use_wandb = len(wandb_project) > 0 or (
            # "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        # )
        # Only overwrite environ if wandb param passed
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
            
        
        ## TODO: get the seed as an argument.
        # self.train_dataset = self.train_dataset.shuffle(seed=42)
        

        '''
        - HF LoRA config file: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
        - HF PEFT config file: https://github.com/huggingface/peft/blob/main/src/peft/config.py
        '''
        ## TODO: find and define target modules of 'BioGPT', 'BioMedLM', 'RST'.
        if self.model_name == 'Galactica':
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "lm_head"]
        
        if self.model_name in ['LLaMA', 'Alpaca', 'LLaMA-2', 'LLaMA-3', 'Solar'] or model_type == 'mistralai/Mistral-7B-Instruct-v0.2':
            '''
            - ref: https://github.com/git-cloner/llama2-lora-fine-tuning/blob/main/finetune-lora.sh
            target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
            
            - ref: https://github.com/tloen/alpaca-lora/
            target_modules = ['q_proj','k_proj','v_proj','o_proj']
            '''
            
            # ref: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
            # target_modules = ["q_proj", "v_proj"] # If only targeting attention blocks of the model
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"] # If targeting all linear layers

        elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            # ref: https://github.com/brevdev/notebooks/blob/main/mixtral-finetune.ipynb
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"]
            
        elif self.model_name == 'Falcon':
            # ref: https://www.labellerr.com/blog/hands-on-with-fine-tuning-llm/
            # target_modules = ["query_key_value",]
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
            # "lm_head" caused this error. -> "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cuda:0! (when checking argument for argument mat2 in method wrapper_CUDA_mm)"
            # target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "lm_head"]
        
        elif self.model_name == 'MPT':
            # ref: https://github.com/rmihaylov/mpttune -> Finetune A Base Model -> MPT-7B-CHAT
            # target_modules = ['Wqkv']
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        
        # elif self.model_name == 'BioMedLM':
            # ref: https://github.com/huggingface/peft/issues/758
            # target_modules = ['c_attn', 'c_proj'] # ['c_attn', 'c_proj'], ['up_proj', 'down_proj']
        
        lora_config = LoraConfig(
            r=8, # 8, 16
            lora_alpha=16, # 8, 16, 32
            lora_dropout=0.05, # 0, 0.05
            fan_in_fan_out=False, # False (default)
            inference_mode=False, # False (default)
            bias="none", # "none" (default) Note that it's a string not None.
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM if self.model_name == 'RST' else TaskType.CAUSAL_LM,
        )
        
        # prepare int-8 model for training
        # model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, lora_config)
        
        # total batch size = per_device_batch_size * gradient_accumulation_steps * number of devices
        training_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=train_batch_size, # options: batch_size, micro_batch_size
            per_device_eval_batch_size=validation_batch_size, # options: batch_size, micro_batch_size
            gradient_accumulation_steps=1, # options: 1 (default), 4, 8, gradient_accumulation_steps
            num_train_epochs=5,
            learning_rate=1e-4, # 5e-05 (default), 1e-4, 2e-4, 2e-5, 3e-4
            warmup_steps=100, # 5, 10, 50, 100, 400
            # max_steps=500,
            optim="paged_adamw_8bit", # "adamw_torch" (default), "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit"
            # weight_decay=0.01, # 0 (default)
            # fp16=True, # False (default) -> True causes an error for Llama model (loss is zero).
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, # 1, 10
            save_strategy="steps", # "epoch", "steps"
            eval_steps=10, # 1, 10, 200
            save_steps=10, # 1, 10, 200
            save_total_limit=3,
            load_best_model_at_end=True, # False (default)
            metric_for_best_model="eval_loss",
            evaluation_strategy="steps", # "epoch", "steps"
            output_dir=output_dir,
            overwrite_output_dir=True,
            report_to="wandb",
            run_name=wandb_run_name,
        )
        
        train_dataset = self.format_dataset(self.train_dataset, "train")
        val_dataset = self.format_dataset(self.val_dataset, "validation")

        import copy
        
        ## TODO: re-check with the reference code.
        # ref: https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/train_gsm8k_llama.py#L532
        def preprocess_function(examples):
            inputs = examples["text"]
            answer = examples["answer"]

            model_inputs = self.tokenizer(inputs, padding=True)

            labels = copy.deepcopy(model_inputs)
            
            ## TODO: label length is not correct. Fix it. Get ignore_pad_token_for_loss as an argument.
            ignore_pad_token_for_loss = True
            
            # this is a temporary code until the code above is fixed. 04-10-2024
            if ignore_pad_token_for_loss:
                # don't calculate the loss from padding (left padding)
                for i in range(len(labels["input_ids"])):
                    for j in range(len(labels["input_ids"][i])):
                        if labels["input_ids"][i][j] == 0:
                            labels["input_ids"][i][j] = -100

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # dataset = dataset.map(
            # preprocess_function,
            # batched=True,
        # )
        
        # dataset = dataset.train_test_split(test_size=test_sample_size//2, shuffle=True)
        
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
        )
        
        from trl.trainer import ConstantLengthDataset
        import warnings
        
        class CustomizedConstantLengthDataset(ConstantLengthDataset):
            def __init__(
                self,
                tokenizer,
                dataset,
                dataset_text_field=None,
                formatting_func=None,
                infinite=False,
                seq_length=1024,
                num_of_sequences=1024,
                chars_per_token=3.6,
                eos_token_id=0,
                shuffle=True,
                append_concat_token=True,
                add_special_tokens=True,
            ):
                self.tokenizer = tokenizer

                if tokenizer.eos_token_id is None:
                    warnings.warn(
                        "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                        f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
                    )

                self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
                self.dataset = dataset
                self.seq_length = seq_length
                self.infinite = infinite
                self.current_size = 0
                self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
                self.shuffle = shuffle
                self.append_concat_token = append_concat_token
                self.add_special_tokens = add_special_tokens
                # if formatting_func is None:
                    # self.formatting_func = lambda x: x[dataset_text_field]
                # else:
                    # self.formatting_func = formatting_func

                if formatting_func is not None:
                    if formatting_func.__code__.co_argcount > 1:
                        warnings.warn(
                            "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                            " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                        )
            
            def __iter__(self):
                iterator = iter(self.dataset)

                more_examples = True
                while more_examples:
                    buffer, buffer_len = [], 0
                    while True:
                        if buffer_len >= self.max_buffer_size:
                            break
                        try:
                            # buffer.append(self.formatting_func(next(iterator)))
                            buffer.append(next(iterator))
                            buffer_len += len(buffer[-1])
                        except StopIteration:
                            if self.infinite:
                                iterator = iter(self.dataset)
                                warnings.warn("The dataset reached end and the iterator is reset to the start.")
                            else:
                                more_examples = False
                                break

                    # tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                        # "input_ids"
                    # ]
                    tokenized_inputs = [x["input_ids"] for x in buffer]
                    tokenized_labels = [x["labels"] for x in buffer]

                    all_input_token_ids = []
                    all_label_token_ids = []
                    for tokenized_input, tokenized_label in zip(tokenized_inputs, tokenized_labels):
                        if self.append_concat_token:
                            tokenized_input = tokenized_input + [self.concat_token_id]
                            tokenized_label = tokenized_label + [self.concat_token_id]
                        all_input_token_ids.extend(tokenized_input)
                        all_label_token_ids.extend(tokenized_label)
                    examples = []
                    for i in range(0, len(all_input_token_ids), self.seq_length):
                        input_ids = all_input_token_ids[i : i + self.seq_length]
                        label_ids = all_label_token_ids[i : i + self.seq_length]
                        if len(input_ids) == self.seq_length:
                            examples.append((input_ids, label_ids))
                    if self.shuffle:
                        random.shuffle(examples)
                    for example in examples:
                        self.current_size += 1
                        yield {
                            "input_ids": torch.LongTensor(example[0]),
                            "labels": torch.LongTensor(example[1]),
                        }
        
        train_seq_length = len(train_dataset["input_ids"][0]) + self.generation_config.max_new_tokens
        val_seq_length = len(val_dataset["input_ids"][0]) + self.generation_config.max_new_tokens
        
        train_dataset = CustomizedConstantLengthDataset(
            self.tokenizer,
            train_dataset,
            infinite=False,
            shuffle=False,
            seq_length=train_seq_length, # 1024 (default)
            # chars_per_token=chars_per_token,
        )

        val_dataset = CustomizedConstantLengthDataset(
            self.tokenizer,
            val_dataset,
            infinite=False,
            shuffle=False,
            seq_length=val_seq_length, # 1024 (default)
            # chars_per_token=chars_per_token,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            # data_collator=default_data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            packing=True,
            tokenizer=self.tokenizer,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=self.model_max_len,
        )
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!

        
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        # model.print_trainable_parameters()
        # print(lora_config)
        # print(model)
        # print(model.config)
        # print(training_args)	

        trainer.train()
        trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))

        del model
        del trainer
        torch.cuda.empty_cache()


    def finetune_by_accelerator(
        self,
        model,
        model_type: str,
        train_batch_size: int,
        validation_batch_size: int,
        output_dir: str,
    ):
        """
        ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning
             https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/train_gsm8k_llama.py
            - HF LoRA config file: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
            - HF PEFT config file: https://github.com/huggingface/peft/blob/main/src/peft/config.py
        
        
        - Working: Mixtral 8x7B model (batch size: 2), 
        - Out of Memory Error: Mixtral 8x7B model (batch size: 4), 
        
        """
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        
        logger = get_logger(__name__)

        task = self.task

        '''
        - HF LoRA config file: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
        - HF PEFT config file: https://github.com/huggingface/peft/blob/main/src/peft/config.py
        '''
        ## TODO: find and define target modules of 'BioGPT', 'BioMedLM', 'RST'.
        if self.model_name == 'Galactica':
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "lm_head"]
        
        if self.model_name in ['LLaMA', 'Alpaca', 'LLaMA-2', 'LLaMA-3', 'Solar'] or model_type == 'mistralai/Mistral-7B-Instruct-v0.2':
            '''
            - ref: https://github.com/git-cloner/llama2-lora-fine-tuning/blob/main/finetune-lora.sh
            target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
            
            - ref: https://github.com/tloen/alpaca-lora/
            target_modules = ['q_proj','k_proj','v_proj','o_proj']
            '''
            
            # ref: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
            # target_modules = ["q_proj", "v_proj"] # If only targeting attention blocks of the model
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"] # If targeting all linear layers

        elif model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            # ref: https://github.com/brevdev/notebooks/blob/main/mixtral-finetune.ipynb
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"]
            
        elif self.model_name == 'Falcon':
            # ref: https://www.labellerr.com/blog/hands-on-with-fine-tuning-llm/
            # target_modules = ["query_key_value",]
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
            # "lm_head" caused this error. -> "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cuda:0! (when checking argument for argument mat2 in method wrapper_CUDA_mm)"
            # target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "lm_head"]
        
        elif self.model_name == 'MPT':
            # ref: https://github.com/rmihaylov/mpttune -> Finetune A Base Model -> MPT-7B-CHAT
            # target_modules = ['Wqkv']
            target_modules = ["Wqkv", "out_proj", "up_proj", "down_proj"]
        
        # elif self.model_name == 'BioMedLM':
            # ref: https://github.com/huggingface/peft/issues/758
            # target_modules = ['c_attn', 'c_proj'] # ['c_attn', 'c_proj'], ['up_proj', 'down_proj']
        
        lora_config = LoraConfig(
            r=8, # 8, 16
            lora_alpha=16, # 8, 16, 32
            lora_dropout=0.05, # 0, 0.05
            fan_in_fan_out=False, # False (default)
            inference_mode=False, # False (default)
            bias="none", # "none" (default) Note that it's a string not None.
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM if self.model_name == 'RST' else TaskType.CAUSAL_LM,
        )
        
        # prepare int-8 model for training
        # model = prepare_model_for_int8_training(model)
        # model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, lora_config)

        # Tell the Accelerator object to log with wandb
        accelerator = Accelerator(cpu=False, mixed_precision="bf16", log_with="wandb") # mixed_precision=None
        
        # test code to finetune LLaMA 2 70B model with FSDP (not working). 04-14-2024
        # ref: https://github.com/brevdev/notebooks/blob/main/mixtral-finetune.ipynb 
        # ref: https://github.com/AnswerDotAI/fsdp_qlora?tab=readme-ov-file
        '''
        from accelerate import FullyShardedDataParallelPlugin, Accelerator
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        '''
        
        # Set the training seed for reproducible training.
        set_seed(42)
        
        accelerator.wait_for_everyone()
        
        ## TODO: re-check with the reference code.
        # ref: https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/train_gsm8k_llama.py#L532
        def preprocess_function_train(examples):
            inputs = examples["text"]
            answer = examples["answer"]

            model_inputs = self.tokenizer(inputs, padding=True)

            labels = copy.deepcopy(model_inputs)
            
            '''
            ## TODO: get ignore_pad_token_for_loss as an argument.
            ignore_pad_token_for_loss = True
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if ignore_pad_token_for_loss:
                # get the length of the target tokens. -1 to kick out the <BOS> token
                label_tokens = self.tokenizer(answer, padding=False)
                # label_len = [len(label) - 1for label in label_tokens["input_ids"]]
                label_len = [len(label) for label in label_tokens["input_ids"]]

                # don't calculate the loss from source and padding (left padding)
                for i in range(len(labels["input_ids"])):
                    
                    # print(labels["input_ids"][i])
                    
                    # decoded_string = self.tokenizer.decode(labels["input_ids"][i])
                    # print('>> labels:', decoded_string)

                    # labels["input_ids"][i, : -label_len[i]] = -100 # Error!!
                    for j in range(len(labels["input_ids"][i]) - label_len[i]):
                        labels["input_ids"][i][j] = -100

                    # print(labels["input_ids"][i])
                    # input('enter..')
            '''
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        
        def preprocess_function_test(examples):
            inputs = examples["text"]
            labels = examples["answer"]
            # entities = examples["entity"]

            # inputs = [source + task_prompt for source in sources]

            # model_inputs = self.tokenizer(inputs, padding=True, truncation=True)
            model_inputs = self.tokenizer(inputs, padding=True)
            # labels = self.tokenizer(labels, padding=True, truncation=True)
            labels = self.tokenizer(labels, padding=True)
            # entities = self.tokenizer(entities, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            # model_inputs["entities"] = entities["input_ids"]

            return model_inputs
        
        
        """
        # Apply the method we just defined to all the examples in all the splits of the dataset
        # starting with the main process first:
        with accelerator.main_process_first():
            

            ## TODO: move this to generate datasets function.
            
            import random
            from itertools import chain
            
            task = self.task
            
            train_dataset = self.train_dataset
            val_dataset = self.val_dataset

            
            '''
            
            data = self.data_reader.train_data
            test_sample_size = self.test_sample_size
            
            sample_keys = random.sample(sorted(data), test_sample_size//2)
                
            ## TODO: for now, only use the first name of the list. 
            # val_dataset = {k: v[0] for k, v in data.items() if k in sample_keys}
            # val_dataset = {k: list(chain.from_iterable(v)) for k, v in data.items() if k in sample_keys}
            val_dataset = {k: [x[0] for x in v] for k, v in data.items() if k in sample_keys}
                            
            for key in sample_keys:
                del data[key]
            
            ## TODO: for now, only use the first name of the list.
            # train_dataset = {k: v[0] for k, v in data.items()}
            # train_dataset = {k: list(chain.from_iterable(v)) for k, v in data.items()}
            train_dataset = {k: [x[0] for x in v] for k, v in data.items()}
            
            formatted_train_dataset = [
                                        {
                                            "text": f"{self.model_prompt['entity_q'](k)}{self.model_prompt['entity_a'](', '.join(list(chain.from_iterable(v))))}",
                                            "answer": ', '.join(list(chain.from_iterable(v))) # use all names.
                                            # "answer": '__DELIMITER__'.join([x[0] for x in v])) # only use the first name of the list. 
                                        }
                                        for k, v in train_dataset.items()
                                    ]
            '''
            
            formatted_train_dataset = [
                                        {
                                            "text": f"{self.model_prompt['entity_type_q'](i['entity'], i['text'], self.ent_type_multiple_choices_str)}{self.model_prompt['entity_type_a'](i['label'])}",
                                            "answer": self.model_prompt['entity_type_a'](i['label'])
                                        }
                                        for i in train_dataset
                                    ]
                                    
            formatted_train_dataset = Dataset.from_list(formatted_train_dataset)
            column_names = formatted_train_dataset.column_names
            
            '''
            train_dataset = formatted_train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on training dataset",
            )
            '''
            train_dataset = formatted_train_dataset.map(
                preprocess_function_train,
                batched=True,
                remove_columns=column_names,
            )
            
            '''
            formatted_val_dataset = [
                                        {
                                            "entity": k,
                                            "text": f"{self.task_prompt[task]}{self.model_prompt['entity_q'](k)}",
                                            "answer": '__DELIMITER__'.join(list(chain.from_iterable(v))) # use all names.
                                            # "answer": '__DELIMITER__'.join([x[0] for x in v])) # only use the first name of the list. 
                                        }
                                        for k, v in val_dataset.items()
                                    ]
            '''
            
            formatted_val_dataset = [
                                        {
                                            "entity": i['entity'],
                                            "text": f"{self.task_prompt[task]}{self.model_prompt['entity_type_q'](i['entity'], i['text'], self.ent_type_multiple_choices_str)}",
                                            "answer": i['label']
                                        }
                                        for i in val_dataset
                                    ]
                                    
            formatted_val_dataset = Dataset.from_list(formatted_val_dataset)
            '''
            val_dataset = formatted_val_dataset.map(
                preprocess_function_test,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
            '''
            val_dataset = formatted_val_dataset.map(
                preprocess_function_test,
                batched=True,
            )
            
            
            
            train_dataset = self.format_dataset(self.train_dataset, "train")
            val_dataset = self.format_dataset(self.val_dataset, "validation")
            
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
            )
            
            val_dataset = val_dataset.map(
                preprocess_function_test,
                batched=True,
            )
        """	
        
        
        with accelerator.main_process_first():
            train_dataset = self.format_dataset(self.train_dataset, "train")
            # val_dataset = self.format_dataset(self.val_dataset, "validation")
                

        # Log a few random samples from the set:
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # for index in random.sample(range(len(val_dataset)), 2):
            # logger.info(f"Sample {index} of the validation set: {val_dataset[index]}.")
        
        '''
        ## TODO: get a different batch size for train and val.
        # eval_dataloader = DataLoader(datasets, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)	
        # ref: https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/train_gsm8k_llama.py#L570
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)	
        val_dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=batch_size)	
        '''
        
        
        def collate_fn(examples):
            inputs = [x["text"] for x in examples]
            labels = [x["answer"] for x in examples]
            # entities = [x["entity"] for x in examples]
            
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            labels = self.tokenizer(labels, padding=True, return_tensors="pt")
            # entities = self.tokenizer(entities, padding=True, return_tensors="pt")
            
            # debug
            '''
            print('>> model_max_len:', model_max_len)
            print('>> len(model_inputs["input_ids"][0]):', len(model_inputs["input_ids"][0]))
            print('>> max_new_tokens:', max_new_tokens)
            '''
            
            # Check if the input length doesn't exceed the model max input length.
            assert self.model_max_len >= len(model_inputs["input_ids"][0]) + self.generation_config.max_new_tokens
        
            model_inputs["labels"] = labels["input_ids"]
            # model_inputs["entities"] = entities["input_ids"]
            
            return model_inputs
        
                
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)	
        # val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=validation_batch_size)	

        
        ## TODO: set training arguments.
        # weight_decay, learning_rate, gradient_accumulation_steps, max_train_steps, num_train_epochs, num_warmup_steps 
        weight_decay = 0 # 0, 0.01 (default)
        learning_rate = 1e-4 # 5e-05 (default), 1e-4, 2e-4, 2e-5, 3e-4
        gradient_accumulation_steps = 1 # options: 1 (default), 4, 8, gradient_accumulation_steps
        max_train_steps = None
        num_train_epochs = 5 # 5, 10
        num_warmup_steps = 100 # 5, 10, 50, 100, 400
        lr_scheduler_type = "linear" # available scheduler types: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_utils.py#L394
        per_device_train_batch_size = train_batch_size
        checkpointing_steps = "epoch"
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora" in n],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        
        # model.to(accelerator.device)
        
        # Prepare everything with our `accelerator`.
        # model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            # model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        # )
        model, optimizer, train_dataloader, _, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, None, lr_scheduler
        )
        
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        
        
        # Figure out how many steps we should save the Accelerator states
        # checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        
        '''
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("clm_no_trainer", experiment_config)
        '''

        # Initialise the wandb run, passing wandb parameters and any config information
        accelerator.init_trackers(
            project_name="BioIE-LLM", 
            config={
                "learning_rate": learning_rate, 
                "weight_decay": weight_decay, 
                "num_warmup_steps": num_warmup_steps, 
                "lr_scheduler_type": lr_scheduler_type, 
                "per_device_train_batch_size": per_device_train_batch_size,
            },
            init_kwargs={"wandb": {"name": model_type}}
        )
        
        # Train!
        total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        
        ## TODO: complete this later.
        '''
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
        '''

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, num_train_epochs):
            model.train()
            
            train_total_loss = 0
            
            ## TODO: complete this later.
            '''
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
            '''
            active_dataloader = train_dataloader
            
            for step, batch in enumerate(active_dataloader):
                
                # debug
                # accelerator.print(batch)
                # input('enter..')
                
                if self.model_name == 'Galactica':
                    del batch["token_type_ids"]
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # debug
                    # accelerator.print(loss)
                    # accelerator.print(type(loss))
                    # input('enter..')
                
                    train_total_loss += loss.detach().float()
                    
                    accelerator.backward(loss)
                    if completed_steps % 50:
                        accelerator.print(f"Epoch: {epoch} | Step: {completed_steps} | Loss: {loss}")
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        checkpointing_output_dir = f"step_{completed_steps}"
                        if output_dir is not None:
                            checkpointing_output_dir = os.path.join(output_dir, checkpointing_output_dir)
                        accelerator.save_state(checkpointing_output_dir)
                
                if completed_steps >= max_train_steps:
                    break
            
            
            # ref: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation.py
            model.eval()
            
            '''
            for step, batch in enumerate(val_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                
                val_total_loss += loss.detach().float()
            '''
            
            if self.model_name == 'MPT':
                self.tokenizer.padding_side = "left" # MPT is trained with right padding, so change it to left padding for inference.
            
            self.infer_by_accelerator(model, "validation", validation_batch_size)
            
            if len(self.results[task]['preprocessed']) == 3:
                # store the source item in the query to be used in entity_relation task for STRING, KEGG. 04/12/2023
                src, pred, true = self.results[task]['preprocessed']
            else:
                src = None
                pred, true = self.results[task]['preprocessed']

            scores = compute_metrics(pred, true)

            # reset the result dict.
            self.results[task]['preprocessed'] = []
            
            if self.model_name == 'MPT':
                self.tokenizer.padding_side = "right" # MPT is trained with right padding.
            
            if checkpointing_steps == "epoch":
                checkpointing_output_dir = f"epoch_{epoch}"
                if output_dir is not None:
                    checkpointing_output_dir = os.path.join(output_dir, checkpointing_output_dir)
                accelerator.save_state(checkpointing_output_dir)

            # Log to wandb by calling `accelerator.log`, `step` is optional
            accelerator.log(
                {
                    "train_loss": train_total_loss.item() / len(train_dataloader),
                    # "accuracy": accuracy,
                    "valid_micro f1": scores["micro_f"],
                    "valid_macro f1": scores["macro_f"],
                    # "validation_loss": val_total_loss.item() / len(val_dataloader),
                    "epoch": epoch + 1,
                },
                step=completed_steps,
            )

        accelerator.end_training()
        
        if output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                # if push_to_hub:
                    # repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    
    
    # for testing, accelerator.split_between_processes
    # from accelerate import Accelerator, DistributedType

    def get_response(self, model, generation_config, batch_input_texts, return_full_text=False):
    # for testing, accelerator.split_between_processes
    # def get_response(self, model, generation_config, batch_input_texts, accelerator, return_full_text=False):
        if self.model_name == 'BioMedLM':
            inputs = self.tokenizer.batch_encode_plus(batch_input_texts, padding=True, return_tensors="pt").to(device)
        else:
            # for testing, accelerator.split_between_processes
            inputs = self.tokenizer(batch_input_texts, padding=True, return_tensors="pt").to(device)
            # inputs = self.tokenizer(batch_input_texts, padding=True, return_tensors="pt").to(accelerator.device) 
        
        input_ids = inputs["input_ids"]
        
        # if self.model_name == 'RST':
            # if self.task in ['relation', 'relation_type']: # relation and relation_type tasks only return a selection of multiple choices (e.g., True or False), so a longer length is not necessary. 
                # max_length = len(inputs.input_ids[0]) + 5
            # else:
                # max_length = len(inputs.input_ids[0]) + 200

            # generated_sequence = model.generate(input_ids=inputs.input_ids, max_length=max_length)
        # else:   
        generated_sequence = model.generate(input_ids=input_ids, generation_config=generation_config)
        
        generated_text = self.tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
        #generated_text = self.tokenizer.batch_decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # debug
        # for i in generated_sequence:
            # print(i)
            # print(len(i))
            # input('enter..')
        # for i in generated_text:
            # print(i)
            # print(len(i))
            # input('enter..')
            
        if self.model_name != 'RST':
            # code reference from transformers pipeline text_generation 'return_full_text=False'
            # ref: https://github.com/huggingface/transformers/blob/6add3b313defc35b5d8ae3d946131aeb625e0441/src/transformers/pipelines/text_generation.py#L294C21-L294C21
            # ref: https://github.com/huggingface/transformers/issues/17117
            if not return_full_text: 
                new_text = []
                for input_id, text in zip(input_ids, generated_text):
                    prompt_length = len(self.tokenizer.decode(input_id, skip_special_tokens=True))
                    new_text.append(text[prompt_length:].strip())
                generated_text = new_text

        # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
        gc.collect()
        torch.cuda.empty_cache()
        
        return generated_text
    
    
    def clean_response(
        self, 
        response, 
        true=None, 
        entity=None
    ):
        """
        Remove a prompt and unnecessary texts in model's generated texts.
        
        """
        
        '''
        if self.model_name == 'RST':
            if self.task == 'entity':
                cleaned_response = [x.strip() for x in response.split(",")]
                cleaned_response = [x for x in cleaned_response if len(x) > 0]
                    
                ## TODO: check if this is really needed. 02/04/2023
                cleaned_response = list(set(cleaned_response)) # remove duplicates.
            else:
                cleaned_response = re.sub(r'[^a-zA-Z]', '', response)
                cleaned_response = cleaned_response.lower()
        else:
        '''
        ## TODO: remove this. prompt removal has been handled in the get_response() function. 08/23/2023
        '''
        response = response.strip()
        response = response.replace("</s>", "")
        prompt = prompt.strip()
        prompt = prompt.replace("</s>", "")
        
        # debug
        #print('>> response:\n', response)
        #print('>> prompt:\n', prompt)
                
        if prompt not in response:
            if self.model_name == 'Falcon':
                # Falcon tokenizer don't produce the same prompt as the input prompt.
                # e.g., prompt  : It has been suggested that the mechanism of the antihypertensive effect of probiotics may be the inhibition of the ACE by IPP and VPP [ xref , xref ].
                #       response: It has been suggested that the mechanism of the antihypertensive effect of probiotics may be the inhibition of the ACE by IPP and VPP [ xref, xref ].
                ## TODO: find a better way.
                split_txt = "Answer:"
                split_idx = response.rfind(split_txt)
                response = response[split_idx+len(split_txt):]
            
            elif self.model_name in ['BioGPT', 'BioMedLM']:
                # BioGPT, BioMedLM tokenizers sometimes don't produce the same prompt as the input prompt. e.g., "u03b1" -> " u03b1", "[31]." -> [31] ."
                ## TODO: find a better way.
                split_txt = "? Answer:"
                split_idx = response.rfind(split_txt)
                response = response[split_idx+len(split_txt):]
                
            # debug
            if split_idx == -1:
                print('>> response:\n', response)
                print('>> prompt:\n', prompt)
                input('enter..')
        else:
            response = response.replace(prompt, "")
        '''
        
        if self.task == 'entity':
            # deprecated
            '''
            ## TODO: handle the cases below.
            # 1. Alpaca: ['4) Veli-like 2 (VEL2)', '1) E-cadherin (CDH1)', '7) Veli-like 5 (VEL5)', '6) Veli-like 4 (VEL4)', '2) Plectin (PLEC)', '8) Veli-', '5) Veli-like 3 (VEL3)', '3) Veli-like 1 (VEL1)'
            # 2. remove a query in the answer. LLaMA included a query in its answer sometimes. - 05/21/2023
            #     e.g., pred_entities = pred_entities.replace(self.model_prompt['entity_q'](item), "")
                    
            cleaned_response = response.replace("Answer: ", "", 1).replace("A: ", "", 1)
            cleaned_response = [x.strip() for x in cleaned_response.split(", ")]
                
            ## TODO: check if this is really needed. 02/04/2023
            cleaned_response = list(set(cleaned_response)) # remove duplicates.
            '''

            cleaned_response = []
            
            for item in true:
                item = item.strip()
                
                # debug
                #print('>> item:', item)
                #print('>> response:', response)
                
                for match in re.finditer(item, response):
                    s = match.start()
                    e = match.end()
                    
                    # check if a previous character is a part of the entity. e.g., 'EB1' and 'RHEB' are different proteins.
                    is_prev_char_part = False
                    if s != 0:
                        prev_char = response[s-1]
                        #is_prev_char_part = prev_char.isalnum()
                        is_prev_char_part = bool(re.match('[a-zA-Z0-9-_]', prev_char))
                        #print(prev_char)
                    
                    # check if a previous character is a part of the entity. e.g., 'PSD' and 'PSD-95' are different proteins.
                    is_next_char_part = False
                    if e != len(response):
                        next_char = response[e]
                        #is_next_char_part = next_char.isalnum()
                        is_next_char_part = bool(re.match('[a-zA-Z0-9-_]', next_char))
                        #print(next_char)

                    if not is_prev_char_part and not is_next_char_part:
                        #print('Found match "%s" at %d:%d' % (response[s:e], s, e))
                        cleaned_response.append(item)
                        break

        elif self.task in ['relation', 'entity_relation', 'relation_type', 'entity_type']:
            if self.task == 'relation_type':
                choices = self.data_reader.rel_types
            elif self.task == 'entity_type':
                choices = self.data_reader.ent_types
            else:
                choices = self.relation_query_answers
            
            
            if self.data_name == 'kbase':
                cleaned_response = 'None' # models (e.g., LLaMA) sometimes generate nothing. In this case, put in 'None'.
                
                entity = entity.lower()
                response = response.lower()
                response = response.replace(entity, '')
                
                c_list = [] # debug - to check if a response has multiple choices.
                for c in choices:
                    if c.lower() in response:
                        c_list.append(c.lower())
                
                if len(c_list) == 1:
                    cleaned_response = c_list[0]
                elif len(c_list) > 1: # debug - to check if a response has multiple choices.
                    print('>> response:', response)
                    print('>> entity:', entity)
                    print('>> c_list:', c_list)
                    # input('enter..')
                    
                    for i in c_list:
                        if i == true:
                            cleaned_response = i
                            break

                
            
            else:
                cleaned_response = 'None' # models (e.g., LLaMA) sometimes generate nothing. In this case, put in 'None'.
                for token in response.split(): # models (e.g., Alpaca) sometimes generate more texts besides an answer (e.g., zero shot). 05/23/2023
                    token = re.sub(r'[^a-zA-Z]', '', token)
                            
                    if any(x.lower() == token.lower() for x in choices):
                        cleaned_response = token
                        break
                        
                cleaned_response = cleaned_response.lower()
            
            """
            if self.model_name == 'Galactica':
                pred_answer = pred_answer.rsplit("\n\n", 1)[1]
                pred_answer = pred_answer.replace("Answer: ", "", 1).replace("A: ", "", 1)
                pred_answer = pred_answer.split()[-1] # remove numbers. E.g., (B) Inhibition -> Inhibition

            elif self.model_name == 'BioGPT':
                relation_type_prompt_with_test_sample = self.task_prompt[task]
                relation_type_prompt_with_test_sample += text
                relation_type_prompt_with_test_sample += self.model_prompt['relation_type_q'](entity_1, entity_2, self.rel_type_multiple_choices_str)
                relation_type_prompt_with_test_sample = relation_type_prompt_with_test_sample.replace("\n\n", " ")
                
                print(relation_type_prompt_with_test_sample)
                print(pred_answer)
                
                pred_answer = pred_answer.replace(relation_type_prompt_with_test_sample, "")
                pred_answer = pred_answer.rsplit("Answer:", 1)[1]
                pred_answer = pred_answer.split()[0]
                pred_answer = pred_answer.strip()
                
                print(pred_answer)
                
            else:
                # other models generates more texts after the choice. e.g., 'Answer: Activation\n\nThe' in Alpaca.
                pred_answer = pred_answer.rsplit("Answer:", 1)[1]
                if len(pred_answer) > 1: # LLaMA sometimes generates nothing.
                    pred_answer = pred_answer.split()[0]
                    #pred_answer = pred_answer.replace("Answer: ", "", 1).replace("A: ", "", 1)
                else:
                    # debug
                    print(pred_answer)
                    
            # debug
            '''
            if len(pred_answer.split()) == 0: # 0 means '' (no answer)
                #or pred_answer not in self.rel_type_multiple_choices_str: # if pred is not in the labels. (e.g., 'Inactivation' generated by Alpaca)
                print(">> text:", item['text'])
                print(">> entity_1:", item['entity_1'], ", entity_2:", item['entity_2'])
                print(">> pred_answer:", pred_answer, ", true_answer:", true_answer)
                print(">> orig_pred_answer:", orig_pred_answer)
                input('enter..')
                #continue
            '''	
            
            pred_answer = re.sub(r'[^a-zA-Z]', '', pred_answer)
            
            if len(pred_answer) == 0:
                pred_answer = 'None'
            """
            
        return cleaned_response

    
    ## TODO: make it cleaner later. 
    def get_model_prompt(self):
        if self.model_name == 'Galactica':
            if self.data_name == 'scierc':
                return GalacticaPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return GalacticaPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return GalacticaPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return GalacticaPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return GalacticaPrompter.get_kbase_prompt(self)
            
        elif self.model_name == 'LLaMA':
            if self.data_name == 'scierc':
                return LlamaPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return LlamaPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return LlamaPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return LlamaPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return LlamaPrompter.get_kbase_prompt(self)
                
        elif self.model_name == 'Alpaca':
            if self.data_name == 'scierc':
                return AlpacaPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return AlpacaPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return AlpacaPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return AlpacaPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return AlpacaPrompter.get_kbase_prompt(self)
                
        elif self.model_name == 'LLaMA-2':
            if self.data_name == 'scierc':
                return Llama2Prompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return Llama2Prompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return Llama2Prompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return Llama2Prompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return Llama2Prompter.get_kbase_prompt(self)
            elif self.data_name == 'lll':
                return Llama2Prompter.get_lll_prompt(self)
        
        elif self.model_name == 'LLaMA-3':
            if self.data_name == 'scierc':
                return Llama3Prompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return Llama3Prompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return Llama3Prompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return Llama3Prompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return Llama3Prompter.get_kbase_prompt(self)
            elif self.data_name == 'lll':
                return Llama3Prompter.get_lll_prompt(self)
        
        elif self.model_name == 'LLaMA-3.1':
            if self.data_name == 'scierc':
                return Llama3_1Prompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return Llama3_1Prompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return Llama3_1Prompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return Llama3_1Prompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return Llama3_1Prompter.get_kbase_prompt(self)
            elif self.data_name == 'lll':
                return Llama3_1Prompter.get_lll_prompt(self)
                    
        elif self.model_name == 'Falcon':
            if self.data_name == 'scierc':
                return FalconPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return FalconPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return FalconPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return FalconPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return FalconPrompter.get_kbase_prompt(self)
        
        elif self.model_name == 'MPT':
            if self.data_name == 'scierc':
                return MptPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return MptPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return MptPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return MptPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return MptPrompter.get_kbase_prompt(self)
                
        elif self.model_name == 'BioGPT':
            if self.data_name == 'scierc':
                return BioGPTPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return BioGPTPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return BioGPTPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return BioGPTPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return BioGPTPrompter.get_kbase_prompt(self)
                
        elif self.model_name == 'BioMedLM':
            if self.data_name == 'scierc':
                return BioMedLMPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return BioMedLMPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return BioMedLMPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return BioMedLMPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return BioMedLMPrompter.get_kbase_prompt(self)
                
        elif self.model_name == 'RST':
            if self.data_name == 'scierc':
                return RstPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return RstPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return RstPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return RstPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return RstPrompter.get_kbase_prompt(self)
        
        elif self.model_name == 'Mistral':
            if self.data_name == 'scierc':
                return MistralPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return MistralPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return MistralPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return MistralPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return MistralPrompter.get_kbase_prompt(self)
            elif self.data_name == 'lll':
                return MistralPrompter.get_lll_prompt(self)	
        
        elif self.model_name == 'Solar':
            if self.data_name == 'scierc':
                return SolarPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return SolarPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return SolarPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return SolarPrompter.get_indra_prompt(self)
            elif self.data_name == 'kbase':
                return SolarPrompter.get_kbase_prompt(self)
            elif self.data_name == 'lll':
                return SolarPrompter.get_lll_prompt(self)	
        else:
            raise ValueError("Invalid model name: " + self.model_name)
    
    
    def sort_and_pad(self, pred, true, max_entity_list_len=10):
        """
        ref: https://stackoverflow.com/questions/73428068/how-to-append-item-to-match-the-length-of-two-list-in-python
        
        """
        common_values = list(set(pred) & set(true))
        new_pred = common_values + list(set(pred) - set(common_values))
        new_true = common_values + list(set(true) - set(common_values))

        if len(new_pred) > max_entity_list_len:
            new_pred = new_pred[:max_entity_list_len]
            
        if len(new_true) > max_entity_list_len:
            new_true = new_true[:max_entity_list_len]
        
        new_pred_len = len(new_pred)
        new_true_len = len(new_true)
        
        diff = abs(new_pred_len - new_true_len)

        # padding number of elements to the end of the list
        if new_pred_len < new_true_len:
            new_pred += ['NONE'] * diff

        return new_pred, new_true
    
    
    def get_rank(self):
        """
        Get the rank of this process in distributed processes.

        Return 0 for single process case.
        """
        if dist.is_initialized():
            return dist.get_rank()
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        return 0
