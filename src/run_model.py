import os
import time
import string
import argparse
from datetime import timedelta, datetime

# Change the HF cache directory to a scratch directory. 
# Make sure to set the variable before importing transformers module (including indirect import through galai).
# ref: https://github.com/paperswithcode/galai/blob/main/notebooks/Introduction%20to%20Galactica%20Models.ipynb
os.environ["TRANSFORMERS_CACHE"] = "/scratch/ac.gpark/.cache/huggingface"

# ref: https://huggingface.co/docs/transformers/v4.21.1/en/troubleshooting#troubleshoot
#os.environ["CUDA_VISIBLE_DEVICES"] = "" # to run on CPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # to get a better traceback from the GPU error

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, BioGptForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Must import galai after transformers. 05/19/2023
# If not, TRANSFORMERS_CACHE directory is set to 'galactica', which overrides the env variable setting above.
import galai as gal

from data_processors import *
from evaluators import *


def get_data_processor(data_name, *argv):
    if data_name == "string":
        # pass 'task' argument for entity_relation task. 04/12/2023
        return StringProcessor(data_name, *argv)
    elif data_name == "kegg":
        return KeggProcessor(data_name, *argv)
    elif data_name == "indra":
        return IndraProcessor(data_name, *argv)
    else:
        raise ValueError("Invalid data name: " + data_name)


if __name__ == "__main__":
    """
    <Galactica>
    - To use parallelizm (Parallelformers), put the model load code under if __name__ == "__main__":
      ref: https://github.com/tunib-ai/parallelformers#do-your-processes-die-or-stop-working

    Parameter descriptions:
    
    Model list
    ============================================
    - Galactica models: mini (125M), base (1.3B), standard (6.7B), large (30B), huge (120B)
    - LLaMA:
    - reStructured: 11b
        
    Task list for each dataset
    ============================================
    - STRING: entity, relation, entity_relation
    - KEGG: entity, relation, entity_relation
    - INDRA: relation_type
    
    Set a batch size to be processed.
    - batch size: number of prompts to infer. I.e., the number of input texts for model generation at once. 
    ============================================
    - STRING entity task: 8, 16, 32
    - STRING relation & entity_relation task: 32, 64 
    - KEGG entity task: 8, 16, 32
    - KEGG relation & entity_relation task: 32, 64 
    - INDRA relation_type task: 4, 8 
    
    Best N-shots for each task
    ============================================
    - STRING (entity): 1 (3)
    - STRING (relation & entity_relation): 1 (0, 2, 3, 5)
    - KEGG (entity): 1
    - INDRA (relation_type): 2, 4
    
    """

    parser = argparse.ArgumentParser()
    
    # general arguments
    parser.add_argument('--model_name', action='store')
    parser.add_argument('--model_type', action='store')
    parser.add_argument('--data_repo_path', action='store')
    parser.add_argument('--output_dir', action='store')
    parser.add_argument('--data_name', action='store')
    parser.add_argument('--task', action='store')
    parser.add_argument('--test_sample_size', action='store', type=int)
    parser.add_argument('--batch_size', action='store', type=int)
    parser.add_argument('--n_shots', action='store', type=int)
    
    # model/data specific arguments
    parser.add_argument("--parallelizm", action="store_true", help="<Galactica> whether to use parallelizm (Parallelformers)")
    parser.add_argument("--lora_weights", action="store", help="<Alpaca> LoRA weights")
    parser.add_argument("--kegg_data_type", action="store", default="low-dose", help="<KEGG data> select either 'high-dose' or 'low-dose'")
    parser.add_argument('--num_of_indra_classes', action='store', type=int)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    model_type = args.model_type
    data_repo_path = os.path.expanduser(args.data_repo_path)
    output_dir = os.path.expanduser(args.output_dir)
    data_name = args.data_name
    task = args.task
    test_sample_size = args.test_sample_size
    batch_size = args.batch_size
    n_shots = args.n_shots
    parallelizm = args.parallelizm
    lora_weights = args.lora_weights
    kegg_data_type = args.kegg_data_type
    num_of_indra_classes = args.num_of_indra_classes
    
    
    # load a model.
    if model_name == 'Galactica':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = 'left'
        tokenizer.model_max_length = 2020
        
        # when parallelize is set, dtype is set to float16. The default dtype is float32.
        model = gal.load_model(model_type, parallelize=parallelizm)
        
        if parallelizm:
            # To fix the error of parallelize in Galactica - OSError: [Errno 9] Bad file descriptor - 02/04/2023
            # ref: https://github.com/tunib-ai/parallelformers/issues/42
            import torch
            torch.multiprocessing.set_sharing_strategy("file_system")
        
    elif model_name == 'LLaMA':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(model_type, device_map="auto")
        
    elif model_name == 'Alpaca':
        tokenizer = LlamaTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        
        #import torch
        model = LlamaForCausalLM.from_pretrained(
            model_type,
            #load_in_8bit=True,
            #torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            #torch_dtype=torch.float16,
        )
        
        # ref: https://github.com/tloen/alpaca-lora/blob/main/generate.py#L75
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        
    elif model_name == 'RST':
        # RST model (https://huggingface.co/XLab) - the model is very heavy. 01/20/2023
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_type, device_map="auto")
        
    elif model_name == 'BioGPT':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.padding_side = "left"
        model = BioGptForCausalLM.from_pretrained(model_type).to("cuda")
    
    elif model_name == 'BioMedLM':
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = GPT2LMHeadModel.from_pretrained(model_type).to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id = 28895
    
    else:
        raise ValueError("Invalid model name: " + model_name)	
    
    # run a task.
    st = time.time()
    
    if data_name == "kegg":
        # kegg_data_type parameter is only used for KEGG data.
        data_processor = get_data_processor(data_name, data_repo_path, task, test_sample_size, model_name, tokenizer, kegg_data_type)
    elif data_name == "indra":
        # num_of_indra_classes parameter is only used for INDRA data.
        data_processor = get_data_processor(data_name, data_repo_path, task, test_sample_size, model_name, tokenizer, num_of_indra_classes)
    else:
        data_processor = get_data_processor(data_name, data_repo_path, task, test_sample_size, model_name, tokenizer)
    
    data_processor.create_prompt(n_shots)	
    results = data_processor.infer(model, batch_size)
    
    et = time.time()
    elapsed_time = et - st
    exec_time = timedelta(seconds=elapsed_time)
    exec_time = str(exec_time)
    print('>> Execution time in hh:mm:ss:', exec_time)

    if len(results[task]) == 3:
        # store the source item in the query to be used in entity_relation task for STRING, KEGG. 04/12/2023
        src, pred, true = results[task]
    else:
        src = None
        pred, true = results[task]
    
    output_dir = os.path.join(output_dir, model_name)
    output_dir = os.path.join(output_dir, model_type.rsplit('/', 1)[1] if '/' in model_type else model_type)
    output_dir = os.path.join(output_dir, data_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if hasattr(data_processor.data_reader, "rel_types"):
        labels = data_processor.data_reader.rel_types
    elif task in ["relation", "entity_relation"]:
        labels = data_processor.relation_query_answers
    else:
        labels = None
    
    compute_metrics_and_save_results(
        src, 
        pred, 
        true, 
        task, 
        labels, 
        output_dir,
        batch_size,
        n_shots,
        test_sample_size,
        data_processor.task_prompt[task],
        data_name,
        kegg_data_type,
        num_of_indra_classes,
        exec_time,
    )

    # get current date and time
    current_datetime = datetime.now()
    # convert datetime obj to string
    str_current_datetime = str(current_datetime)
    print('>> Current date and time:', str_current_datetime)
