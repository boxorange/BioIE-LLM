import os
import time
import string
import argparse
from datetime import timedelta, datetime

from dataset_processor import *
from evaluator import *

import galai as gal

# change HF cache directory to scratch directory. 
# Make sure to set the variable before importing transformers module (including indirect import through galai).
# ref: https://github.com/paperswithcode/galai/blob/main/notebooks/Introduction%20to%20Galactica%20Models.ipynb
os.environ['TRANSFORMERS_CACHE'] = "/scratch/ac.gpark/.cache/galactica"

# ref: https://huggingface.co/docs/transformers/v4.21.1/en/troubleshooting#troubleshoot
#os.environ["CUDA_VISIBLE_DEVICES"] = "" # to run on CPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # to get a better traceback from the GPU error

from transformers import LlamaForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    

if __name__ == "__main__":
    """
    <Galactica>
    - To use parallelizm (Parallelformers), put the model load code under if __name__ == "__main__":
      ref: https://github.com/tunib-ai/parallelformers#do-your-processes-die-or-stop-working

    """
    
    '''
    Parameter descriptions:
    
    Model list
    ============================================
    - Galactica models: mini (125M), base (1.3B), standard (6.7B), large (30B), huge (120B)
    - LLaMA:
    - reStructured: 11b
        
    Task list for each dataset
    ============================================
    - SciERC: entity, entity_type, relation_type
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
    - SciERC (entity): TBD
    - SciERC (entity_type): TBD
    - SciERC (relation_type): TBD
    - STRING (entity): 1 (3)
    - STRING (relation & entity_relation): 1 (0, 2, 3, 5)
    - KEGG (entity): 1
    - INDRA (relation_type): 2, 4
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', action='store')
    parser.add_argument('--model_type', action='store')
    parser.add_argument('--data_repo_path', action='store')
    parser.add_argument('--output_dir', action='store')
    parser.add_argument('--data_name', action='store')
    parser.add_argument('--task', action='store')
    parser.add_argument('--batch_size', action='store', type=int)
    parser.add_argument('--n_shots', action='store', type=int)
    parser.add_argument("--parallelizm", action="store_true", help="<Galactica> whether to use parallelizm (Parallelformers)")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    model_type = args.model_type
    data_repo_path = os.path.expanduser(args.data_repo_path)
    output_dir = os.path.expanduser(args.output_dir)
    data_name = args.data_name
    task = args.task
    batch_size = args.batch_size
    n_shots = args.n_shots
    parallelizm = args.parallelizm
    
    # load model.
    if model_name == 'Galactica':
        # when parallelize is set, dtype is set to float16. The default dtype is float32.
        model = gal.load_model(model_type, parallelize=parallelizm)
        #model = gal.load_model("mini", num_gpus=6, parallelize=True) 
        
        if parallelizm:
            # To fix the error of parallelize in Galactica - OSError: [Errno 9] Bad file descriptor - 02/04/2023
            # ref: https://github.com/tunib-ai/parallelformers/issues/42
            import torch
            torch.multiprocessing.set_sharing_strategy("file_system")
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = 'left'
        tokenizer.model_max_length = 2020
        
    elif model_name == 'LLaMA':
        model = LlamaForCausalLM.from_pretrained(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token 
    
    elif model_name == 'RST':
        # RST model (https://huggingface.co/XLab) - the model is very heavy. 01/20/2023
        tokenizer = AutoTokenizer.from_pretrained("XLab/rst-all-11b")
        model = AutoModelForSeq2SeqLM.from_pretrained("XLab/rst-all-11b")

        inputs = tokenizer.encode("TEXT: this is the best cast iron skillet you will ever buy. QUERY: Is this review \"positive\" or \"negative\"", return_tensors="pt")
        outputs = model.generate(inputs)
        print(outputs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        input('enter..')


    data_processor = DatasetProcessor(data_name, data_repo_path, tokenizer, task)
    data_processor.create_prompt(task, n_shots)

    st = time.time()
    
    results = data_processor.infer(model, task, batch_size)
    
    et = time.time()
    elapsed_time = et - st
    td = timedelta(seconds=elapsed_time)
    print('>> Execution time in hh:mm:ss:', td)
    
    '''
    if task == "entity":
        pred, true = results['entity']
    elif task == "relation":
        pred, true = results['relation']
    '''
    
    # [STRING - entity task] store the source item in the query to be used in the relation task. 04/12/2023
    if len(results[task]) == 3:
        src, pred, true = results[task]
    else:
        src = None
        pred, true = results[task]
    
    output_dir = os.path.join(output_dir, data_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if hasattr(data_processor.data_reader, "rel_types"):
        labels = data_processor.data_reader.rel_types # e.g., labels = ["True", "False"]
    elif task in ["relation", "entity_relation"]:
        #labels = ["True", "False"]
        labels = ["yes", "no"]
        #labels = ["Related", "Unrelated"]
    else:
        labels = None
    
    compute_metrics(src, pred, true, task, labels, output_dir)
    
    # get current date and time
    current_datetime = datetime.now()
    # convert datetime obj to string
    str_current_datetime = str(current_datetime)
    print('>> Current date and time:', str_current_datetime)
    

    '''
    pred, true = results['entity']
    types = list(set(pred + true))
    
    # for test
    #pred = ['ChemProt_BLURB', 'DDI_BLURB', 'TACRED', 'DDI_BLURB', 'TACRED']
    #true = ['ChemProt_BLURB', 'TACRED', 'TACRED', "", ""]
    #types = list(set(pred + true))
        
    print(compute_metrics_spert(pred, true, types, True))
    '''