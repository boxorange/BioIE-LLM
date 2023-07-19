import sys
import time
import string
import random
import re
import itertools

from itertools import chain
from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import KeggReader

from .base_processor import BaseProcessor


random.seed(42)


class KeggProcessor(BaseProcessor):
    def __init__(
        self, 
        data_name: str, 
        data_repo_path: str,
        task: str,
        test_sample_size: int, 
        model_name: str,
        tokenizer,
        kegg_data_type: str,
    ):
        super().__init__(data_name, task, test_sample_size, model_name, tokenizer)

        self.data_reader = KeggReader(data_repo_path, task, kegg_data_type, self.relation_query_answers)
            
        self.all_pos_relations = set() # used for relation task.
        self.all_neg_relations = set() # used for relation task.


    def create_prompt(
        self, 
        n_shots: int = 0
    ):
        self.model_prompt = self.get_model_prompt()
        task = self.task
        data = self.data_reader.train_data
        
        self.task_prompt[task] = ""
        
        if n_shots == 0:
            if task == "relation":
                all_genes = [] # to make negative relations.
                for pathway, related_gene_list in data.items():
                    for gene_aliases in related_gene_list:
                        for gene in gene_aliases:
                            self.all_pos_relations.add(tuple([gene, pathway]))
                            all_genes.append(gene)
                                
                all_genes = list(set(all_genes))
                
                pathways = list(data.keys())
                for pathway in pathways:
                    for gene in all_genes:
                        if tuple([gene, pathway]) not in self.all_pos_relations:
                            self.all_neg_relations.add(tuple([gene, pathway]))

                # to generate the same random samples, convert it to an ordered list. 
                self.all_pos_relations = sorted(self.all_pos_relations)
            return

        if task == "entity":
            keys = list(data.keys())
            
            # shuffle data to get random samples.
            random.shuffle(keys)
            
            # get few-shot samples.
            for key in keys:
                
                ## TODO: remove this later.
                #if len(data[key]) > 10:
                #	continue

                self.shot_samples.append((key, data[key]))
                if len(self.shot_samples) >= n_shots:
                    break

            '''
            # get few-shot samples.
            sorted_data = sorted(data.items(), key = lambda item : len(item[1]))
            for k, v in sorted_data:
                self.shot_samples.append((k, v))
                if len(self.shot_samples) >= n_shots:
                    break
            '''

            for sample in self.shot_samples:
                self.task_prompt[task] += self.model_prompt['entity_q'](sample[0])
                
                ## TODO: for now, only use the first name of the list. 
                entity_list = [x[0] for x in sample[1]]
                
                self.task_prompt[task] += self.model_prompt['entity_a'](", ".join(entity_list))
        
        elif task in ["relation", "entity_relation"]:
            # len(self.all_pos_relations): 17,552
            # len(self.all_neg_relations): 190,148
            all_genes = [] # to make negative relations.
            for pathway, related_gene_list in data.items():
                for gene_aliases in related_gene_list:
                    for gene in gene_aliases:
                        self.all_pos_relations.add(tuple([gene, pathway]))
                        all_genes.append(gene)
                            
            all_genes = list(set(all_genes))
            
            pathways = list(data.keys())
            for pathway in pathways:
                for gene in all_genes:
                    if tuple([gene, pathway]) not in self.all_pos_relations:
                        self.all_neg_relations.add(tuple([gene, pathway]))
                
            # Reproducibility - to generate the same random samples, convert it to an ordered list. 
            self.all_pos_relations = sorted(self.all_pos_relations)
            self.all_neg_relations = sorted(self.all_neg_relations)

            pos_relations = random.sample(self.all_pos_relations, n_shots)
            neg_relations = random.sample(self.all_neg_relations, n_shots)
            
            for pos_rel, neg_rel in zip(pos_relations, neg_relations):
                self.task_prompt[task] += self.model_prompt['relation_q'](pos_rel[0], pos_rel[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[0])
                self.task_prompt[task] += self.model_prompt['relation_q'](neg_rel[0], neg_rel[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[1])
            
            self.shot_samples.extend(pos_relations + neg_relations)				
            
            # exclude shot samples for inference.
            for i in pos_relations:
                self.all_pos_relations.remove(i)
            
            for i in neg_relations:
                self.all_neg_relations.remove(i)

        # debug
        if len(self.task_prompt[task]) != 0:
            print(self.task_prompt[task])
        
        #input('enter..')
        
    
    def infer(
        self,
        model, 
        batch_size: int = 1,
    ):
        test_data = self.data_reader.test_data
        task = self.task
        results = self.results
        
        ## TODO: make it cleaner later.
        if task == "entity":
            shots_keys = [x[0] for x in self.shot_samples]

        ## TODO: to reduce the number of test samples for a preliminary test. 
        if task == "relation":
            test_sample_size = self.test_sample_size
            
            if test_sample_size == -1: # -1 means using all positive and negative
                # all_pos_relations is smaller than all_neg_relations. 
                # - len(self.all_pos_relations): 17,552
                # - len(self.all_neg_relations): 190,148
                test_sample_size = len(self.all_pos_relations)

            pos_relations = random.sample(self.all_pos_relations, test_sample_size)
            neg_relations = random.sample(self.all_neg_relations, test_sample_size) # draw negative samples.

            pos_relations = [list(x) for x in pos_relations]
            for i in pos_relations:
                i.append(self.relation_query_answers[0])
            
            neg_relations = [list(x) for x in neg_relations]
            for i in neg_relations:
                i.append(self.relation_query_answers[1])
            
            test_data = pos_relations + neg_relations

        start = 0
        stop = start + batch_size

        while True:
            # ref: dict(itertools.islice(d.items(), 2)) # for dictionary
            batch_data = itertools.islice(test_data, start, stop)
            
            ## TODO: make it cleaner later.
            if task == "entity":
                batch_items = [] # debug
                batch_input_texts = []
                prompt_list = []
                true_list = []

                for item in batch_data:
                    # skip samples used in few-shots.
                    if item in shots_keys:
                        continue
                    
                    batch_items.append(item)
                    
                    entity_prompt_with_test_sample = self.task_prompt[task]
                    entity_prompt_with_test_sample += self.model_prompt['entity_q'](item)
                    
                    prompt_list.append(entity_prompt_with_test_sample)
                    batch_input_texts.append(entity_prompt_with_test_sample)
                    true_list.append(test_data[item])

                pred_list = self.get_response(model, batch_input_texts)
                
                
                # debug
                print(pred_list)
                

                for item, pred, prompt, true in zip(batch_items, pred_list, prompt_list, true_list):
                    orig_pred = pred # debug
                    
                    pred = self.clean_response(pred, prompt)

                    # use all names.
                    true = list(chain.from_iterable(true))
                    #true = [x[0] for x in true] # only use the first name of the list. 

                    pred, true = self.sort_and_pad(pred, true)

                    ## TODO: for now, check only precision. 03/14/2023
                    pred = [x for x in pred if x != 'NONE']
                    true = [x for x in true if x != 'NONE']
                    if len(pred) > len(true):
                        pred = pred[:len(true)]
                    elif len(true) > len(pred):
                        true = true[:len(pred)]
                    
                    # store the source item in the query to be used in the relation task. 04/12/2023
                    src = [item] * len(pred)
                    if len(results[task]) != 0:
                        results[task][0].extend(src)
                        results[task][1].extend(pred)
                        results[task][2].extend(true)
                    else:
                        results[task] = [src, pred, true]

                    # debug
                    
                    #print(">> The number of processed samples:", str(num))
                    print(">> Which genes are involved in:", item)
                    print(">> src:", src)
                    print(">> pred:", pred)
                    print(">> true:", true)
                    print(">> orig_pred:", orig_pred)
                    #input('enter..')
                    
            
            elif task in ["relation", "entity_relation"]:
                batch_items = [] # debug
                batch_input_texts = []
                prompt_list = []
                true_list = []

                for item in batch_data:
                    batch_items.append(item)
                    
                    relation_prompt_with_test_sample = self.task_prompt[task]
                    relation_prompt_with_test_sample += self.model_prompt['relation_q'](item[0], item[1])
                    
                    prompt_list.append(relation_prompt_with_test_sample)
                    batch_input_texts.append(relation_prompt_with_test_sample)
                    
                    # test for LLaMA - 05/21/2023
                    #batch_input_texts.append('Answer these questions:\n' + relation_prompt_with_test_sample)
                    
                    true_list.append(item[2].lower())

                pred_list = self.get_response(model, batch_input_texts)
                
                
                # debug
                print(pred_list)
                
                
                for item, pred, prompt, true in zip(batch_items, pred_list, prompt_list, true_list):
                    orig_pred = pred # debug
                    
                    pred = self.clean_response(pred, prompt)

                    if len(results[task]) != 0:
                        results[task][0].append(pred)
                        results[task][1].append(true)
                    else:
                        results[task] = [[pred], [true]]
                    
                    # debug
                    
                    print(">> orig_pred:", orig_pred)
                    print(">> item:", item)
                    print(">> pred:", pred)
                    print(">> true:", true)
                    #input('enter..')
                    
            
            #print(f">> batch processed - len(test_data): {len(test_data) - len(self.shot_samples)}, start: {start}, stop: {stop}")
            print(f">> batch processed - len(test_data): {len(test_data)}, start: {start}, stop: {stop}")

            #if stop >= (len(test_data) - len(self.shot_samples)):
            if stop >= len(test_data):
                break

            start = stop
            stop = start + batch_size 

        return results
