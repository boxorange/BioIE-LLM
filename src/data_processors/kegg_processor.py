import sys
import time
import string
import random
import re

from datasets import Dataset
from itertools import chain, islice
from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import KeggReader
from .base_processor import BaseProcessor

random.seed(42)


class KeggProcessor(BaseProcessor):
    def __init__(
        self, 
        *argv, 
        **kwargs
    ):
        super().__init__(*argv)
        
        self.kegg_data_type = kwargs['kegg_data_type']
        self.data_reader = KeggReader(self.data_repo_path, self.task, self.kegg_data_type, self.relation_query_answers)
            
        self.all_pos_relations = set() # used for relation task.
        self.all_neg_relations = set() # used for relation task.
    
    
    def generate_datasets(
        self, 
        n_shots, 
        is_training
    ):
        task = self.task
        data = self.data_reader.train_data
        test_sample_size = self.test_sample_size
        
        self.task_prompt[task] = ""
        
        train_dataset = None
        val_dataset = None
        test_dataset = None
        
        ## TODO: complete this when 'relation' task is used for KEGG data.
        '''
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
        '''
        
        if task == "entity":
            if n_shots > 0:
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
                
            ## TODO: get the size of training/val data as an argument.
            if is_training:
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

                # val_dataset = self.format_dataset(val_dataset, task)
                # train_dataset = self.format_dataset(train_dataset, task)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        assert test_sample_size <= len(self.data_reader.test_data)
        
        self.test_dataset = dict(list(self.data_reader.test_data.items())[:test_sample_size])
        
        ## TODO: complete this when 'relation' task is used for KEGG data.
        '''
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
        '''
        
        # debug
        if self.get_rank() == 0:
            if len(self.task_prompt[task]) != 0:
                print(self.task_prompt[task])
                # input('enter..')
            

    def format_dataset(
        self, 
        dataset, 
        data_type
    ):
        task = self.task
        
        if task == "entity":
            if data_type in ['train', 'validation']:
                
                ## TODO: fix this.
                formatted_dataset = [
                                        {
                                            "text": f"{self.model_prompt['entity_q'](k)}{self.model_prompt['entity_a'](', '.join(list(chain.from_iterable(v))))}",
                                            "answer": ', '.join(list(chain.from_iterable(v))) # use all names.
                                        }
                                        for k, v in dataset.items()
                                    ]
            elif data_type == 'test':
                formatted_dataset = [
                                        {
                                            "entity": k,
                                            "text": f"{self.task_prompt[task]}{self.model_prompt['entity_q'](k)}",
                                            "answer": '__DELIMITER__'.join(list(chain.from_iterable(v))) # use all names.
                                            # "answer": '__DELIMITER__'.join([x[0] for x in v])) # only use the first name of the list. 
                                        }
                                        for k, v in dataset.items()
                                    ]

        formatted_dataset = Dataset.from_list(formatted_dataset)

        return formatted_dataset

    
    def update_results(
        self,
        decoded_entity, 
        decoded_pred, 
        decoded_gold
    ):
        task = self.task

        decoded_gold = [x.split('__DELIMITER__') for x in decoded_gold]
        decoded_gold = [[x.strip() for x in sublist] for sublist in decoded_gold]
        
        for item, pred, true in zip(decoded_entity, decoded_pred, decoded_gold):
            orig_pred = pred # debug
            orig_true = true # debug

            pred = self.clean_response(pred, true=true)
            
            ## TODO: get max_entity_list_len as an argument.
            ## TODO: currently, only check 10 entities. Increase this for further prediction evaluation.
            pred, true = self.sort_and_pad(pred, true, max_entity_list_len=10)

            # store the source item in the query to be used in the relation task. 04/12/2023
            src = [item] * len(pred)
            if len(self.results[task]['preprocessed']) != 0:
                self.results[task]['preprocessed'][0].extend(src)
                self.results[task]['preprocessed'][1].extend(pred)
                self.results[task]['preprocessed'][2].extend(true)
            else:
                self.results[task]['preprocessed'] = [src, pred, true]
            
            ## TODO: very naive way. find a better way.
            orig_pred_list = [x.strip() for x in orig_pred.split(',')]
            orig_true_list = orig_true
            
            common_values = list(set(orig_pred_list) & set(orig_true_list))
            orig_pred_list = common_values + list(set(orig_pred_list) - set(common_values))
            orig_true_list = common_values + list(set(orig_true_list) - set(common_values))
            orig_pred_txt = ', '.join(orig_pred_list)
            orig_true_txt = ', '.join(orig_true_list)

            self.results[task]['original'].append((item, orig_pred_txt, orig_true_txt))
            
            # debug
            if self.get_rank() == 0:
                print(">> Which genes are associated with", item + "?")
                print(">> orig_pred:", orig_pred)
                print(">> orig_true:", orig_true)
                print(">> pred:", pred)
                print(">> true:", true)

                # input('enter..')


    def infer(
        self,
        model, 
        generation_config,
        batch_size: int = 1,
    ):
        test_data = self.test_dataset
        task = self.task
        results = self.results
        
        # shots are from train data, so this isn't necessary. 08/28/2023
        '''
        ## TODO: make it cleaner later.
        if task == "entity":
            shots_keys = [x[0] for x in self.shot_samples]
        '''
        
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
            # ref: dict(islice(d.items(), 2)) # for dictionary
            batch_data = islice(test_data, start, stop)
            
            ## TODO: make it cleaner later.
            if task == "entity":
                batch_items = [] # debug
                batch_input_texts = []
                true_list = []

                for item in batch_data:
                    # shots are from train data, so this isn't necessary. 08/28/2023
                    '''
                    # skip samples used in few-shots.
                    if item in shots_keys:
                        continue
                    '''
                    
                    batch_items.append(item)
                    
                    entity_prompt_with_test_sample = self.task_prompt[task]
                    entity_prompt_with_test_sample += self.model_prompt['entity_q'](item)
                    
                    batch_input_texts.append(entity_prompt_with_test_sample)
                    true_list.append(test_data[item])

                pred_list = self.get_response(model, generation_config, batch_input_texts)
                
                for item, pred, true in zip(batch_items, pred_list, true_list):
                    # use all names.
                    true = list(chain.from_iterable(true))
                    #true = [x[0] for x in true] # only use the first name of the list. 
                    
                    orig_pred = pred # debug
                    orig_true = true # debug

                    pred = self.clean_response(pred, true=true)
                    pred, true = self.sort_and_pad(pred, true)
                    
                    '''
                    ## TODO: for now, check only precision. 03/14/2023
                    pred = [x for x in pred if x != 'NONE']
                    true = [x for x in true if x != 'NONE']
                    if len(pred) > len(true):
                        pred = pred[:len(true)]
                    elif len(true) > len(pred):
                        true = true[:len(pred)]
                    '''
                    
                    # store the source item in the query to be used in the relation task. 04/12/2023
                    src = [item] * len(pred)
                    if len(results[task]) != 0:
                        results[task][0].extend(src)
                        results[task][1].extend(pred)
                        results[task][2].extend(true)
                    else:
                        results[task] = [src, pred, true]

                    # debug

                    print(">> Which genes are associated with", item + "?")
                    print(">> orig_pred:", orig_pred)
                    print(">> orig_true:", orig_true)
                    print(">> pred:", pred)
                    print(">> true:", true)
                    # input('enter..')
                    
                    
            
            elif task in ["relation", "entity_relation"]:
                batch_items = [] # debug
                batch_input_texts = []
                true_list = []

                for item in batch_data:
                    batch_items.append(item)
                    
                    relation_prompt_with_test_sample = self.task_prompt[task]
                    relation_prompt_with_test_sample += self.model_prompt['relation_q'](item[0], item[1])
                    
                    batch_input_texts.append(relation_prompt_with_test_sample)
                    
                    # test for LLaMA - 05/21/2023
                    #batch_input_texts.append('Answer these questions:\n' + relation_prompt_with_test_sample)
                    
                    true_list.append(item[2].lower())

                pred_list = self.get_response(model, generation_config, batch_input_texts)
                
                
                # debug
                print(pred_list)
                
                
                for item, pred, true in zip(batch_items, pred_list, true_list):
                    orig_pred = pred # debug
                    
                    pred = self.clean_response(true=true)

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
