import os
import sys
import time
import string
import random
import re
import itertools
import pickle

from datasets import Dataset
from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import StringReader
from .base_processor import BaseProcessor

random.seed(42)


class StringProcessor(BaseProcessor):
    def __init__(self, *argv):
        super().__init__(*argv)

        # pass task argument for entity_relation task. 04/12/2023
        self.data_reader = StringReader(self.data_repo_path, self.task, self.relation_query_answers)
        
        ## TODO: remove this later. this is to filter out samples with many answers (long list of proteins).
        self.max_entity_list_len = 30 # used for entity task.
        
        self.all_pos_prot_pairs = set() # used for relation task.
        self.all_neg_prot_pairs = set() # used for relation task.

    
    def generate_datasets(
        self, 
        n_shots, 
        is_training
    ):
        task = self.task
        pos_ppi_data = self.data_reader.pos_ppi_data
        neg_ppi_data = self.data_reader.neg_ppi_data
        
        test_sample_size = self.test_sample_size
        
        self.task_prompt[task] = ""
        
        train_dataset = None
        val_dataset = None
        test_dataset = None

        st = time.time()

        if task == "entity":
            if n_shots > 0:
                keys = list(pos_ppi_data.keys())
                
                # shuffle data to get random samples.
                random.shuffle(keys)

                # get few-shot samples (used for inference).
                for key in keys:
                    ## TODO: remove this later. this is to filter out samples with many answers (long list of proteins).
                    if len(pos_ppi_data[key]) > self.max_entity_list_len:
                        continue

                    self.shot_samples.append((key, pos_ppi_data[key]))
                    
                    # skip samples used in few-shots.
                    del pos_ppi_data[key]
                    
                    if len(self.shot_samples) == n_shots:
                        break

                '''
                # get few-shot samples.
                sorted_data = sorted(pos_ppi_data.items(), key = lambda item : len(item[1]))
                for k, v in sorted_data:
                    self.shot_samples.append((k, v))
                    if len(self.shot_samples) == n_shots:
                        break
                '''

                for sample in self.shot_samples:
                    self.task_prompt[task] += self.model_prompt['entity_q'](sample[0])
                    self.task_prompt[task] += self.model_prompt['entity_a'](", ".join(sorted(list(set(sample[1]))))) # Reproducibility - the order of list affects the model inference.
            
            
            
            
            
            

            ## TODO: remove if-condition later. this is to filter out samples with many answers (long list of proteins).
            pos_ppi_data = {k: v for k, v in pos_ppi_data.items() if len(v) >= 10}	
            #test_data = {k: v for k, v in test_data.items() if len(v) <= self.max_entity_list_len}	
            # print(len(test_data)) # if len(v) <= 20 --> 163, if len(v) <= 30 --> 287, if len(v) <= 50 --> 559
            
            
            # debug
            # print(len(pos_ppi_data))
            # input('enter..')
            
            
            sample_keys = random.sample(sorted(pos_ppi_data), test_sample_size)
            test_dataset = {k: v for k, v in pos_ppi_data.items() if k in sample_keys}
            
            '''
            # task_prompt (few-shots) are only used for test datasets.
            test_dataset = [
                                {
                                    # "instruction": "Find binding proteins to a given protein.",
                                    "input": self.task_prompt[task] + self.model_prompt['entity_q'](k),
                                    # "output": self.model_prompt['entity_a'](", ".join(sorted(list(set(v)))))
                                    "output": sorted(list(set(v)))
                                }
                                for k, v in test_dataset.items()
                            ]
            
            test_dataset = Dataset.from_list(test_dataset)
            '''
            
            ## TODO: get the size of training/val data as an argument.
            if is_training:
                for key in sample_keys:
                    del pos_ppi_data[key]
                
                sample_keys = random.sample(sorted(pos_ppi_data), test_sample_size//2)
                val_dataset = {k: v[:10] for k, v in pos_ppi_data.items() if k in sample_keys}

                for key in sample_keys:
                    del pos_ppi_data[key]
                
                sample_keys = random.sample(sorted(pos_ppi_data), test_sample_size*5)
                train_dataset = {k: v[:10] for k, v in pos_ppi_data.items() if k in sample_keys}

                # val_dataset = self.format_dataset(val_dataset, task)
                # train_dataset = self.format_dataset(train_dataset, task)

        elif task in ["relation", "entity_relation"]:
            ## TODO: change the return statement!!
            """
            if n_shots == 0:
                if task == "relation":
                    for prot_1, binding_prot_list in pos_ppi_data.items():
                        for prot_2 in binding_prot_list:
                            # protein interactions can be directional, so don't change the orders in the pairs. 08/25/2023
                            #self.all_pos_prot_pairs.add(tuple(sorted([prot_1, prot_2])))
                            self.all_pos_prot_pairs.add(tuple([prot_1, prot_2]))
                    
                    # to generate the same random samples, convert it to an ordered list. 
                    self.all_pos_prot_pairs = sorted(self.all_pos_prot_pairs)
                    
                    # get negative samples from Negatome.
                    self.all_neg_prot_pairs = neg_ppi_data
                    
                    # get negative samples from STRING DB not found interactions.
                    '''				
                    keys = list(pos_ppi_data.keys())
                    all_prot_pairs = list(itertools.combinations(keys, 2))
                    #self.all_neg_prot_pairs = list(set(all_prot_pairs) - set(self.all_pos_prot_pairs))
                    self.all_neg_prot_pairs = list(filter(lambda i: i not in self.all_pos_prot_pairs, all_prot_pairs)) # this is much faster than the above way.
                    #self.all_neg_prot_pairs = sorted(self.all_neg_prot_pairs) # all_neg_prot_pairs is already sorted by combinations(). 
                    '''
                return
            """
        
            # len(self.all_pos_prot_pairs) # 11,937,359 (w/o duplicates: 5,968,680)
            for prot_1, binding_prot_list in pos_ppi_data.items():
                for prot_2 in binding_prot_list:
                    # protein interactions can be directional, so don't change the orders in the pairs. 08/25/2023
                    #self.all_pos_prot_pairs.add(tuple(sorted([prot_1, prot_2])))
                    self.all_pos_prot_pairs.add(tuple([prot_1, prot_2]))
            
            # debug
            if self.get_rank() == 0:
                st = time.time()
            
            # to generate the same random samples, convert it to an ordered list. 
            self.all_pos_prot_pairs = sorted(self.all_pos_prot_pairs)
            
            # debug
            if self.get_rank() == 0:
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                print('>> (sorting all_pos_prot_pairs) Execution time in hh:mm:ss:', td)
                
            # get negative samples from unconnected interactions in STRING DB.
            '''
            st = time.time()
            keys = list(pos_ppi_data.keys())
            all_prot_pairs = list(itertools.combinations(keys, 2))
            #self.all_neg_prot_pairs = list(set(all_prot_pairs) - set(self.all_pos_prot_pairs))
            self.all_neg_prot_pairs = list(filter(lambda i: i not in self.all_pos_prot_pairs, all_prot_pairs)) # this is much faster than the above way.
            #self.all_neg_prot_pairs = sorted(self.all_neg_prot_pairs) # all_neg_prot_pairs is already sorted by combinations(). 
            et = time.time()
            elapsed_time = et - st
            td = timedelta(seconds=elapsed_time)
            print('>> Execution time in hh:mm:ss:', td)
            '''
            
            # get negative samples from Negatome.
            # debug			
            # self.all_neg_prot_pairs = neg_ppi_data
            # self.all_neg_prot_pairs = list(filter(lambda i: i not in self.all_pos_prot_pairs, self.all_neg_prot_pairs))
            # print(len(self.all_neg_prot_pairs))
            self.all_neg_prot_pairs = neg_ppi_data

            pos_prot_pairs = random.sample(self.all_pos_prot_pairs, n_shots)
            neg_prot_pairs = random.sample(self.all_neg_prot_pairs, n_shots)

            for pos_pair, neg_pair in zip(pos_prot_pairs, neg_prot_pairs):
                self.task_prompt[task] += self.model_prompt['relation_q'](neg_pair[0], neg_pair[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[1])
                self.task_prompt[task] += self.model_prompt['relation_q'](pos_pair[0], pos_pair[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[0])
                
            # use this when the numbers of pos and neg are different. 
            '''
            for neg_pair in neg_prot_pairs:
                self.task_prompt[task] += self.model_prompt['relation_q'](neg_pair[0], neg_pair[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[1])
                
            for pos_pair in pos_prot_pairs:
                self.task_prompt[task] += self.model_prompt['relation_q'](pos_pair[0], pos_pair[1])
                self.task_prompt[task] += self.model_prompt['relation_a'](self.relation_query_answers[0])
            '''
            
            self.shot_samples.extend(pos_prot_pairs + neg_prot_pairs)				
            
            
            # since it takes too long to generate datasets, save/load them in files.
            path = os.path.expanduser(self.data_repo_path)
            data_path = os.path.join(path, "STRING/converted")
            n_shots_test_file_path = os.path.join(data_path, str(n_shots) + "_shots_test_relation.pickle")

            if os.path.exists(n_shots_test_file_path):
                with open(n_shots_test_file_path, 'rb') as fin:
                    self.test_dataset = pickle.load(fin)
                
                return
                
                
            # debug
            if self.get_rank() == 0:
                st = time.time()

            self.all_pos_prot_pairs = list(filter(lambda i: i not in pos_prot_pairs, self.all_pos_prot_pairs)) # this is much faster than the above way.			
            self.all_neg_prot_pairs = list(filter(lambda i: i not in neg_prot_pairs, self.all_neg_prot_pairs)) # this is much faster than the above way.			
            
            # exclude shot samples from datasets.
            # for i in pos_prot_pairs:
                # self.all_pos_prot_pairs.remove(i)
            
            # for i in neg_prot_pairs:
                # self.all_neg_prot_pairs.remove(i)

            # self.all_pos_prot_pairs = list(self.all_pos_prot_pairs)
            # self.all_neg_prot_pairs = list(self.all_neg_prot_pairs)
            
            # debug
            if self.get_rank() == 0:
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                
                print('>> len(self.all_pos_prot_pairs):', len(self.all_pos_prot_pairs))
                print('>> len(self.all_neg_prot_pairs):', len(self.all_neg_prot_pairs))
                print('>> (exclude shot samples from datasets) Execution time in hh:mm:ss:', td)

            def sample_pos_neg_data(all_pos, all_neg, size):
                pos = random.sample(all_pos, size)
                neg = random.sample(all_neg, size)
                
                if is_training:
                    # debug
                    if self.get_rank() == 0:
                        st = time.time()
                    
                    for i in pos:
                        all_pos.remove(i)
                    
                    for i in neg:
                        all_neg.remove(i)
                    
                    # debug
                    if self.get_rank() == 0:
                        et = time.time()
                        elapsed_time = et - st
                        td = timedelta(seconds=elapsed_time)
                        
                        print('>> len(self.all_pos_prot_pairs):', len(self.all_pos_prot_pairs))
                        print('>> len(self.all_neg_prot_pairs):', len(self.all_neg_prot_pairs))
                        print('>> (removing sampled data in all data) Execution time in hh:mm:ss:', td)
                        
                    input('enter..')
                
                pos = [list(x) for x in pos]
                for i in pos:
                    i.append(self.relation_query_answers[0])
                
                neg = [list(x) for x in neg]
                for i in neg:
                    i.append(self.relation_query_answers[1])
                
                return pos + neg
                
            # test_sample_size for each positive and negative E.g., if test_sample_size = 500, 500 pos + 500 neg
            test_dataset = sample_pos_neg_data(self.all_pos_prot_pairs, self.all_neg_prot_pairs, test_sample_size)
            
            # since it takes too long to generate datasets, save/load them in files.
            with open(n_shots_test_file_path, 'wb') as fout:
                    pickle.dump(test_dataset, fout)

            ## TODO: get the size of training/val data as an argument.
            if is_training:
                '''
                st = time.time()
                
                for i in pos:
                    all_pos.remove(i)
                
                for i in neg:
                    all_neg.remove(i)
                
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                print('>> (removing items in all data) Execution time in hh:mm:ss:', td)
                '''
                val_dataset = sample_pos_neg_data(self.all_pos_prot_pairs, self.all_neg_prot_pairs, test_sample_size//2)
                train_dataset = sample_pos_neg_data(self.all_pos_prot_pairs, self.all_neg_prot_pairs, test_sample_size*5)
                
                # val_dataset = self.format_dataset(val_dataset, task)
                # train_dataset = self.format_dataset(train_dataset, task)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # debug
        if self.get_rank() == 0:
            if len(self.task_prompt[task]) != 0:
                print(self.task_prompt[task])
    
    
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
                                            # "instruction": "Find binding proteins to a given protein.",
                                            # "input": self.task_prompt[task] + self.model_prompt['entity_q'](k),
                                            # "input": self.model_prompt['entity_q'](k),
                                            # "output": self.model_prompt['entity_a'](", ".join(sorted(list(set(v))))),
                                            # "text": f"{self.model_prompt['entity_q'](k)}{self.model_prompt['entity_a'](', '.join(v))}"
                                            "text": f"{self.model_prompt['entity_q'](k)}{self.model_prompt['entity_a'](', '.join(sorted(list(set(v)))))}"
                                        }
                                        for k, v in dataset.items()
                                    ]
            elif data_type == 'test':
                formatted_dataset = [
                                        {
                                            "entity": k,
                                            "text": f"{self.task_prompt[task]}{self.model_prompt['entity_q'](k)}",
                                            "answer": '__DELIMITER__'.join(sorted(list(set(v)))) 
                                        }
                                        for k, v in dataset.items()
                                    ]
                                    
        elif task in ["relation", "entity_relation"]:
            
            ## TODO: format train/val datasets.
            
            formatted_dataset = [
                                    {
                                        "entity": '__DELIMITER__'.join([i[0], i[1]]),
                                        "text": f"{self.task_prompt[task]}{self.model_prompt['relation_q'](i[0], i[1])}",
                                        # "answer": f"{self.model_prompt['relation_a'](i[2])}",
                                        "answer": i[2] # no performance diff compared to the one above.
                                    }
                                    for i in dataset
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
        
        if task == "entity":
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
                    print(">> Which proteins interact with", item + "?")
                    print(">> orig_pred:", orig_pred)
                    print(">> orig_true:", orig_true)
                    print(">> pred:", pred)
                    print(">> true:", true)
                    # input('enter..')
        
        elif task in ["relation", "entity_relation"]:
            decoded_entity = [x.split('__DELIMITER__') for x in decoded_entity]
            decoded_entity = [[x.strip() for x in sublist] for sublist in decoded_entity]
            
            for item, pred, true in zip(decoded_entity, decoded_pred, decoded_gold):
                orig_pred = pred # debug
                orig_true = true # debug
                
                pred = pred.strip()
                true = true.strip()

                pred = self.clean_response(pred, true=true)

                if len(self.results[task]['preprocessed']) != 0:
                    self.results[task]['preprocessed'][0].append([item[0], item[1]])
                    self.results[task]['preprocessed'][1].append(pred)
                    self.results[task]['preprocessed'][2].append(true)
                else:
                    self.results[task]['preprocessed'] = [[[item[0], item[1]]], [pred], [true]]

                # debug
                if self.get_rank() == 0:
                    # print(">> QUESTION: Do {x} and {y} interact with each other?".format(x=item[0], y=item[1]))
                    print(">> QUESTION: Does {x} interact with {y}?".format(x=item[0], y=item[1]))
                    print(">> orig_pred:", orig_pred)
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
        
        start = 0
        stop = start + batch_size

        while True:
            # ref: dict(itertools.islice(d.items(), 2)) # for dictionary
            batch_data = itertools.islice(test_data, start, stop)

            ## TODO: make it cleaner later.
            if task == "entity":
                batch_items = [] # debug
                batch_input_texts = []
                true_list = []
                
                for item in batch_data:
                    batch_items.append(item)
                    
                    entity_prompt_with_test_sample = self.task_prompt[task]
                    entity_prompt_with_test_sample += self.model_prompt['entity_q'](item)
                    #entity_prompt_with_test_sample += self.model_prompt['entity_q']("ARFGAP3") # debug
                                        
                    batch_input_texts.append(entity_prompt_with_test_sample)
                    true_list.append(test_data[item])

                pred_list = self.get_response(model, generation_config, batch_input_texts)

                for item, pred, true in zip(batch_items, pred_list, true_list):
                    orig_pred = pred # debug
                    orig_true = true # debug
                    
                    pred = self.clean_response(pred, true)
                    
                    ## TODO: get max_entity_list_len as an argument.
                    pred, true = self.sort_and_pad(pred, true, max_entity_list_len=10)
                    
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
                    
                    #print(">> The number of processed samples:", str(num))
                    print(">> Which proteins interact with", item + "?")
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

                for item, pred, true in zip(batch_items, pred_list, true_list):
                    orig_pred = pred # debug
                    
                    pred = self.clean_response(pred)
                    
                    if len(results[task]) != 0:
                        results[task][0].append(pred)
                        results[task][1].append(true)
                    else:
                        results[task] = [[pred], [true]]
                    
                    # debug
                    #print(">> QUESTION: Do {x} and {y} interact with each other?".format(x=item[0], y=item[1]))
                    print(">> QUESTION: Does {x} interact with {y}?".format(x=item[0], y=item[1]))
                    print(">> orig_pred:", orig_pred)
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
