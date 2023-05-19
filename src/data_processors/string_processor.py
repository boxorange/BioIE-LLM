import os
import sys
import time
import re
import string
import random
import itertools

from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import *

from .base_processor import BaseProcessor

random.seed(42)


class StringProcessor(BaseProcessor):
    def __init__(
        self, 
        data_name: str, 
        data_repo_path: str,
        task: str,
        model_name: str,
        tokenizer,
    ):
        super().__init__(data_name, task, model_name, tokenizer)

        # pass task argument for entity_relation task. 04/12/2023
        self.data_reader = StringReader(data_repo_path, task)
        
        ## TODO: remove this later. this is to filter out samples with many answers (long list of proteins).
        self.max_entity_list_len = 30 # used for entity task.
        
        self.all_pos_prot_pairs = set() # used for relation task.
        self.all_neg_prot_pairs = set() # used for relation task.


    def create_prompt(
        self, 
        n_shots: int = 0
    ):
        self.model_prompt = self.get_model_prompt()
        task = self.task
        
        if hasattr(self.data_reader, "ent_types"):
            ent_types_included = {x: 0 for x in self.data_reader.ent_types}
            
            assert len(string.ascii_uppercase) >= len(self.data_reader.ent_types)
            
            ent_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.ent_types, string.ascii_uppercase)}
            self.ent_type_multiple_choices_str = ["(" + x + ") " + y for x, y in zip(string.ascii_uppercase, self.data_reader.ent_types)]
            self.ent_type_multiple_choices_str = " ".join(self.ent_type_multiple_choices_str)

        if hasattr(self.data_reader, "rel_types"):
            rel_types_included = {x: 0 for x in self.data_reader.rel_types}
            
            assert len(string.ascii_uppercase) >= len(self.data_reader.rel_types)
            
            rel_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.rel_types, string.ascii_uppercase)}
            #self.rel_type_multiple_choices_str = ["(" + x + ") " + y for x, y in zip(string.ascii_uppercase, self.data_reader.rel_types)]
            #self.rel_type_multiple_choices_str = " ".join(self.rel_type_multiple_choices_str)
            
            self.rel_type_multiple_choices_str = ", ".join(['"' + x + '"' for x in self.data_reader.rel_types])


        data = self.data_reader.train_data
        
        if n_shots == 0:
            if task == "relation":
                for prot_1, binding_prot_list in data.items():
                    for prot_2 in binding_prot_list:
                        self.all_pos_prot_pairs.add(tuple(sorted([prot_1, prot_2])))
                
                keys = list(data.keys())
                all_prot_pairs = list(itertools.combinations(keys, 2))
                #self.all_neg_prot_pairs = list(set(all_prot_pairs) - set(self.all_pos_prot_pairs))
                self.all_neg_prot_pairs = list(filter(lambda i: i not in self.all_pos_prot_pairs, all_prot_pairs)) # this is much faster than the above way.
                
                # to generate the same random samples, convert it to an ordered list. 
                self.all_pos_prot_pairs = sorted(self.all_pos_prot_pairs) 
                #self.all_neg_prot_pairs = sorted(self.all_neg_prot_pairs) # all_neg_prot_pairs is already sorted by combinations(). 

            return


        if task == "entity":
            keys = list(data.keys())
            
            # shuffle data to get random samples.
            random.shuffle(keys)
            
            # get few-shot samples.
            for key in keys:
                
                ## TODO: remove this later. this is to filter out samples with many answers (long list of proteins).
                if len(data[key]) > self.max_entity_list_len:
                    continue

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
                self.entity_prompt += self.model_prompt['entity_q'](sample[0])
                self.entity_prompt += self.model_prompt['entity_a'](", ".join(list(set(sample[1]))))
            
        elif task in ["relation", "entity_relation"]:
            # len(self.all_pos_prot_pairs) # 11,937,359 (w/o duplicates: 5,968,680)
            for prot_1, binding_prot_list in data.items():
                for prot_2 in binding_prot_list:
                    self.all_pos_prot_pairs.add(tuple(sorted([prot_1, prot_2])))

            st = time.time()
            keys = list(data.keys())
            all_prot_pairs = list(itertools.combinations(keys, 2))
            #self.all_neg_prot_pairs = list(set(all_prot_pairs) - set(self.all_pos_prot_pairs))
            self.all_neg_prot_pairs = list(filter(lambda i: i not in self.all_pos_prot_pairs, all_prot_pairs)) # this is much faster than the above way.
            et = time.time()
            elapsed_time = et - st
            td = timedelta(seconds=elapsed_time)
            print('>> Execution time in hh:mm:ss:', td)			
            
            st = time.time()
            # to generate the same random samples, convert it to an ordered list. 
            self.all_pos_prot_pairs = sorted(self.all_pos_prot_pairs)
            #self.all_neg_prot_pairs = sorted(self.all_neg_prot_pairs) # all_neg_prot_pairs is already sorted by combinations(). 
            et = time.time()
            elapsed_time = et - st
            td = timedelta(seconds=elapsed_time)
            print('>> Execution time in hh:mm:ss:', td)
            
            
            pos_prot_pairs = random.sample(self.all_pos_prot_pairs, n_shots)
            neg_prot_pairs = random.sample(self.all_neg_prot_pairs, n_shots)
            
            
            
                
            for pos_pair, neg_pair in zip(pos_prot_pairs, neg_prot_pairs):
                self.relation_prompt += self.model_prompt['relation_q'](neg_pair[0], neg_pair[1])
                #self.relation_prompt += self.model_prompt['relation_a']("False")
                self.relation_prompt += self.model_prompt['relation_a']("no")
                #self.relation_prompt += self.model_prompt['relation_a']("Unrelated")
                self.relation_prompt += self.model_prompt['relation_q'](pos_pair[0], pos_pair[1])
                #self.relation_prompt += self.model_prompt['relation_a']("True")
                self.relation_prompt += self.model_prompt['relation_a']("yes")
                #self.relation_prompt += self.model_prompt['relation_a']("Related")
                    
            
            
            # use this when the numbers of pos and neg are different. 
            '''
            for neg_pair in neg_prot_pairs:
                self.relation_prompt += self.relation_q(neg_pair[0], neg_pair[1])
                self.relation_prompt += self.relation_a("False")
                #self.relation_prompt += self.relation_a("Unrelated")
                
            for pos_pair in pos_prot_pairs:
                self.relation_prompt += self.relation_q(pos_pair[0], pos_pair[1])
                self.relation_prompt += self.relation_a("True")
                #self.relation_prompt += self.relation_a("Related")
            '''
            
            
            # test code for various orders of pos and neg samples.
            '''
            self.relation_prompt += self.relation_q(neg_prot_pairs[0][0], neg_prot_pairs[0][1])
            self.relation_prompt += self.relation_a("False")
            
            self.relation_prompt += self.relation_q(pos_prot_pairs[0][0], pos_prot_pairs[0][1])
            self.relation_prompt += self.relation_a("True")

            self.relation_prompt += self.relation_q(pos_prot_pairs[1][0], pos_prot_pairs[1][1])
            self.relation_prompt += self.relation_a("True")
            
            self.relation_prompt += self.relation_q(pos_prot_pairs[2][0], pos_prot_pairs[2][1])
            self.relation_prompt += self.relation_a("True")
            
            self.relation_prompt += self.relation_q(neg_prot_pairs[1][0], neg_prot_pairs[1][1])
            self.relation_prompt += self.relation_a("False")
            
            self.relation_prompt += self.relation_q(pos_prot_pairs[3][0], pos_prot_pairs[3][1])
            self.relation_prompt += self.relation_a("True")
                
            self.relation_prompt += self.relation_q(pos_prot_pairs[4][0], pos_prot_pairs[4][1])
            self.relation_prompt += self.relation_a("True")

            self.relation_prompt += self.relation_q(pos_prot_pairs[5][0], pos_prot_pairs[5][1])
            self.relation_prompt += self.relation_a("True")
            
            self.relation_prompt += self.relation_q(neg_prot_pairs[2][0], neg_prot_pairs[2][1])
            self.relation_prompt += self.relation_a("False")
            
            self.relation_prompt += self.relation_q(pos_prot_pairs[6][0], pos_prot_pairs[6][1])
            self.relation_prompt += self.relation_a("True")
                
            self.relation_prompt += self.relation_q(pos_prot_pairs[7][0], pos_prot_pairs[7][1])
            self.relation_prompt += self.relation_a("True")

            self.relation_prompt += self.relation_q(pos_prot_pairs[8][0], pos_prot_pairs[8][1])
            self.relation_prompt += self.relation_a("True")
            
            self.relation_prompt += self.relation_q(pos_prot_pairs[9][0], pos_prot_pairs[9][1])
            self.relation_prompt += self.relation_a("True")
            '''
            


            self.shot_samples.extend(pos_prot_pairs + neg_prot_pairs)				
            
            # exclude shot samples for inference.
            for i in pos_prot_pairs:
                self.all_pos_prot_pairs.remove(i)
            
            for i in neg_prot_pairs:
                self.all_neg_prot_pairs.remove(i)
            
            st = time.time()
            #self.all_pos_prot_pairs = set(self.all_pos_prot_pairs) - set(pos_prot_pairs)
            #self.all_pos_prot_pairs = list(self.all_pos_prot_pairs)
            #self.all_neg_prot_pairs = set(self.all_neg_prot_pairs) - set(neg_prot_pairs)
            #self.all_neg_prot_pairs = list(self.all_neg_prot_pairs)
            self.all_pos_prot_pairs = list(filter(lambda i: i not in pos_prot_pairs, self.all_pos_prot_pairs)) # this is much faster than the above way.
            self.all_neg_prot_pairs = list(filter(lambda i: i not in neg_prot_pairs, self.all_neg_prot_pairs)) # this is much faster than the above way.
            et = time.time()
            elapsed_time = et - st
            td = timedelta(seconds=elapsed_time)
            print('>> Execution time in hh:mm:ss:', td)
    
        # debug
        if len(self.entity_prompt) != 0:
            print(self.entity_prompt)
        
        if len(self.relation_prompt) != 0:
            print(self.relation_prompt)
        
        if len(self.entity_type_prompt) != 0:
            print(self.entity_type_prompt)
        
        if len(self.relation_type_prompt) != 0:
            print(self.relation_type_prompt)
        
    
    def infer(
        self,
        model, 
        batch_size: int = 1,
    ):
        """
        Generate model inference with a batch sized input texts.
        
        """
        test_data = self.data_reader.test_data
        task = self.task
        
        results = {'entity': [], 'relation': [], 'entity_type': [], 'relation_type': [], 'entity_relation': []}
        
        
        ## TODO: make it cleaner later.
        if task == "entity":
            shots_keys = [x[0] for x in self.shot_samples]

        ## TODO: to reduce the number of test samples for a preliminary test. 
        if task == "entity":
            test_sample_size = 10000
            
            # skip samples used in few-shots.
            test_data = {k: v for k, v in test_data.items() if k not in shots_keys}
            
            ## TODO: remove if-condition later. this is to filter out samples with many answers (long list of proteins).
            #test_data = {k: v for k, v in test_data.items() if len(v) <= self.max_entity_list_len}	
            #print(len(test_data)) # if len(v) <= 20 --> 163, if len(v) <= 30 --> 287, if len(v) <= 50 --> 559
            
            sample_keys = random.sample(sorted(test_data), test_sample_size)
            test_data = {k: v for k, v in test_data.items() if k in sample_keys}
            
        elif task == "relation":
            test_sample_size = 50000 # for each positive and negative
            
            pos_prot_pairs = random.sample(self.all_pos_prot_pairs, test_sample_size)
            neg_prot_pairs = random.sample(self.all_neg_prot_pairs, test_sample_size) # draw negative samples.

            pos_prot_pairs = [list(x) for x in pos_prot_pairs]
            for i in pos_prot_pairs:
                #i.append("True")
                i.append("yes")
            
            neg_prot_pairs = [list(x) for x in neg_prot_pairs]
            for i in neg_prot_pairs:
                #i.append("False")
                i.append("no")
            
            test_data = pos_prot_pairs + neg_prot_pairs
    
        
        # debug
        entity_prompt_max_length = 0
        flag = False
        error_list = []


        start = 0
        stop = start + batch_size

        while True:
            # ref: dict(itertools.islice(d.items(), 2)) # for dictionary
            batch_data = itertools.islice(test_data, start, stop)
            
            ## TODO: make it cleaner later.
            if task == "entity":
                batch_items = [] # debug
                batch_input_texts = []
                true_entities_list = []
                for item in batch_data:
                    batch_items.append(item)
                    true_entities_list.append(test_data[item])
                    
                    entity_prompt_with_test_sample = self.entity_prompt
                    entity_prompt_with_test_sample += self.model_prompt['entity_q'](item)
                    
                    batch_input_texts.append(entity_prompt_with_test_sample)
                
                pred_entities_list = self.generate(model, batch_input_texts)
                
                
                print(pred_entities_list)
                

                for item, pred_entities, true_entities in zip(batch_items, pred_entities_list, true_entities_list):
                    orig_pred_answer = pred_entities # debug
                    pred_entities = pred_entities.rsplit("\n\n", 1)[1]
                    pred_entities = pred_entities.replace("Answer: ", "", 1).replace("A: ", "", 1)
                    pred_entities = [x.strip() for x in pred_entities.split(", ")]
                    
                    ## TODO: check if this is really needed. 02/04/2023
                    pred_entities = list(set(pred_entities)) # remove duplicates.

                    pred_entities, true_entities = self.sort_and_pad(pred_entities, true_entities)
                    
                    ## TODO: for now, check only precision. 03/14/2023
                    pred_entities = [x for x in pred_entities if x != 'NONE']
                    true_entities = [x for x in true_entities if x != 'NONE']
                    if len(pred_entities) > len(true_entities):
                        pred_entities = pred_entities[:len(true_entities)]
                    elif len(true_entities) > len(pred_entities):
                        true_entities = true_entities[:len(pred_entities)]
                    
                    # store the source item in the query to be used in the relation task. 04/12/2023
                    src = [item] * len(pred_entities)
                    if len(results[task]) != 0:
                        results[task][0].extend(src)
                        results[task][1].extend(pred_entities)
                        results[task][2].extend(true_entities)
                    else:
                        results[task] = [src, pred_entities, true_entities]
                    
                    # debug
                    
                    #print(">> The number of processed samples:", str(num))
                    print(">> Which proteins are related to:", item)
                    print(">> src:", src)
                    print(">> pred_entities:", pred_entities)
                    print(">> true_entities:", true_entities)
                    print(">> orig_pred_answer:", orig_pred_answer)
                    input('enter..')
                    
            
            elif task in ["relation", "entity_relation"]:
                batch_items = [] # debug
                batch_input_texts = []
                true_answer_list = []
                for item in batch_data:
                    batch_items.append(item)
                    relation_prompt_with_test_sample = self.relation_prompt
                    relation_prompt_with_test_sample += self.model_prompt['relation_q'](item[0], item[1])
                    
                    batch_input_texts.append(relation_prompt_with_test_sample)
                    
                    true_answer_list.append(item[2])

                pred_entities_list = self.generate(model, batch_input_texts)

                for item, pred_answer, true_answer in zip(batch_items, pred_answer_list, true_answer_list):
                    pred_answer = pred_answer.rsplit("\n\n", 1)[1]
                    pred_answer = pred_answer.replace("Answer: ", "", 1).replace("A: ", "", 1)
                    pred_answer = re.sub(r'[^a-zA-Z]', '', pred_answer)
                    pred_answer = pred_answer.lower()
                    true_answer = true_answer.lower()
                    
                    if len(results[task]) != 0:
                        results[task][0].append(pred_answer)
                        results[task][1].append(true_answer)
                    else:
                        results[task] = [[pred_answer], [true_answer]]
                    
                    # debug
                    '''
                    print(">> item:", item)
                    print(">> pred_answer:", pred_answer)
                    print(">> true_answer:", true_answer)
                    input('enter..')
                    '''
        
            #print(f">> batch processed - len(test_data): {len(test_data) - len(self.shot_samples)}, start: {start}, stop: {stop}")
            print(f">> batch processed - len(test_data): {len(test_data)}, start: {start}, stop: {stop}")

            #if stop >= (len(test_data) - len(self.shot_samples)):
            if stop >= len(test_data):
                break

            start = stop
            stop = start + batch_size 
            
            # debug
            #print(">> entity_prompt_max_length:", entity_prompt_max_length)

        return results
