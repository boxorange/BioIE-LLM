import sys
import time
import string
import random
import re
import itertools

from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import IndraReader

from .base_processor import BaseProcessor

random.seed(42)


class IndraProcessor(BaseProcessor):
    def __init__(
        self, 
        data_name: str, 
        data_repo_path: str,
        task: str,
        test_sample_size: int, 
        model_name: str,
        tokenizer,
        num_of_indra_classes: int,
    ):
        super().__init__(data_name, task, test_sample_size, model_name, tokenizer)
        
        self.data_reader = IndraReader(data_repo_path, num_of_indra_classes)
        

    def create_prompt(
        self, 
        n_shots: int = 0
    ):
        self.model_prompt = self.get_model_prompt()
        task = self.task
        data = self.data_reader.train_data
        
        self.task_prompt[task] = ""
        
        rel_types_included = {x: 0 for x in self.data_reader.rel_types}
            
        assert len(string.ascii_uppercase) >= len(self.data_reader.rel_types)
        
        rel_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.rel_types, string.ascii_uppercase)}
        #self.rel_type_multiple_choices_str = ["(" + x + ") " + y for x, y in zip(string.ascii_uppercase, self.data_reader.rel_types)]
        #self.rel_type_multiple_choices_str = " ".join(self.rel_type_multiple_choices_str)
        
        self.rel_type_multiple_choices_str = ", ".join(['"' + x + '"' for x in self.data_reader.rel_types])
        
        # shuffle data to get random samples.
        random.shuffle(data)
        
        # get few-shot samples.
        for item in data:
            rel_type = item['rel_type']

            if all(value == n_shots for value in rel_types_included.values()):
                break
            
            if rel_type in rel_types_included:
                if rel_types_included[rel_type] < n_shots:
                    self.shot_samples.append(item)
                    rel_types_included[rel_type] += 1
        
        random.shuffle(self.shot_samples)
        
        for sample in self.shot_samples:
            text = sample['text']
            entity_1 = sample['entity_1'][0]
            entity_2 = sample['entity_2'][0]
            rel_type = sample['rel_type']

            self.task_prompt[task] += text
            self.task_prompt[task] += self.model_prompt['relation_type_q'](entity_1, entity_2, self.rel_type_multiple_choices_str)
            self.task_prompt[task] += self.model_prompt['relation_type_a'](rel_type)
            
            # testing for question, text (context), answer format used in PubMedQA - 05/31/2023
            # but, the results were worse than the original format.
            #self.task_prompt[task] += self.model_prompt['relation_type_q'](entity_1, entity_2, text, self.rel_type_multiple_choices_str)
            #self.task_prompt[task] += self.model_prompt['relation_type_a'](rel_type)
            
            
        # debug
        if len(self.task_prompt[task]) != 0:
            print(self.task_prompt[task])

    
    def infer(
        self,
        model, 
        batch_size: int = 1,
    ):
        test_data = self.data_reader.test_data
        task = self.task
        results = self.results

        shots_keys = [x['id'] for x in self.shot_samples]
        
        if task == "relation_type":
            # use all data.
            test_data = self.data_reader.train_data + self.data_reader.dev_data + self.data_reader.test_data

            test_sample_size = self.test_sample_size # for each class
            
            # debug
            #test_stat = {}
            
            filtered_test_data = {}
            
            for sample in test_data:
                
                # debug
                #if sample['rel_type'] in test_stat:
                #	test_stat[sample['rel_type']] += 1
                #else:
                #	test_stat[sample['rel_type']] = 1
                
                if sample['id'] in shots_keys: # skip samples used in few-shots.
                    continue
                
                '''
                This is for finding examples for each class. 03/15/2023
                classes: "Activation", "Inhibition", "Phosphorylation", "Dephosphorylation", "Ubiquitination", "Deubiquitination"
                '''
                #if sample['rel_type'] in ['Deubiquitination']:
                if sample['rel_type'] in self.data_reader.rel_types:
                    if sample['rel_type'] not in filtered_test_data:
                        filtered_test_data[sample['rel_type']] = [sample]
                    else:
                        filtered_test_data[sample['rel_type']].append(sample)
            
            # debug
            #test_stat = sorted(test_stat.items(), key=lambda x:x[1], reverse=True)
            #test_stat = dict(test_stat)
            #for k, v in test_stat.items():
            #	print(k, v)
            
            test_data = []
            for k, v in filtered_test_data.items():
                test_data.extend(random.sample(v, test_sample_size))
            
            #test_data = random.sample(filtered_test_data, test_sample_size)
            
            ## TODO: make it cleaner later. 02/24/2023
            '''
            filtered_test_data_activation = [x for x in filtered_test_data if x['rel_type'] == 'Activation']
            filtered_test_data_inhibition = [x for x in filtered_test_data if x['rel_type'] == 'Inhibition']
            test_data = random.sample(filtered_test_data_activation, int(test_sample_size/2)) + random.sample(filtered_test_data_inhibition, int(test_sample_size/2))
            
            
            filtered_test_data_activation = [x for x in filtered_test_data if x['rel_type'] == 'Activation']
            filtered_test_data_inhibition = [x for x in filtered_test_data if x['rel_type'] == 'Inhibition']
            filtered_test_data_phosphorylation = [x for x in filtered_test_data if x['rel_type'] == 'Phosphorylation']
            test_data = random.sample(filtered_test_data_activation, int(test_sample_size/3)) + random.sample(filtered_test_data_inhibition, int(test_sample_size/3)) + random.sample(filtered_test_data_phosphorylation, int(test_sample_size/3))
            '''
            
            #print(len(filtered_test_data))
            #print(len([x for x in filtered_test_data if x['rel_type'] == 'Activation']))
            #print(len([x for x in filtered_test_data if x['rel_type'] == 'Inhibition']))
            #print(len([x for x in test_data if x['rel_type'] == 'Activation']))
            #print(len([x for x in test_data if x['rel_type'] == 'Inhibition']))
            #input('enter..')

        start = 0
        stop = start + batch_size

        while True:
            # ref: dict(itertools.islice(d.items(), 2)) # for dictionary
            batch_data = itertools.islice(test_data, start, stop)
            
            ## TODO: make it cleaner later.
            if task == "relation_type":
                batch_items = [] # debug
                batch_input_texts = []
                prompt_list = []
                true_list = []
                
                for item in batch_data:
                    batch_items.append(item)
                    text = item['text']
                    entity_1 = item['entity_1'][0]
                    entity_2 = item['entity_2'][0]
                    rel_type = item['rel_type']
                    
                    true_list.append(rel_type.lower())
                    
                    '''
                    # testing for question, text (context), answer format used in PubMedQA - 05/31/2023
                    # but, the results were worse than the original format.
                    if self.model_name == 'RST':
                        if len(self.task_prompt[task]) > 0:
                            relation_type_prompt_with_test_sample = 'TEXT: ' + self.task_prompt[task]
                        else:
                            relation_type_prompt_with_test_sample = self.task_prompt[task]
                        
                        #relation_type_prompt_with_test_sample += 'QUERY: ' + self.model_prompt['relation_type_q'](entity_1, entity_2, self.rel_type_multiple_choices_str).replace('Question: ', '')
                        relation_type_prompt_with_test_sample += 'QUERY: ' + self.model_prompt['relation_type_q'](entity_1, entity_2, text, self.rel_type_multiple_choices_str).replace('Question: ', '')
                    
                    elif self.model_name == 'BioMedLM':
                        relation_type_prompt_with_test_sample = self.task_prompt[task]
                        relation_type_prompt_with_test_sample += self.model_prompt['relation_type_q'](entity_1, entity_2, text, self.rel_type_multiple_choices_str).replace('Question: ', '')
                    
                    else:
                    '''
                    relation_type_prompt_with_test_sample = self.task_prompt[task]
                    relation_type_prompt_with_test_sample += text
                    relation_type_prompt_with_test_sample += self.model_prompt['relation_type_q'](entity_1, entity_2, self.rel_type_multiple_choices_str)

                    prompt_list.append(relation_type_prompt_with_test_sample)
                    batch_input_texts.append(relation_type_prompt_with_test_sample)

                pred_list = self.get_response(model, batch_input_texts)
                
                
                print(pred_list)
                #input('enter..')

                for item, pred, prompt, true in zip(batch_items, pred_list, prompt_list, true_list):
                    orig_pred = pred # debug
                    
                    pred = self.clean_response(pred, prompt)

                    if len(results[task]) != 0:
                        results[task][0].append(pred)
                        results[task][1].append(true)
                    else:
                        results[task] = [[pred], [true]]

                    # debug
                    
                    #if pred != 'Phosphorylation':
                    #print(">> The number of processed samples:", str(num))
                    print(">> text:", item['text'])
                    print(">> entity_1:", item['entity_1'], ", entity_2:", item['entity_2'])
                    print(">> pred:", pred, ", true:", true)
                    print(">> orig_pred:", orig_pred)
                    #print(">> relation_type_prompt:", self.relation_type_prompt)
                    #if pred == 'Deubiquitination':
                    #	input('enter..')
                    #input('enter..')
                    

            #print(f">> batch processed - len(test_data): {len(test_data) - len(self.shot_samples)}, start: {start}, stop: {stop}")
            print(f">> batch processed - len(test_data): {len(test_data)}, start: {start}, stop: {stop}")

            #if stop >= (len(test_data) - len(self.shot_samples)):
            if stop >= len(test_data):
                break

            start = stop
            stop = start + batch_size 

        return results
