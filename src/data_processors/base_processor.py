import sys
import re
from abc import abstractmethod

import torch, gc

# setting path
sys.path.append('../prompters')
from prompters import *


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class BaseProcessor:
    def __init__(
        self, 
        data_name: str, 
        task: str,
        test_sample_size: int,
        model_name: str,
        tokenizer, 
    ):
        self.data_name = data_name
        self.task = task
        self.test_sample_size = test_sample_size
        self.model_name = model_name
        self.tokenizer = tokenizer
        
        self.task_prompt = {}
        self.shot_samples = []
        
        self.relation_query_answers = ['yes', 'no']
        #self.relation_query_answers = ["True", "False"]
        #self.relation_query_answers = ["Related", "Unrelated"]
        #self.relation_query_answers = ["related", "unrelated"]
        #self.relation_query_answers = ["linked", "unlinked"]
        
        self.results = {task: []}
    
    
    @abstractmethod
    def create_prompt(
        self,
        task: str,
        n_shots: int,
    ):
        raise NotImplementedError
    
    
    @abstractmethod
    def infer(
        self,
        model, 
        task: str, 
        batch_size: int,
    ):
        raise NotImplementedError
    
    
    def get_response(self, model, batch_input_texts):
        if self.model_name == 'Galactica':
            response = model.generate(batch_input_texts, task=self.task)
        else:
            inputs = self.tokenizer(batch_input_texts, return_tensors="pt", padding=True).to(device)
            
            if self.task in ['relation', 'relation_type']: # relation and relation_type tasks only return a selection of multiple choices (e.g., True or False), so a longer length is not necessary. 
                max_length = len(inputs.input_ids[0]) + 5
            else:
                max_length = len(inputs.input_ids[0]) + 100

            generate_ids = model.generate(input_ids=inputs.input_ids, max_length=max_length)
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
        gc.collect()
        torch.cuda.empty_cache()
        
        return response
    
    
    def clean_response(self, response, prompt):
        """
        Remove a prompt and unnecessary texts in model's generated texts.
        
        """
        if self.model_name == 'RST':
            if self.task == 'entity':
                cleaned_response = [response]
            else:
                ## TODO: test and fix this.
                cleaned_response = response
        else:
            response = response.strip()
            prompt = prompt.strip()
            
            if self.model_name == 'BioGPT': 
                prompt = prompt.replace("\n\n", " ")
                
            response = response.replace(prompt, "")
            
            if self.task == 'entity':
                ## TODO: handle the cases below.
                # 1. Alpaca: ['4) Veli-like 2 (VEL2)', '1) E-cadherin (CDH1)', '7) Veli-like 5 (VEL5)', '6) Veli-like 4 (VEL4)', '2) Plectin (PLEC)', '8) Veli-', '5) Veli-like 3 (VEL3)', '3) Veli-like 1 (VEL1)'
                # 2. remove a query in the answer. LLaMA included a query in its answer sometimes. - 05/21/2023
                #     e.g., pred_entities = pred_entities.replace(self.model_prompt['entity_q'](item), "")
                        
                cleaned_response = response.replace("Answer: ", "", 1).replace("A: ", "", 1)
                cleaned_response = [x.strip() for x in cleaned_response.split(", ")]
                    
                ## TODO: check if this is really needed. 02/04/2023
                cleaned_response = list(set(cleaned_response)) # remove duplicates.

            elif self.task in ['relation', 'entity_relation', 'relation_type']:
                if self.task == 'relation_type':
                    choices = self.data_reader.rel_types
                else:
                    choices = self.relation_query_answers
                
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
            
        elif self.model_name == 'LLaMA':
            if self.data_name == 'scierc':
                return LlamaPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return LlamaPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return LlamaPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return LlamaPrompter.get_indra_prompt(self)
        
        elif self.model_name == 'Alpaca':
            if self.data_name == 'scierc':
                return AlpacaPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return AlpacaPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return AlpacaPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return AlpacaPrompter.get_indra_prompt(self)
        
        elif self.model_name == 'BioGPT':
            if self.data_name == 'scierc':
                return BioGPTPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return BioGPTPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return BioGPTPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return BioGPTPrompter.get_indra_prompt(self)
                
        elif self.model_name == 'RST':
            if self.data_name == 'scierc':
                return RstPrompter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return RstPrompter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return RstPrompter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return RstPrompter.get_indra_prompt(self)


    def sort_and_pad(self, list_1, list_2):
        """
        ref: https://stackoverflow.com/questions/73428068/how-to-append-item-to-match-the-length-of-two-list-in-python
        
        """
        common_values = list(set(list_1) & set(list_2))
        new_list_1 = common_values + list(set(list_1) - set(common_values))
        new_list_2 = common_values + list(set(list_2) - set(common_values))
        
        new_list_1_len = len(new_list_1)
        new_list_2_len = len(new_list_2)
        diff = abs(new_list_1_len - new_list_2_len)
        
        # padding number of elements to the end of the list
        if new_list_1_len < new_list_2_len:
            new_list_1 += ['NONE'] * diff
        elif new_list_1_len > new_list_2_len:
            new_list_2 += ['NONE'] * diff

        return new_list_1, new_list_2
                    