import sys
from abc import abstractmethod

import torch, gc

# setting path
sys.path.append('../prompt_formatters')
from prompt_formatters import *


class BaseProcessor:
    def __init__(
        self, 
        data_name: str, 
        task: str,
        model_name: str,
        tokenizer, 
    ):
        self.data_name = data_name
        self.task = task
        self.model_name = model_name
        self.tokenizer = tokenizer
        
        self.entity_prompt, self.relation_prompt, self.entity_type_prompt, self.relation_type_prompt = "", "", "", ""
        self.shot_samples = []
    
    
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
    
    
    def generate(self, model, batch_input_texts):
        if self.model_name == 'Galactica':
            pred_entities_list = model.generate(batch_input_texts, task=self.task)
        else:
            inputs = self.tokenizer(batch_input_texts, return_tensors="pt", padding=True).to("cuda")
            
            if self.task in ['relation', 'relation_type']: # relation and relation_type tasks only return a selection of multiple choices (e.g., True or False), so a longer length is not necessary. 
                max_length = len(inputs.input_ids[0]) + 10
            else:
                max_length = len(inputs.input_ids[0]) + 100

            generate_ids = model.generate(inputs.input_ids, max_length=max_length)
            pred_entities_list = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
        gc.collect()
        torch.cuda.empty_cache()
        
        return pred_entities_list


    def get_model_prompt(self):
        if self.model_name == 'Galactica':
            if self.data_name == 'scierc':
                return GalacticaPromptFormatter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return GalacticaPromptFormatter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return GalacticaPromptFormatter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return GalacticaPromptFormatter.get_indra_prompt(self)
            
        elif self.model_name == 'LLaMA':
            if self.data_name == 'scierc':
                return LlamaPromptFormatter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return LlamaPromptFormatter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return LlamaPromptFormatter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return LlamaPromptFormatter.get_indra_prompt(self)
        
        elif self.model_name == 'RST':
            if self.data_name == 'scierc':
                return RstPromptFormatter.get_scierc_prompt(self)
            elif self.data_name == 'string':
                return RstPromptFormatter.get_string_prompt(self)
            elif self.data_name == 'kegg':
                return RstPromptFormatter.get_kegg_prompt(self)
            elif self.data_name == 'indra':
                return RstPromptFormatter.get_indra_prompt(self)


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
                    