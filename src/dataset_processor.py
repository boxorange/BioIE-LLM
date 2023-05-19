import os
import time
from datetime import timedelta
import re
import string
import random
import itertools

import torch, gc

from data import *
from prompt_formatter import *


random.seed(42)


class DatasetProcessor:
    """
    Prompt 에서 "Q: ", "A: " 유무가 결과에 차이를 냄. github 에 예시대로 빼고 할 것. 2023-01-06 
    However, an error occurs when you leave "Q: ", "A: " out of input text.

    ref:
        - reStructured_Pre-training: https://www.researchgate.net/publication/361480300_reStructured_Pre-training
        - https://arxiv.org/pdf/2212.13138.pdf 

    TODO: check if adding quotation marks for entities improves the performance.

    Warning!! 
        - if there is a space before/after "\n\n", , it causes "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED".
          e.g., "\n\nQ: Which of the following is the entity type of \"{x}\"? {y} \n\n" -> causes an error!!

    E.g.,
    input_text = ("XREF_BIBR, XREF_BIBR BCL2 inhibits mitochondrial release of cytochrome c and subsequently blocks the caspase cascade, thereby inhibiting apoptosis."
          "\n\nQ: What gene or protein entities are mentioned in this text?\n\n"
          "A: BCL2, cytochrome</s>\n\n"
          
          "XREF_BIBR, XREF_BIBR BCL2 inhibits mitochondrial release of cytochrome c and subsequently blocks the caspase cascade, thereby inhibiting apoptosis."
          "\n\nQ: What is the relationship between BCL2 and cytochrome in this text?\n\n"
          "A: inhibition</s>\n\n"

    """
    
    def __init__(self, data_name, data_repo_path, tokenizer, task=None):
        self.data_name = data_name
        self.tokenizer = tokenizer

        self.entity_prompt, self.relation_prompt, self.entity_type_prompt, self.relation_type_prompt = "", "", "", ""
        self.shot_samples = []
        
        if self.data_name == "scierc":
            self.data_reader = SciercReader(data_repo_path)

            ## TODO: make it cleaner later.
            self.entity_q = SciercPromptFormat.entity_q
            self.entity_a = SciercPromptFormat.entity_a
            self.entity_type_q = SciercPromptFormat.entity_type_q
            self.entity_type_a = SciercPromptFormat.entity_type_a
            self.relation_type_q = SciercPromptFormat.relation_type_q
            self.relation_type_a = SciercPromptFormat.relation_type_a
            
        elif self.data_name == "string":
            # pass task argument for entity_relation task. 04/12/2023
            self.data_reader = StringReader(data_repo_path, task)
            
            ## TODO: remove this later. this is to filter out samples with many answers (long list of proteins).
            self.max_entity_list_len = 30 # used for entity task.
            
            self.all_pos_prot_pairs = set() # used for relation task.
            self.all_neg_prot_pairs = set() # used for relation task.
            
            self.entity_q = StringPromptFormat.entity_q
            self.entity_a = StringPromptFormat.entity_a
            self.relation_q = StringPromptFormat.relation_q
            self.relation_a = StringPromptFormat.relation_a
            
        elif self.data_name == "kegg":
            self.data_reader = KeggReader(data_repo_path, task)
            
            self.all_pos_relations = set() # used for relation task.
            self.all_neg_relations = set() # used for relation task.
            
            self.entity_q = KeggPromptFormat.entity_q
            self.entity_a = KeggPromptFormat.entity_a
            self.relation_q = KeggPromptFormat.relation_q
            self.relation_a = KeggPromptFormat.relation_a
            
        elif self.data_name == "indra":
            self.data_reader = IndraReader(data_repo_path)

            self.relation_type_q = IndraPromptFormat.relation_type_q
            self.relation_type_a = IndraPromptFormat.relation_type_a
        
        else:
            raise ValueError("Invalid data name: " + data_name)

    
    def create_prompt(self, task, n_shots=0):
                
        if hasattr(self.data_reader, "ent_types"):
            ent_types_included = {x: 0 for x in self.data_reader.ent_types}
            
            assert len(string.ascii_uppercase) >= len(self.data_reader.ent_types)
            
            ent_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.ent_types, string.ascii_uppercase)}
            self.ent_type_multiple_choices_str = ["(" + x + ") " + y for x, y in zip(string.ascii_uppercase, self.data_reader.ent_types)]
            self.ent_type_multiple_choices_str = " ".join(self.ent_type_multiple_choices_str)

            if self.data_name == "scierc":
                # change "OtherScientificTerm" to more readable and understandable.
                self.ent_type_multiple_choices_str = self.ent_type_multiple_choices_str.replace("OtherScientificTerm", "Other Scientific Term")

        if hasattr(self.data_reader, "rel_types"):
            rel_types_included = {x: 0 for x in self.data_reader.rel_types}
            
            assert len(string.ascii_uppercase) >= len(self.data_reader.rel_types)
            
            rel_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.rel_types, string.ascii_uppercase)}
            #self.rel_type_multiple_choices_str = ["(" + x + ") " + y for x, y in zip(string.ascii_uppercase, self.data_reader.rel_types)]
            #self.rel_type_multiple_choices_str = " ".join(self.rel_type_multiple_choices_str)
            
            self.rel_type_multiple_choices_str = ", ".join(['"' + x + '"' for x in self.data_reader.rel_types])
            
            
        
        
        data = self.data_reader.train_data
        
        if n_shots == 0:
            if self.data_name == "string":
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
            
            if self.data_name == "kegg":
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

        
        ## TODO: make it cleaner later.
        if self.data_name == "indra":
            # shuffle data to get random samples.
            random.shuffle(data)
            
            # get few-shot samples.
            for item in data:
                rel_type = item['rel_type']

                if all(value == n_shots for value in rel_types_included.values()):
                    break
                
                ## TODO: the current relation type file only contains "Activation", "Inhibition". 02/24/2023
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

                self.relation_type_prompt += text
                self.relation_type_prompt += self.relation_type_q(entity_1, entity_2, self.rel_type_multiple_choices_str)
                #self.relation_type_prompt += self.relation_type_a(rel_type_multiple_choices_dict[rel_type], rel_type)
                self.relation_type_prompt += self.relation_type_a(rel_type)
                
                
            #print(self.relation_type_prompt)
            #input('enter..')

        elif self.data_name == "kegg":
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
                    self.entity_prompt += self.entity_q(sample[0])
                    
                    ## TODO: for now, only use the first name of the list. 
                    entity_list = [x[0] for x in sample[1]]
                    
                    self.entity_prompt += self.entity_a(", ".join(entity_list))
            
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
                
                st = time.time()
                pathways = list(data.keys())
                for pathway in pathways:
                    for gene in all_genes:
                        if tuple([gene, pathway]) not in self.all_pos_relations:
                            self.all_neg_relations.add(tuple([gene, pathway]))
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                print('>> Execution time in hh:mm:ss:', td)			
                                
                st = time.time()
                # to generate the same random samples, convert it to an ordered list. 
                self.all_pos_relations = sorted(self.all_pos_relations)
                #self.all_neg_relations = sorted(self.all_neg_relations) # all_neg_relations is already sorted by combinations(). 
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                print('>> Execution time in hh:mm:ss:', td)
                
                
                pos_relations = random.sample(self.all_pos_relations, n_shots)
                neg_relations = random.sample(self.all_neg_relations, n_shots)
                

                for pos_rel, neg_rel in zip(pos_relations, neg_relations):
                    self.relation_prompt += self.relation_q(pos_rel[0], pos_rel[1])
                    #self.relation_prompt += self.relation_a("True")
                    self.relation_prompt += self.relation_a("yes")
                    #self.relation_prompt += self.relation_a("Related")
                    
                    self.relation_prompt += self.relation_q(neg_rel[0], neg_rel[1])
                    #self.relation_prompt += self.relation_a("False")
                    self.relation_prompt += self.relation_a("no")
                    #self.relation_prompt += self.relation_a("Unrelated")
                    
                    
                
                self.shot_samples.extend(pos_relations + neg_relations)				
                
                # exclude shot samples for inference.
                for i in pos_relations:
                    self.all_pos_relations.remove(i)
                
                for i in neg_relations:
                    self.all_neg_relations.remove(i)
                
                st = time.time()
                self.all_pos_relations = list(filter(lambda i: i not in pos_relations, self.all_pos_relations)) # this is much faster than the above way.
                self.all_neg_relations = list(filter(lambda i: i not in neg_relations, self.all_neg_relations)) # this is much faster than the above way.
                et = time.time()
                elapsed_time = et - st
                td = timedelta(seconds=elapsed_time)
                print('>> Execution time in hh:mm:ss:', td)

        elif self.data_name == "string":
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
                    self.entity_prompt += self.entity_q(sample[0])
                    self.entity_prompt += self.entity_a(", ".join(list(set(sample[1]))))
                
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
                    self.relation_prompt += self.relation_q(neg_pair[0], neg_pair[1])
                    #self.relation_prompt += self.relation_a("False")
                    self.relation_prompt += self.relation_a("no")
                    #self.relation_prompt += self.relation_a("Unrelated")
                    self.relation_prompt += self.relation_q(pos_pair[0], pos_pair[1])
                    #self.relation_prompt += self.relation_a("True")
                    self.relation_prompt += self.relation_a("yes")
                    #self.relation_prompt += self.relation_a("Related")
                        
                
                
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
                

        elif self.data_name == "scierc":
            # shuffle data to get random samples.
            random.shuffle(data)
            
            ## TODO: include all relation types as well (refer to the code above in INDRA). 02/24/2023
            # get few-shot samples.
            for item in data:
                id = item['id']
                text = item['text']
                entities = item['entities']
                relations = item['relations']
                directed = item['directed']
                
                if all(value >= n_shots for value in ent_types_included.values()):
                    break
                
                for ent in entities:
                    ent_types_included[ent[1]] += 1
                
                self.shot_samples.append(item)
                
                # debug
                '''
                print(id)
                print(text)
                print(entities)
                print(relations)
                print(directed)
                input('enter...')
                '''
            
            if self.data_name == "scierc":
                ent_types = ["Other Scientific Term" if x == "OtherScientificTerm" else x for x in self.data_reader.ent_types]

            for sample in self.shot_samples:
                self.entity_prompt += sample['text']
                self.entity_prompt += self.entity_q
                self.entity_prompt += self.entity_a(", ".join([x[0] for x in sample['entities']]))
            
                for ent_name, ent_type in sample['entities']:
                    self.entity_type_prompt += sample['text']
                    self.entity_type_prompt += self.entity_type_q(ent_name, self.ent_type_multiple_choices_str)
                    self.entity_type_prompt += self.entity_type_a(ent_type_multiple_choices_dict[ent_type], ent_type)
                    
                for ent1, ent2, rel_type in sample['relations']:
                    self.relation_type_prompt += sample['text']
                    self.relation_type_prompt += self.relation_type_q(ent1, ent2, self.rel_type_multiple_choices_str)
                    self.relation_type_prompt += self.relation_type_a(rel_type_multiple_choices_dict[rel_type], rel_type)
            
            # debug
            '''
            #print(shot_samples + '\n\n')
            print(self.entity_prompt)
            print(self.entity_type_prompt)
            print(self.relation_type_prompt)
            input('enter..')
            '''
        
        # debug
        if len(self.entity_prompt) != 0:
            print(self.entity_prompt)
        
        if len(self.relation_prompt) != 0:
            print(self.relation_prompt)
        
        if len(self.entity_type_prompt) != 0:
            print(self.entity_type_prompt)
        
        if len(self.relation_type_prompt) != 0:
            print(self.relation_type_prompt)
        
    
    def convert_to_IOB2(self, input_text, entities, labels=None):
        """
        Convert texts to IOB2 formats for NER evalution.
        
        """

        iob2_tagged_sent = []
        for entity in entities:
            
            ## TODO: find a better solution including other exceptions. 
            # add escape characters to parenthesis.
            entity = entity.replace('(', '\(').replace(')', '\)')
            
            # find all indices of the entity
            indices_obj = re.finditer(pattern=entity, string=input_text)
            entity_idx = [[idx.start(), idx.end()] for idx in indices_obj]
            
            tokenized_inputs = self.tokenizer(input_text, return_tensors="pt").to("cpu") # use cpu to avoid errors when running paralleism.
            input_ids = tokenized_inputs.input_ids
            input_ids = input_ids[0].tolist()
            tokens = tokenized_inputs.tokens()

            def get_token_idx(char_idx):
                while True:
                    # if it's the last index, return the last token.
                    if char_idx == len(input_text):
                        return len(tokenized_inputs) - 1
                    
                    token_idx = tokenized_inputs.char_to_token(char_idx)
                    # Whitespaces have no token and will return None.
                    if token_idx is not None:
                        return token_idx
                    char_idx += 1
                    
                    # debug
                    if char_idx == len(input_text):
                        raise Exception("End token not found: " f"{tokens}")

            e_start_indices, e_span_indices = [], []
            for start, end in entity_idx:
                e_span_s = get_token_idx(start)
                e_span_e = get_token_idx(end)
                e_start_indices.append(e_span_s)
                e_span_indices.append((e_span_s, e_span_e))
            
            def is_in_entity_span(tok_idx):
                for (e_span_s, e_span_e) in e_span_indices:
                    if tok_idx > e_span_s and tok_idx < e_span_e:
                        return True
                return False
            
            for token_idx, (token_id, token) in enumerate(zip(input_ids, tokens)):
                tag = 'O'
                if token_idx in e_start_indices:
                    tag = 'B-ENTITY'
                elif is_in_entity_span(token_idx):
                    tag = 'I-ENTITY'

                iob2_tagged_sent.append(token.strip() + ' ' + tag + '\n') # remove leading and trailing whitespaces of tokens.

            # debug
            if entity in input_text:
                if len(entity_idx) == 0:
                    print('>> tokenizer:', self.tokenizer)
                    print('>> input_text:', input_text)
                    print('>> entity:', entity)
                    print('>> input_ids:', input_ids)
                    print('>> tokens:', tokens)
                    print('>> e_span_indices:', e_span_indices)
                    print('>> iob2_tagged_sent:')
                    for i in iob2_tagged_sent:
                        print(i)
                assert len(entity_idx) != 0

        return iob2_tagged_sent
    

    def sort_and_pad(self, list_1, list_2):
        """
        # ref: https://stackoverflow.com/questions/73428068/how-to-append-item-to-match-the-length-of-two-list-in-python
        
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
                    
    
    def infer(self, model, task, batch_size=1):
        """
        Generate model inference with a batch sized input texts.
        
        """
        test_data = self.data_reader.test_data
        
        results = {'entity': [], 'relation': [], 'entity_type': [], 'relation_type': [], 'entity_relation': []}
        
        
        ## TODO: make it cleaner later.
        if self.data_name in ["string", "kegg"] and task == "entity":
            shots_keys = [x[0] for x in self.shot_samples]
        
        elif self.data_name == "indra" and task == "relation_type":
            shots_keys = [x['id'] for x in self.shot_samples]
        
        ## TODO: to reduce the number of test samples for a preliminary test. 
        if self.data_name == "string":
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
        
        elif self.data_name == "kegg":
            if task == "relation":
                
                test_sample_size = 'all' # 'all' for each positive and negative
                
                if test_sample_size == 'all':
                    # all_pos_relations is smaller than all_neg_relations. 
                    # - len(self.all_pos_relations): 17,552
                    # - len(self.all_neg_relations): 190,148
                    test_sample_size = len(self.all_pos_relations)

                pos_relations = random.sample(self.all_pos_relations, test_sample_size)
                neg_relations = random.sample(self.all_neg_relations, test_sample_size) # draw negative samples.
    
                pos_relations = [list(x) for x in pos_relations]
                for i in pos_relations:
                    #i.append("True")
                    i.append("yes")
                
                neg_relations = [list(x) for x in neg_relations]
                for i in neg_relations:
                    #i.append("False")
                    i.append("no")
                
                test_data = pos_relations + neg_relations
                
        elif self.data_name == "indra":
            if task == "relation_type":
                
                
                # use all data.
                test_data = self.data_reader.train_data + self.data_reader.dev_data + self.data_reader.test_data
                
                
                
                test_sample_size = 250 # for each class
                
                
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
            if self.data_name == "indra":
                
                if task == "relation_type":
                    batch_items = [] # debug
                    batch_input_texts = []
                    true_answer_list = []
                    
                    for item in batch_data:
                        batch_items.append(item)
                        text = item['text']
                        entity_1 = item['entity_1'][0]
                        entity_2 = item['entity_2'][0]
                        rel_type = item['rel_type']
                        
                        true_answer_list.append(rel_type)
                        
                        relation_type_prompt_with_test_sample = self.relation_type_prompt
                        relation_type_prompt_with_test_sample += text
                        relation_type_prompt_with_test_sample += self.relation_type_q(entity_1, entity_2, self.rel_type_multiple_choices_str)
                        
                        batch_input_texts.append(relation_type_prompt_with_test_sample)

                    pred_answer_list = model.generate(batch_input_texts)

                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    for item, pred_answer, true_answer in zip(batch_items, pred_answer_list, true_answer_list):
                        orig_pred_answer = pred_answer # debug
                        pred_answer = pred_answer.rsplit("\n\n", 1)[1]
                        pred_answer = pred_answer.replace("Answer: ", "", 1).replace("A: ", "", 1)
                        
                        # debug
                        if len(pred_answer.split()) == 0: # 0 means '' (no answer)
                            print(">> text:", item['text'])
                            print(">> entity_1:", item['entity_1'], ", entity_2:", item['entity_2'])
                            print(">> pred_answer:", pred_answer, ", true_answer:", true_answer)
                            print(">> orig_pred_answer:", orig_pred_answer)
                            input('enter..')
                            continue
                            
                        pred_answer = pred_answer.split()[-1] # remove numbers. E.g., (B) Inhibition -> Inhibition
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
                        #if pred_answer != 'Phosphorylation':
                        #print(">> The number of processed samples:", str(num))
                        print(">> text:", item['text'])
                        print(">> entity_1:", item['entity_1'], ", entity_2:", item['entity_2'])
                        print(">> pred_answer:", pred_answer, ", true_answer:", true_answer)
                        print(">> orig_pred_answer:", orig_pred_answer)
                        #print(">> relation_type_prompt:", self.relation_type_prompt)
                        if pred_answer == 'Deubiquitination':
                            input('enter..')
                        '''
                        
                            
                        
            elif self.data_name == "kegg":
                
                if task == "entity":
                    batch_items = [] # debug
                    batch_input_texts = []
                    true_entities_list = []
                    for item in batch_data:
                        # skip samples used in few-shots.
                        if item in shots_keys:
                            continue
                        
                        batch_items.append(item)
                        true_entities_list.append(test_data[item])
                        
                        entity_prompt_with_test_sample = self.entity_prompt
                        entity_prompt_with_test_sample += self.entity_q(item)
                        
                        batch_input_texts.append(entity_prompt_with_test_sample)

                    pred_entities_list = model.generate(batch_input_texts)

                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    for item, pred_entities, true_entities in zip(batch_items, pred_entities_list, true_entities_list):
                        orig_pred_answer = pred_entities # debug
                        pred_entities = pred_entities.rsplit("\n\n", 1)[1]
                        pred_entities = pred_entities.replace("Answer: ", "", 1).replace("A: ", "", 1)
                        pred_entities = [x.strip() for x in pred_entities.split(", ")]
                        
                        ## TODO: check if this is really needed. 02/04/2023
                        pred_entities = list(set(pred_entities)) # remove duplicates.
                        
                        ## TODO: for now, only use the first name of the list. 
                        #true_entities = [x[0] for x in true_entities]
                        
                        
                        # use all names.
                        # import chain
                        from itertools import chain
                        true_entities = list(chain.from_iterable(true_entities))

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
                        '''
                        #print(">> The number of processed samples:", str(num))
                        print(">> Which genes are involved in :", item)
                        print(">> src:", src)
                        print(">> pred_entities:", pred_entities)
                        print(">> true_entities:", true_entities)
                        print(">> orig_pred_answer:", orig_pred_answer)
                        input('enter..')
                        '''
                
                elif task in ["relation", "entity_relation"]:
                    batch_items = [] # debug
                    batch_input_texts = []
                    true_answer_list = []
                    for item in batch_data:
                        batch_items.append(item)
                        relation_prompt_with_test_sample = self.relation_prompt
                        relation_prompt_with_test_sample += self.relation_q(item[0], item[1])
                        
                        batch_input_texts.append(relation_prompt_with_test_sample)
                        
                        true_answer_list.append(item[2])

                    pred_answer_list = model.generate(batch_input_texts, task="relation")
                    
                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()
                    
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
                
                
            elif self.data_name == "string":
                if task == "entity":
                    batch_items = [] # debug
                    batch_input_texts = []
                    true_entities_list = []
                    for item in batch_data:
                        batch_items.append(item)
                        true_entities_list.append(test_data[item])
                        
                        entity_prompt_with_test_sample = self.entity_prompt
                        entity_prompt_with_test_sample += self.entity_q(item)
                        
                        batch_input_texts.append(entity_prompt_with_test_sample)
                    
                    
                    
                    
                    inputs = self.tokenizer(batch_input_texts, return_tensors="pt", padding=True)
                    generate_ids = model.generate(inputs.input_ids, max_length=500)
                    pred_entities_list = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    
                    print(pred_entities_list)
                    input('enter..')
                    
                    
                    
                    #pred_entities_list = model.generate(batch_input_texts)

                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()
                    
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
                        relation_prompt_with_test_sample += self.relation_q(item[0], item[1])
                        
                        batch_input_texts.append(relation_prompt_with_test_sample)
                        
                        true_answer_list.append(item[2])

                    pred_answer_list = model.generate(batch_input_texts, task='relation')
                    
                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()
                    
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
    
                        
            ## TODO: complete the code for batch sized input. 02/10/2023
            elif self.data_name == "scierc":
                batch_input_texts = []
                for item in batch_data:
                    id = item['id']
                    text = item['text']
                    entities = item['entities']
                    relations = item['relations']
                    directed = item['directed'] 
                    
                    '''
                    if text == "As new versions are added with appropriate tags and attributes in the original multilingual document , nothing is ever lost , and cooperative working on a document is rendered feasible .":
                        flag = True
                    
                    if not flag:
                        continue
                    
                    if self.data_name == "scierc":
                        ## TODO: fix this later.
                        if "``" in text:
                            print('Error text:', text)
                            continue
                    '''


                    entity_prompt_with_test_sample = self.entity_prompt
                    entity_prompt_with_test_sample += text
                    entity_prompt_with_test_sample += self.entity_q
                    #entity_prompt_with_test_sample += "A: "
                
                
                # [START] debug 
                '''
                input_ids = tokenizer(entity_prompt_with_test_sample, return_tensors="pt", padding=False).input_ids.to("cuda")
                
                #print(">> input_ids:", input_ids)
                #print(">> len(input_ids[0]):", len(input_ids[0]))
                #input('enter...')
                
                if len(input_ids[0]) > entity_prompt_max_length:
                    entity_prompt_max_length = len(input_ids[0])
                '''
                # [END] debug 

                # >> entity_prompt_max_length (SciERC): 716
                pred_entities = model.generate(entity_prompt_with_test_sample)
                
                # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                gc.collect()
                torch.cuda.empty_cache()

                pred_entities = pred_entities.rsplit("\n\n", 1)[1]
                pred_entities = pred_entities.replace("Answer: ", "", 1).replace("A: ", "", 1)
                pred_entities = [x.strip() for x in pred_entities.split(", ")]
                
                ## TODO: check if this is really needed. 02/04/2023
                pred_entities = list(set(pred_entities)) # remove duplicates.
                true_entities = [x[0] for x in entities]
                
                # Used for RST model.
                #rst_input = tokenizer.encode(entity_prompt_with_test_sample, return_tensors="pt")
                #pred_entities_by_rst = rst_model.generate(rst_input)
                #pred_entities_by_rst = tokenizer.decode(pred_entities_by_rst[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                '''
                print('>> entity_prompt_with_test_sample:', entity_prompt_with_test_sample)
                print('>> Pred entities:', pred_entities)
                #print('>> Pred entities by RST:', pred_entities_by_rst)
                print('>> True entities:', ", ".join([x[0] for x in entities]))
                input('enter...')
                '''
                
                #results['entity'].append([text, pred_entities, true_entities])
                
                
                #st = time.time()
                pred_entities_in_iob2_format = self.convert_to_IOB2(text, pred_entities, ['entity'])
                true_entities_in_iob2_format = self.convert_to_IOB2(text, true_entities, ['entity'])
                #et = time.time()
                #elapsed_time = et - st
                #td = timedelta(seconds=elapsed_time)
                #print('>> Execution time in hh:mm:ss:', td)
                #input('enter..')
                
                #results['entity']['pred'].extend(pred_entities_in_iob2_format)
                #results['entity']['true'].extend(true_entities_in_iob2_format)
                
                
                """
                ## TODO: check if a predicted entity is a true entity.
                for ent_name in pred_entities:
                    entity_type_prompt_with_test_sample = self.entity_type_prompt
                    entity_type_prompt_with_test_sample += text
                    entity_type_prompt_with_test_sample += self.entity_type_q(ent_name, self.ent_type_multiple_choices_str)
                
                
                    print('>> entity_type_prompt_with_test_sample:', entity_type_prompt_with_test_sample)
                    
                    
                    pred_ent_type = model.generate(entity_type_prompt_with_test_sample)
                    
                    # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                    gc.collect()
                    torch.cuda.empty_cache()

                    pred_ent_type = pred_ent_type.rsplit("\n\n", 1)[1]
                    pred_ent_type = pred_ent_type.replace("Answer: ", "", 1).replace("A: ", "", 1)
                    
                    # Used for RST model.			
                    #rst_input = tokenizer.encode(entity_type_prompt_with_test_sample, return_tensors="pt")
                    #pred_ent_type_by_rst = rst_model.generate(rst_input)
                    #pred_ent_type_by_rst = tokenizer.decode(pred_ent_type_by_rst[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    '''
                    print('>> entity_type_prompt_with_test_sample:', entity_type_prompt_with_test_sample)
                    print('>> Pred entity type:', pred_ent_type)
                    #print('>> Pred entity type by RST:', pred_ent_type_by_rst)
                    print('>> True entity type:', entities)
                    input('enter...')
                    '''
                    
                    
                    for ent1, ent2, rel_type in relations:
                        relation_type_prompt_with_test_sample = self.relation_type_prompt
                        relation_type_prompt_with_test_sample += text
                        relation_type_prompt_with_test_sample += self.relation_type_q(ent1, ent2, self.rel_type_multiple_choices_str)
                        
                        pred_rel_type = model.generate(relation_type_prompt_with_test_sample)
                        
                        # To fix the error of parallelize - RuntimeError: The size of tensor a (XXXX e.g., 5064) must match the size of tensor b (0) at non-singleton dimension 0 - 02/04/2023
                        gc.collect()
                        torch.cuda.empty_cache()

                        pred_rel_type = pred_rel_type.rsplit("\n\n", 1)[1]
                        pred_rel_type = pred_rel_type.replace("Answer: ", "", 1).replace("A: ", "", 1)
                        
                        #rst_input = tokenizer.encode(relation_type_prompt_with_test_sample, return_tensors="pt")
                        #pred_rel_type_by_rst = rst_model.generate(rst_input)
                        #pred_rel_type_by_rst = tokenizer.decode(pred_rel_type_by_rst[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        
                        '''
                        if self.data_name == "scierc":
                            if pred_rel_type != "(D) Evaluate-for":
                                print('>> relation_type_prompt_with_test_sample:', relation_type_prompt_with_test_sample)
                                print('>> Pred relation type:', pred_rel_type)
                                #print('>> Pred relation type by RST:', pred_rel_type_by_rst)
                                print('>> True relation type:', rel_type)
                                #input('enter...')
                        '''
                """
            
            #print(f">> batch processed - len(test_data): {len(test_data) - len(self.shot_samples)}, start: {start}, stop: {stop}")
            print(f">> batch processed - len(test_data): {len(test_data)}, start: {start}, stop: {stop}")

            #if stop >= (len(test_data) - len(self.shot_samples)):
            if stop >= len(test_data):
                break

            start = stop
            stop = start + batch_size 
            
            # debug
            #print(">> entity_prompt_max_length:", entity_prompt_max_length)
            

        
            #from parallelformers import parallelize
            #parallelize.deparallelize(model)
            
            #results['entity'] = [(self.convert_to_IOB2(x, y), self.convert_to_IOB2(x, z)) for x, y, z in results['entity']]
            
        return results
