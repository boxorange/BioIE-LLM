import sys
import time
import string
import random
import re
import itertools

from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import StringReader

from .base_processor import BaseProcessor

random.seed(42)


class StringProcessor(BaseProcessor):
	def __init__(
		self, 
		data_name: str, 
		data_repo_path: str,
		task: str,
		test_sample_size: int, 
		model_name: str,
		tokenizer,
	):
		super().__init__(data_name, task, test_sample_size, model_name, tokenizer)

		# pass task argument for entity_relation task. 04/12/2023
		self.data_reader = StringReader(data_repo_path, task, self.relation_query_answers)
		
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
		data = self.data_reader.train_data
		
		self.task_prompt[task] = ""
		
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
				self.task_prompt[task] += self.model_prompt['entity_q'](sample[0])
				self.task_prompt[task] += self.model_prompt['entity_a'](", ".join(sorted(list(set(sample[1]))))) # Reproducibility - the order of list affects the model inference.
			
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
			
			# test code for various orders of pos and neg samples.
			'''
			self.task_prompt[task] += self.relation_q(neg_prot_pairs[0][0], neg_prot_pairs[0][1])
			self.task_prompt[task] += self.relation_a("False")
			
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[0][0], pos_prot_pairs[0][1])
			self.task_prompt[task] += self.relation_a("True")

			self.task_prompt[task] += self.relation_q(pos_prot_pairs[1][0], pos_prot_pairs[1][1])
			self.task_prompt[task] += self.relation_a("True")
			
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[2][0], pos_prot_pairs[2][1])
			self.task_prompt[task] += self.relation_a("True")
			
			self.task_prompt[task] += self.relation_q(neg_prot_pairs[1][0], neg_prot_pairs[1][1])
			self.task_prompt[task] += self.relation_a("False")
			
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[3][0], pos_prot_pairs[3][1])
			self.task_prompt[task] += self.relation_a("True")
				
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[4][0], pos_prot_pairs[4][1])
			self.task_prompt[task] += self.relation_a("True")

			self.task_prompt[task] += self.relation_q(pos_prot_pairs[5][0], pos_prot_pairs[5][1])
			self.task_prompt[task] += self.relation_a("True")
			
			self.task_prompt[task] += self.relation_q(neg_prot_pairs[2][0], neg_prot_pairs[2][1])
			self.task_prompt[task] += self.relation_a("False")
			
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[6][0], pos_prot_pairs[6][1])
			self.task_prompt[task] += self.relation_a("True")
				
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[7][0], pos_prot_pairs[7][1])
			self.task_prompt[task] += self.relation_a("True")

			self.task_prompt[task] += self.relation_q(pos_prot_pairs[8][0], pos_prot_pairs[8][1])
			self.task_prompt[task] += self.relation_a("True")
			
			self.task_prompt[task] += self.relation_q(pos_prot_pairs[9][0], pos_prot_pairs[9][1])
			self.task_prompt[task] += self.relation_a("True")
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
		
		## TODO: make it cleaner later.
		if task == "entity":
			shots_keys = [x[0] for x in self.shot_samples]

		## TODO: to reduce the number of test samples for a preliminary test. 
		if task == "entity":
			test_sample_size = self.test_sample_size # e.g., 1000
			
			# skip samples used in few-shots.
			test_data = {k: v for k, v in test_data.items() if k not in shots_keys}
			
			## TODO: remove if-condition later. this is to filter out samples with many answers (long list of proteins).
			#test_data = {k: v for k, v in test_data.items() if len(v) <= self.max_entity_list_len}	
			#print(len(test_data)) # if len(v) <= 20 --> 163, if len(v) <= 30 --> 287, if len(v) <= 50 --> 559
			
			sample_keys = random.sample(sorted(test_data), test_sample_size)
			test_data = {k: v for k, v in test_data.items() if k in sample_keys}
			
		elif task == "relation":
			test_sample_size = self.test_sample_size # e.g., 500 for each positive and negative
			
			pos_prot_pairs = random.sample(self.all_pos_prot_pairs, test_sample_size)
			neg_prot_pairs = random.sample(self.all_neg_prot_pairs, test_sample_size) # draw negative samples.

			pos_prot_pairs = [list(x) for x in pos_prot_pairs]
			for i in pos_prot_pairs:
				i.append(self.relation_query_answers[0])
			
			neg_prot_pairs = [list(x) for x in neg_prot_pairs]
			for i in neg_prot_pairs:
				i.append(self.relation_query_answers[1])
			
			test_data = pos_prot_pairs + neg_prot_pairs

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
					print(">> Which proteins are related to:", item)
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
