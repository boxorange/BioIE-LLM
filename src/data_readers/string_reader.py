import os
import json


class StringReader:
	
	def __init__(self, path, task, relation_query_answers):
		path = os.path.expanduser(path)
		
		if task == 'entity_relation':
			# Galactica
			#entity_task_result_file = '~/BioIE-LLM/result/Galactica/standard/string/entity/entity_result_2023-05-27 15:58:01.783728.txt'
			
			# LLaMA
			#entity_task_result_file = '~/BioIE-LLM/result/LLaMA/7B/string/entity/entity_result_2023-05-27 20:03:38.317080.txt'
			
			# Alpaca
			#entity_task_result_file = '~/BioIE-LLM/result/Alpaca/7B/string/entity/entity_result_2023-05-22 19:59:01.535768.txt'
			
			# RST
			entity_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/string/entity/entity_result_2023-05-30 11:25:22.210287.txt"
			
			# BioGPT
			#entity_task_result_file = '~/BioIE-LLM/result/BioGPT/BioGPT-Large/string/entity/entity_result_2023-05-26 09:47:13.072443.txt'
			
			# BioMedLM
			#entity_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/string/entity/entity_result_2023-05-29 16:25:04.623927.txt"
			
			entity_task_result_file = os.path.expanduser(entity_task_result_file)
			
			prot_pairs = []
			with open(entity_task_result_file) as fin:
				lines = fin.readlines()
				delimiter_idx = lines.index("********************************************************************\n")
				for line in lines[delimiter_idx+1:]:
					
					'''
					# ignore predictions with 
					skip_substring_list = ['Question:', 'FREETEXT', 'PARAGRAPH']
					if any(substring in line for substring in skip_substring_list):
						continue
					'''
					
					
					line = line.replace(',,', ',')
					
					
					if len(line.split(',')) != 3:
						print('>> line with unexpected inference:', line)
						continue
					
					
					src, pred, true = line.split(',')
					src = src.strip()
					pred = pred.strip()
					true = true.strip()
					
					prot_pairs.append([src, true, relation_query_answers[0]])

			self.test_data = prot_pairs
			
			converted_data_dir = os.path.join(path, "STRING/converted")
			self.train_data = json.load(open(os.path.join(converted_data_dir, "protein_binding_info.json")))
		else:
			orig_data_dir = os.path.join(path, "STRING/original")
			converted_data_dir = os.path.join(path, "STRING/converted")
			
			if not os.path.exists(converted_data_dir):
			   os.makedirs(converted_data_dir)
			
			if len(os.listdir(converted_data_dir)) == 0:
				self._convert_data(orig_data_dir, converted_data_dir)
			
			self.train_data = json.load(open(os.path.join(converted_data_dir, "protein_binding_info.json")))
			self.test_data = json.load(open(os.path.join(converted_data_dir, "protein_binding_info.json")))
			

	def _convert_data(self, orig_data_dir, converted_data_dir):
		# read protein names.
		protein_name_dict = {}
		with open(os.path.join(orig_data_dir, "9606.protein.info.v11.5.txt")) as fin:
			next(fin)
			for line in fin:
				string_protein_id, preferred_name, protein_size, annotation = line.split('\t')
				string_protein_id = string_protein_id.strip()
				preferred_name = preferred_name.strip()
				protein_name_dict[string_protein_id] = preferred_name

		# read protein binding information.
		protein_binding_dict = {}
		with open(os.path.join(orig_data_dir, "9606.protein.links.v11.5.txt")) as fin:
			next(fin)
			for line in fin:
				protein_1_id, protein_2_id, combined_score = line.split(' ')
				protein_1_id = protein_1_id.strip()
				protein_2_id = protein_2_id.strip()
				
				protein_1_name = protein_name_dict[protein_1_id]
				protein_2_name = protein_name_dict[protein_2_id]
				
				if protein_1_name in protein_binding_dict:
					protein_binding_dict[protein_1_name].add(protein_2_name)
				else:
					protein_binding_dict[protein_1_name] = set([protein_2_name])
				
				if protein_2_name in protein_binding_dict:
					protein_binding_dict[protein_2_name].add(protein_1_name)
				else:
					protein_binding_dict[protein_2_name] = set([protein_1_name])
		
		protein_binding_dict = {k: list(v) for k, v in protein_binding_dict.items()}
		
		outfile = "protein_binding_info.json"
		outfile = os.path.join(converted_data_dir, outfile)

		with open(outfile, "w") as fout:
			json.dump(protein_binding_dict, fout)
