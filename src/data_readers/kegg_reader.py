import os
import json
import re


class KeggReader:
	
	def __init__(self, path, task, kegg_data_type, relation_query_answers):
		path = os.path.expanduser(path)

		if task == 'entity_relation':
			if kegg_data_type == 'high-dose':
				# Galactica
				entity_task_result_file = '~/radbio/Entity_Relation_Extraction/result/kegg/entity/entity_result_2023-04-15 11:15:45.440396.txt'
			elif kegg_data_type == 'low-dose':
				# Galactica
				#entity_task_result_file = "~/BioIE-LLM/result/Galactica/standard/kegg/entity/entity_result_2023-05-22 17:57:31.785102.txt"
				
				# LLaMA
				#entity_task_result_file = "~/BioIE-LLM/result/LLaMA/7B/kegg/entity/entity_result_2023-05-22 18:31:54.660388.txt"
				
				# Alpaca
				#entity_task_result_file = "~/BioIE-LLM/result/Alpaca/7B/kegg/entity/entity_result_2023-05-22 18:52:04.698486.txt"
				
				# RST
				entity_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/kegg/entity/entity_result_2023-05-30 10:33:30.006105.txt"
				
				# BioGPT
				#entity_task_result_file = "~/BioIE-LLM/result/BioGPT/BioGPT-Large/kegg/entity/entity_result_2023-05-27 10:18:54.554211.txt"
				
				# BioMedLM
				#entity_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/kegg/entity/entity_result_2023-05-31 11:33:50.389967.txt"
			
			entity_task_result_file = os.path.expanduser(entity_task_result_file)

			pathway_relations = []
			with open(entity_task_result_file) as fin:
				lines = fin.readlines()
				delimiter_idx = lines.index("********************************************************************\n")
				for line in lines[delimiter_idx+1:]:
					line = line.replace(',,', ',')
					
					if len(line.split(',')) != 3:
						print('>> line with unexpected inference:', line)
						continue
						
					src, pred, true = line.split(',')
					src = src.strip()
					pred = pred.strip()
					true = true.strip()

					pathway_relations.append([src, true, relation_query_answers[0]])
		
			self.test_data = pathway_relations
			
			converted_data_dir = os.path.join(path, "KEGG/converted")
			
			if kegg_data_type == 'high-dose':
				self.train_data = json.load(open(os.path.join(converted_data_dir, "high_dose_pathway_genes.json")))
			elif kegg_data_type == 'low-dose':
				self.train_data = json.load(open(os.path.join(converted_data_dir, "low_dose_pathway_genes.json")))
		else:
			orig_data_dir = os.path.join(path, "KEGG/original")
			converted_data_dir = os.path.join(path, "KEGG/converted")
			
			if not os.path.exists(converted_data_dir):
			   os.makedirs(converted_data_dir)
			
			if len(os.listdir(converted_data_dir)) == 0:
				self._convert_data(orig_data_dir, converted_data_dir)
			
			## TODO: split data into train/dev/test.
			if kegg_data_type == 'high-dose':
				self.train_data = json.load(open(os.path.join(converted_data_dir, "high_dose_pathway_genes.json")))
				self.test_data = json.load(open(os.path.join(converted_data_dir, "high_dose_pathway_genes.json")))
			elif kegg_data_type == 'low-dose':
				self.train_data = json.load(open(os.path.join(converted_data_dir, "low_dose_pathway_genes.json")))
				self.test_data = json.load(open(os.path.join(converted_data_dir, "low_dose_pathway_genes.json")))


	def _convert_data(self, orig_data_dir, converted_data_dir):
		# read high/low dose pathways.
		high_dose_pathways = []
		with open(os.path.join(orig_data_dir, "high_dose_pathways.txt")) as fin:
			for line in fin:
				high_dose_pathways.append(line.strip())
		
		low_dose_pathways = []
		with open(os.path.join(orig_data_dir, "low_dose_pathways.txt")) as fin:
			for line in fin:
				low_dose_pathways.append(line.strip())
				
		# read pathway names.
		pathway_name_dict = {}
		with open(os.path.join(orig_data_dir, "kegg.pathway.hsa")) as fin:
			for line in fin:
				pathway_id, pathway_name = line.split('\t')
				pathway_id = pathway_id.replace("path:", "").strip()
				pathway_name = pathway_name.replace(" - Homo sapiens (human)", "").strip()
				pathway_name_dict[pathway_id] = pathway_name
		
		# read gene names.
		gene_name_dict = {}
		with open(os.path.join(orig_data_dir, "kegg.hsa.genes")) as fin:
			for line in fin:
				gene_id, gene_name = line.split('\t')
				gene_id = gene_id.strip()
				gene_name = re.split(r'[,;]', gene_name)
				gene_name = [x.strip() for x in gene_name]
				gene_name_dict[gene_id] = gene_name
				
		# read pathway genes.
		pathway_gene_dict = {}
		with open(os.path.join(orig_data_dir, "kegg.hsa.pathway.genes")) as fin:
			for line in fin:
				gene_id, pathway_id = line.split('\t')
				gene_id = gene_id.strip()
				pathway_id = pathway_id.strip().replace("path:", "")
				if pathway_id in pathway_gene_dict:
					pathway_gene_dict[pathway_id].append(gene_id)
				else:
					pathway_gene_dict[pathway_id] = [gene_id]

		# find high/low dose pathway genes.
		high_dose_pathway_gene_dict = {}
		for pathway_id in high_dose_pathways:
			pathway_name = pathway_name_dict[pathway_id]
			gene_ids = pathway_gene_dict[pathway_id]
			gene_names = [gene_name_dict[x] for x in gene_ids]
			high_dose_pathway_gene_dict[pathway_name] = gene_names
		
		outfile = "high_dose_pathway_genes.json"
		outfile = os.path.join(converted_data_dir, outfile)

		with open(outfile, "w") as fout:
			json.dump(high_dose_pathway_gene_dict, fout)
			
		low_dose_pathway_gene_dict = {}
		for pathway_id in low_dose_pathways:
			pathway_name = pathway_name_dict[pathway_id]
			gene_ids = pathway_gene_dict[pathway_id]
			gene_names = [gene_name_dict[x] for x in gene_ids]
			low_dose_pathway_gene_dict[pathway_name] = gene_names
		
		outfile = "low_dose_pathway_genes.json"
		outfile = os.path.join(converted_data_dir, outfile)

		with open(outfile, "w") as fout:
			json.dump(low_dose_pathway_gene_dict, fout)
		
