import os
import json


class IndraReader:
	
	def __init__(self, path, num_of_indra_classes):
		path = os.path.expanduser(path)
		
		orig_data_dir = os.path.join(path, "INDRA/original")
		converted_data_dir = os.path.join(path, "INDRA/converted")
		rel_type_file = os.path.join(path, "INDRA/converted/relation_types.json")
		
		if not os.path.exists(converted_data_dir):
		   os.makedirs(converted_data_dir)
		
		if len(os.listdir(converted_data_dir)) == 0:
			self._convert_data(orig_data_dir, converted_data_dir)
		
		self.train_data = json.load(open(os.path.join(converted_data_dir, "train.json")))
		self.dev_data = json.load(open(os.path.join(converted_data_dir, "dev.json")))
		self.test_data = json.load(open(os.path.join(converted_data_dir, "test.json")))
		
		rel_type_data = json.load(open(rel_type_file))
		
		self.rel_types = [x for x in rel_type_data]
		self.rel_types = self.rel_types[:num_of_indra_classes]


	def _convert_data(self, orig_data_dir, converted_data_dir):
		
		for file in os.listdir(orig_data_dir):
			if file not in ['train.json', 'dev.json', 'test.json']:
				continue
			
			output_data = []

			with open(os.path.join(orig_data_dir, file)) as fin:
				for line in fin.readlines():
					entry = json.loads(line)

					"""
					    "relation": [{
							"relation_type": "Phosphorylation",
							"relation_id": 25,
							"entity_1": "STAT1",
							"entity_1_idx": [12, 17],
							"entity_1_idx_in_text_with_entity_marker": [16, 21],
							"entity_1_type": "GENE",
							"entity_1_type_id": 0,
							"entity_2": "MKP-1",
							"entity_2_idx": [57, 62],
							"entity_2_idx_in_text_with_entity_marker": [70, 75],
							"entity_2_type": "GENE",
							"entity_2_type_id": 0
						}
					],
					"""


					
					id = entry['id']
					text = entry['text']
					entity_1 = (entry['relation'][0]['entity_1'], entry['relation'][0]['entity_1_idx'])
					entity_2 = (entry['relation'][0]['entity_2'], entry['relation'][0]['entity_2_idx'])
					rel_type = entry['relation'][0]['relation_type']
					
					
					## TODO: for now, only consider texts containing the entities only once. 02/24/2023
					#tokenized_text = text.split()
					#if tokenized_text.count(entity_1[0]) == 1 and tokenized_text.count(entity_2[0]) == 1:
					if text.count(entity_1[0]) == 1 and text.count(entity_2[0]) == 1:
						converted = {"id": id,
									 "text": text,
									 "entity_1": entity_1,
									 "entity_2": entity_2,
									 "rel_type": rel_type}
									 
						output_data.append(converted)

			outfile = os.path.join(converted_data_dir, file)

			with open(outfile, "w") as fout:
				json.dump(output_data, fout)
		
