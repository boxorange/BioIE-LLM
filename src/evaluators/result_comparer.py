"""

"""
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

'''
# Galactica
#entity_task_result_file = '~/BioIE-LLM/result/Galactica/standard/kegg/entity/entity_result_2023-05-22 17:57:31.785102.txt'
entity_task_result_file = '~/BioIE-LLM/result/Galactica/standard/string/entity/entity_result_2023-05-27 15:58:01.783728.txt'
entity_task_result_file = os.path.expanduser(entity_task_result_file)

#entity_relation_task_result_file = "~/BioIE-LLM/result/Galactica/standard/kegg/entity_relation/entity_relation_result_2023-05-28 18:49:50.302782.txt"
entity_relation_task_result_file = "~/BioIE-LLM/result/Galactica/standard/string/entity_relation/entity_relation_result_2023-05-28 20:00:42.280054.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)
'''

'''
# LLaMA
entity_task_result_file = "~/BioIE-LLM/result/LLaMA/7B/string/entity/entity_result_2023-05-27 20:03:38.317080.txt"
#entity_task_result_file = "~/BioIE-LLM/result/LLaMA/7B/kegg/entity/entity_result_2023-05-22 18:31:54.660388.txt"
entity_task_result_file = os.path.expanduser(entity_task_result_file)

entity_relation_task_result_file = "~/BioIE-LLM/result/LLaMA/7B/string/entity_relation/entity_relation_result_2023-05-30 03:18:49.766152.txt"
#entity_relation_task_result_file = "~/BioIE-LLM/result/LLaMA/7B/kegg/entity_relation/entity_relation_result_2023-05-28 18:43:45.610207.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)
'''

'''
# Alpaca
#entity_task_result_file = '~/BioIE-LLM/result/Alpaca/7B/string/entity/entity_result_2023-05-22 19:59:01.535768.txt'
entity_task_result_file = "~/BioIE-LLM/result/Alpaca/7B/kegg/entity/entity_result_2023-05-22 18:52:04.698486.txt"
entity_task_result_file = os.path.expanduser(entity_task_result_file)

#entity_relation_task_result_file = "~/BioIE-LLM/result/Alpaca/7B/string/entity_relation/entity_relation_result_2023-05-28 15:24:35.467452.txt"
entity_relation_task_result_file = "~/BioIE-LLM/result/Alpaca/7B/kegg/entity_relation/entity_relation_result_2023-05-28 17:47:20.284434.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)
'''

'''
# RST
#entity_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/string/entity/entity_result_2023-05-30 11:25:22.210287.txt"
entity_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/kegg/entity/entity_result_2023-05-30 10:33:30.006105.txt"
entity_task_result_file = os.path.expanduser(entity_task_result_file)

#entity_relation_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/string/entity_relation/entity_relation_result_2023-05-30 18:45:25.230453.txt"
entity_relation_task_result_file = "~/BioIE-LLM/result/RST/rst-all-11b/kegg/entity_relation/entity_relation_result_2023-05-31 14:59:51.252483.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)
'''

'''
# BioGPT
#entity_task_result_file = '~/BioIE-LLM/result/BioGPT/BioGPT-Large/string/entity/entity_result_2023-05-26 09:47:13.072443.txt'
entity_task_result_file = "~/BioIE-LLM/result/BioGPT/BioGPT-Large/kegg/entity/entity_result_2023-05-27 10:18:54.554211.txt"
entity_task_result_file = os.path.expanduser(entity_task_result_file)

#entity_relation_task_result_file = "~/BioIE-LLM/result/BioGPT/BioGPT-Large/string/entity_relation/entity_relation_result_2023-05-28 11:17:21.450173.txt"
entity_relation_task_result_file = "~/BioIE-LLM/result/BioGPT/BioGPT-Large/kegg/entity_relation/entity_relation_result_2023-05-28 17:52:08.720373.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)
'''


# BioMedLM
#entity_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/string/entity/entity_result_2023-05-29 16:25:04.623927.txt"
entity_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/kegg/entity/entity_result_2023-05-31 11:33:50.389967.txt"
entity_task_result_file = os.path.expanduser(entity_task_result_file)

#entity_relation_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/string/entity_relation/entity_relation_result_2023-05-30 11:01:47.024225.txt"
entity_relation_task_result_file = "~/BioIE-LLM/result/BioMedLM/BioMedLM/kegg/entity_relation/entity_relation_result_2023-05-31 15:22:31.034616.txt"
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)



ent_task_preds = []
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
		
		ent_task_preds.append(bool(pred == true))
		
rel_task_preds = []
with open(entity_relation_task_result_file) as fin:
	lines = fin.readlines()
	delimiter_idx = lines.index("********************************************************************\n")
	for line in lines[delimiter_idx+1:]:
		line = line.replace(',,', ',')
		
		pred, true = line.split(',')
		pred = pred.strip()
		true = true.strip()
		
		rel_task_preds.append(bool(pred == true))

print('len(ent_task_preds):', len(ent_task_preds))
print(accuracy_score(ent_task_preds, rel_task_preds))
print(accuracy_score(ent_task_preds, rel_task_preds, normalize=False))
print(f1_score(ent_task_preds, rel_task_preds, average='micro'))

updated_ent_task_preds = []
updated_rel_task_preds = []

for e, r in zip(ent_task_preds, rel_task_preds):
	if e == True:
		updated_ent_task_preds.append(e)
		updated_rel_task_preds.append(r)

print('len(updated_ent_task_preds):', len(updated_ent_task_preds))
print(accuracy_score(updated_ent_task_preds, updated_rel_task_preds))
print(f1_score(updated_ent_task_preds, updated_rel_task_preds, average='micro'))

true = ['yes' if x else 'no' for x in updated_ent_task_preds] 
pred = ['yes' if x else 'no' for x in updated_rel_task_preds] 

cm = confusion_matrix(true, pred, labels=['yes', 'no'])
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['yes', 'no'])
cm_disp.plot()
cm_disp.figure_.savefig("confusion_matrix.png", dpi=300)





