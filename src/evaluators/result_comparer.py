"""

"""
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


#entity_task_result_file = '~/radbio/Entity_Relation_Extraction/result/string/entity/1-shot/entity_result.txt'
#entity_task_result_file = '~/radbio/Entity_Relation_Extraction/result/string/entity/1k/(1-shot) entity_result.txt'
#entity_task_result_file = '~/radbio/Entity_Relation_Extraction/result/kegg/entity/entity_result_2023-04-15 11:15:45.440396.txt'
entity_task_result_file = '~/radbio/Entity_Relation_Extraction/result/kegg/entity/low-dose/(1-shot) entity_result_2023-04-18 12:00:51.259557.txt'
entity_task_result_file = os.path.expanduser(entity_task_result_file)

ent_task_preds = []
with open(entity_task_result_file) as fin:
    for line in fin.readlines()[4:]:
        line = line.replace(',,', ',')
        
        src, pred, true = line.split(',')
        src = src.strip()
        pred = pred.strip()
        true = true.strip()
        
        ent_task_preds.append(bool(pred == true))
        

#entity_relation_task_result_file = '~/radbio/Entity_Relation_Extraction/result/string/entity_relation/1-shot (Question: Are {x} and {y} related to each other) yes or no_result_2023-04-13 07:39:44.368683.txt'
#entity_relation_task_result_file = '~/radbio/Entity_Relation_Extraction/result/kegg/entity_relation/entity_relation_result_2023-04-16 07:59:19.694537.txt'
entity_relation_task_result_file = '~/radbio/Entity_Relation_Extraction/result/kegg/entity_relation/entity_relation_result_2023-04-16 07:59:19.694537.txt'
entity_relation_task_result_file = os.path.expanduser(entity_relation_task_result_file)

rel_task_preds = []
with open(entity_relation_task_result_file) as fin:
    for line in fin.readlines()[4:]:
        line = line.replace(',,', ',')
        
        pred, true = line.split(',')
        pred = pred.strip()
        true = true.strip()
        
        rel_task_preds.append(True if pred == true else False)

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





