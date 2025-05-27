"""
Date: 08/28/2023



"""

import os
from itertools import chain
from sklearn.metrics import f1_score, accuracy_score


data_result_dict = {
    "string": {
        "BioGPT-Large": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioGPT/BioGPT-Large/string/entity/entity_result_2024-02-27 08:54:22.012860.txt"},
        "BioMedLM": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioMedLM/BioMedLM/string/entity/entity_result_2024-02-27 08:59:01.732846.txt"},
        "galactica-6.7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-6.7b/string/entity/entity_result_2024-02-27 09:18:18.009792.txt"},
        "galactica-30b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-30B/string/entity/entity_result_2023-08-26 21:11:53.444592.txt"},
        "Alpaca": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Alpaca/7B/string/entity/entity_result_2024-02-27 10:16:39.316170.txt"},
        "RST": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/RST/rst-all-11b/string/entity/entity_result_2024-02-27 18:25:10.236733.txt"},
        "falcon-7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/string/entity/entity_result_2024-02-27 09:37:30.557651.txt"},
        "falcon-40b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/string/entity/entity_result_2024-02-27 19:19:04.060634.txt"},
        "mpt-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/string/entity/entity_result_2024-02-27 09:53:49.632422.txt"},
        "mpt-30b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/string/entity/entity_result_2023-10-10 19:05:07.847651.txt"},
        "Llama-2-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/string/entity/entity_result_2024-02-04 22:33:34.871163.txt"},
        "Llama-2-70b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/string/entity/entity_result_2023-08-27 04:21:01.889030.txt"},
        "Mistral-7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/string/entity/entity_result_2024-02-04 22:51:29.670038.txt"},
        "Mixtral-8x7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/string/entity/entity_result_2024-02-05 04:28:28.731392.txt"},
        "SOLAR-10.7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/string/entity/entity_result_2024-02-11 21:16:48.861085.txt"}
    },
    "kegg": {
        "BioGPT-Large": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioGPT/BioGPT-Large/kegg/entity/entity_result_2024-02-10 23:23:20.865008.txt"},
        "BioMedLM": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioMedLM/BioMedLM/kegg/entity/entity_result_2024-02-27 09:02:24.001415.txt"},
        "galactica-6.7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-6.7b/kegg/entity/entity_result_2024-02-27 09:26:12.945650.txt"},
        "galactica-30b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-30B/kegg/entity/entity_result_2023-08-29 05:58:31.580017.txt"},
        "Alpaca": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Alpaca/7B/kegg/entity/entity_result_2024-02-27 10:43:44.378075.txt"},
        "RST": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/RST/rst-all-11b/kegg/entity/entity_result_2024-02-27 11:29:57.832713.txt"},
        "falcon-7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/kegg/entity/entity_result_2024-02-27 09:44:08.834211.txt"},
        "falcon-40b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/kegg/entity/entity_result_2024-02-27 10:26:12.225201.txt"},
        "mpt-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/kegg/entity/entity_result_2024-02-01 20:23:07.014361.txt"},
        "mpt-30b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/kegg/entity/entity_result_2023-10-10 20:28:20.806244.txt"},
        "Llama-2-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/kegg/entity/entity_result_2024-02-11 22:08:06.677694.txt"},
        "Llama-2-70b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/kegg/entity/entity_result_2023-08-29 04:52:01.878161.txt"},
        "Mistral-7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/kegg/entity/entity_result_2024-02-05 22:23:35.207548.txt"},
        "Mixtral-8x7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kegg/entity/entity_result_2024-02-10 17:53:16.915981.txt"},
        "SOLAR-10.7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/kegg/entity/entity_result_2024-02-11 16:39:25.951369.txt"}
    }
}

for data_name, model_dict in data_result_dict.items():
    for model_name, result_dict in model_dict.items():
        result_file = result_dict["result_file"]
        
        y_true = []
        y_pred = []
        with open(result_file) as fin:
            lines = fin.readlines()
            delimiter_idx = lines.index("********************************************************************\n")
            
            # stop at the original text if exists.
            if "####################################################################\n" in lines:
                end_idx = lines.index("####################################################################\n")
            else:
                end_idx = -1

            for idx, line in enumerate(lines[delimiter_idx+1:], delimiter_idx+1):
                
                if idx == end_idx:
                    break
                    
                num_src, pred, true = line.rsplit(', ', 2)
                num, src = num_src.split(', ', 1)
                
                # if len(line.rsplit(', ')) != 2:
                    # print('>> line with unexpected inference:', line)
                    # input('enter..')
                    # continue
                # num, src, pred, true = line.split(', ')
                
                num = num.strip()
                src = src.strip()
                pred = pred.strip()
                true = true.strip()

                y_true.append(src)
                y_pred.append(src if pred == true else pred)

        macro_f1 = f1_score(y_true, y_pred, average='macro')

        y_full_true = list(set(y_true))
        y_full_true = sorted(y_full_true)
        y_full_pred = []
        y_none_pred = []

        for i in y_full_true:
            if y_pred.count(i) == 10:
                y_full_pred.append(i)
                # print(i)
            else:
                y_full_pred.append('None')
            
            if y_pred.count(i) == 0:
                y_none_pred.append(i)

        # print(accuracy_score(y_full_true, y_full_pred))
        # print(len([x for x in y_full_pred if x != 'None']))
        
        result_dict['macro_f1'] = macro_f1
        result_dict['full_match'] = [x for x in y_full_pred if x != 'None']
        result_dict['none_match'] = [x for x in y_none_pred]

# sort items by micro f1 score.
for data_name, model_dict in data_result_dict.items():
    data_result_dict[data_name] = dict(sorted(model_dict.items(), key=lambda item: item[1]['macro_f1'], reverse=True))

# get all models.
top_N = len(data_result_dict["string"])

# get the top 5 models.
# top_N = 5

def common_elements_intersection(lists):
    """Finds the common elements between all lists using set intersection.

    Args:
        lists (list of list): A list of lists to find common elements from.

    Returns:
        set: A set containing the common elements.
    """

    if not lists:
        return set()

    intersection = set(lists[0])
    for l in lists[1:]:
        intersection &= set(l)  # Efficient intersection using "&="
    return intersection
    
for data_name, model_dict in data_result_dict.items():
    full_match_list_of_top_N_models = []
    none_match_list_of_top_N_models = []
    
    unique_full_match_dict = {}
    
    for model_name, result_dict in dict(list(model_dict.items())[0: top_N]).items():
        print(f">> {model_name} ({data_name}) - Macro F1: {result_dict['macro_f1']:.4f}, Full match: {len(result_dict['full_match'])}, None match: {len(result_dict['none_match'])}")
        
        full_match_list_of_top_N_models.append(result_dict['full_match'])
        none_match_list_of_top_N_models.append(result_dict['none_match'])
        
        # full_match_list_of_top_N_models.extend(result_dict['full_match'])
        # none_match_list_of_top_N_models.extend(result_dict['none_match'])
        
        key = model_name + ' (' + data_name + ')'
        unique_full_match_dict[key] = result_dict['full_match']

    # print('>> full_match_list_of_top_N_models:', ', '.join(list(set(full_match_list_of_top_N_models))))
    # print('>> none_match_list_of_top_N_models:', ', '.join(list(set(none_match_list_of_top_N_models))))
    
    common_elems = common_elements_intersection(full_match_list_of_top_N_models)
    print('>> common elems in full_match_list_of_top_N_models:', ', '.join(list(common_elems)))
    
    common_elems = common_elements_intersection(none_match_list_of_top_N_models)
    print('>> common elems in none_match_list_of_top_N_models:', ', '.join(list(common_elems)))
    
    for k, v in unique_full_match_dict.items():
        other_full_match_list = [vv for kk, vv in unique_full_match_dict.items() if kk != k]
        other_full_match_list = list(chain.from_iterable(other_full_match_list))
        unique_full_match_dict[k] = list(set(v) - set(other_full_match_list))
    
    for k, v in unique_full_match_dict.items():
        print(f">> unique full match list of {k}: {v} (Total number: {len(v)})")
    
    print("-----------------------------------------------------")
    
# common_values = list(set(pred) & set(true))
# new_pred = common_values + list(set(pred) - set(common_values))
# new_true = common_values + list(set(true) - set(common_values))




