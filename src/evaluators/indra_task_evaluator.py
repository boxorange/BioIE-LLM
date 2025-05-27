"""
Date: 02/20/2024


"""

import os
import json
from itertools import chain


data_result_dict = {
    "indra": {
        # "BioGPT-Large": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioGPT/BioGPT-Large/indra/relation_type/relation_type_result_2024-02-20 18:40:54.532186.txt"},
        # "BioMedLM": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioMedLM/BioMedLM/indra/relation_type/relation_type_result_2024-02-20 18:43:36.656058.txt"},
        "galactica-6.7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-6.7b/indra/relation_type/relation_type_result_2024-02-20 18:19:56.991679.txt"},
        "galactica-30b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-30B/indra/relation_type/relation_type_result_2024-02-20 04:16:21.273263.txt"},
        # "Alpaca": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Alpaca/7B/indra/relation_type/relation_type_result_2024-02-20 01:36:07.347226.txt"},
        "RST": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/RST/rst-all-11b/indra/relation_type/relation_type_result_2024-02-20 18:35:07.668362.txt"},
        # "falcon-7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/indra/relation_type/relation_type_result_2024-02-20 18:16:50.344575.txt"},
        "falcon-40b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/indra/relation_type/relation_type_result_2024-02-20 12:26:59.574013.txt"},
        "mpt-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/indra/relation_type/relation_type_result_2024-02-20 18:12:43.997818.txt"},
        "mpt-30b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/indra/relation_type/relation_type_result_2024-02-20 12:32:41.067209.txt"},
        "Llama-2-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/indra/relation_type/relation_type_result_2024-02-20 01:22:02.262276.txt"},
        "Llama-2-70b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/indra/relation_type/relation_type_result_2024-02-20 04:23:29.423265.txt"},
        "Mistral-7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/indra/relation_type/relation_type_result_2024-02-19 23:58:14.292973.txt"},
        "Mixtral-8x7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/indra/relation_type/relation_type_result_2024-02-19 23:37:13.671506.txt"},
        "SOLAR-10.7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/indra/relation_type/relation_type_result_2024-02-20 01:03:05.625746.txt"}
    }
}

test_data_file = "/home/ac.gpark/BioIE-LLM-WIP/data/INDRA/converted/test.json"

test_data_dict = {}

with open(test_data_file) as fin:
    test_data = json.load(fin)
    
    for i in test_data:
        id = i["id"]
        text = i["text"]
        entity_1 = i['entity_1'][0] + '_' + str(i['entity_1'][1][0]) + '_' + str(i['entity_1'][1][1])
        entity_2 = i['entity_2'][0] + '_' + str(i['entity_2'][1][0]) + '_' + str(i['entity_2'][1][1])
        rel_type = i["rel_type"]
                
        entities = [entity_1, entity_2]
        
        key = '(' + ', '.join(sorted(entities)) + ')'

        if key not in test_data_dict:
            test_data_dict[key] = {'text': text, 'rel_type': rel_type}


'''
[{
        "id": "35288590807722125_21157_1885_1",
        "text": "Higher TTD resulted in a significant decrease of CgA after therapy .",
        "entity_1": ["Higher TTD", [0, 10]],
        "entity_2": ["significant decrease of CgA", [25, 52]],
        "rel_type": "Inhibition"
    }, {
        "id": "-22905172186968140_31923_2567_1",
        "text": "In a yeast two-hybrid interaction assay, the p.I784fs, p.N630fs, and p.S586fs mutations completely disrupted interaction of OFD1 with the fragment containing the predicted coiled-coil domains 1 and 2 of lebercilin, as evidenced by the fact that none of the reporter genes for interaction were activated.",
        "entity_1": ["OFD1", [124, 128]],
        "entity_2": ["lebercilin", [203, 213]],
        "rel_type": "Inhibition"
    }, {
        "id": "-7095764808016091_11892_4013_1",
        "text": "Addition of TNF (100 U/ml over 5 days) enhanced DNA synthesis from 718 +/- 284 (mean cpm +/- SE) to 2730 +/- 545 compared to cells cultured in medium alone (n = 16, p < 0.01).",
        "entity_1": ["TNF", [12, 15]],
        "entity_2": ["SE", [93, 95]],
        "rel_type": "Activation"
    }, {
        "id": "-21390623440207390_3178_9445_1",
        "text": "Endothelin-3 (ET-3) inhibited in a dose dependent, significant fashion prolactin release from cultured anterior pituitary cells (ovariectomized female and intact male rat donors, ED50 = 5 X 10 (-9) M).",
        "entity_1": ["ET-3", [14, 18]],
        "entity_2": ["prolactin", [71, 80]],
        "rel_type": "Inhibition"
    }, {
 '''   

# for k, v in test_data_dict.items():
    # print(k, v)
    # input('enter..')

    
indra_result_dict = {}


for data_name, model_dict in data_result_dict.items():
    for model_name, result_dict in model_dict.items():
        result_file = result_dict["result_file"]

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
                
                # print(result_file)
                # print(line)

                num_src, pred, true = line.rsplit(', ', 2)
                num, src = num_src.split(', ', 1)

                num = num.strip()
                src = src.strip()
                pred = pred.strip()
                true = true.strip()
                
                if src in test_data_dict:
                    text = test_data_dict[src]['text']
                else:
                    print('>> wrong key:', src)
                    

                if src in indra_result_dict:
                    indra_result_dict[src]['pred'].append(pred)
                    indra_result_dict[src]['models'].append(model_name)
                else:
                    indra_result_dict[src] = {'text': text, 
                                              'true': true,
                                              'pred': [pred],
                                              'models': [model_name]}

all_predicted_indra_list = []
none_predicted_indra_list = []

for src, value in indra_result_dict.items():
    if len(value['models']) == len(data_result_dict["indra"]):
        s_p = list(set(value['pred']))
        if len(s_p) == 1:
            text = value['text']
            pred = 'Pred: ' + s_p[0]
            true = 'Tred: ' + value['true']
            if value['true'] == s_p[0]:
                all_predicted_indra_list.append([src, pred, true, text])
            else:
                none_predicted_indra_list.append([src, pred, true, text])

with open('all_predicted_indra_list.txt', 'w+') as fout:
    for i in all_predicted_indra_list:
        fout.write(', '.join(i) + '\n')

with open('none_predicted_indra_list.txt', 'w+') as fout:
    for i in none_predicted_indra_list:
        fout.write(', '.join(i) + '\n')

