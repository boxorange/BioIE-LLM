"""
Date: 02/20/2024


"""

import os
from itertools import chain


data_result_dict = {
    "string": {
        # "BioGPT-Large": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioGPT/BioGPT-Large/string/relation/relation_result_2024-02-12 14:39:24.126727.txt"},
        "BioMedLM": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/BioMedLM/BioMedLM/string/relation/relation_result_2024-02-12 01:32:27.084012.txt"},
        # "galactica-6.7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-6.7b/string/relation/relation_result_2024-02-12 01:37:04.294908.txt"},
        "galactica-30b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Galactica/galactica-30B/string/relation/relation_result_2024-02-12 03:12:51.925268.txt"},
        "Alpaca": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Alpaca/7B/string/relation/relation_result_2024-02-12 03:33:24.844953.txt"},
        "RST": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/RST/rst-all-11b/string/relation/relation_result_2024-02-12 03:38:45.325783.txt"},
        # "falcon-7b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/string/relation/relation_result_2024-02-12 02:13:03.963940.txt"},
        # "falcon-40b": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/string/relation/relation_result_2024-02-12 02:21:19.909487.txt"},
        "mpt-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/string/relation/relation_result_2024-02-12 02:02:36.148598.txt"},
        "mpt-30b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/string/relation/relation_result_2024-02-12 02:29:22.298776.txt"},
        "Llama-2-7b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/string/relation/relation_result_2024-02-12 01:53:29.110004.txt"},
        "Llama-2-70b-chat": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/string/relation/relation_result_2024-02-12 03:33:48.067093.txt"},
        "Mistral-7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/string/relation/relation_result_2024-02-12 15:18:44.331272.txt"},
        "Mixtral-8x7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/string/relation/relation_result_2024-02-19 17:17:06.938621.txt"},
        "SOLAR-10.7B-Instruct": {"result_file": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/string/relation/relation_result_2024-02-11 18:27:54.586942.txt"}
    }
}

ppi_result_dict = {}


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

                if src in ppi_result_dict:
                    ppi_result_dict[src]['pred'].append(pred)
                    ppi_result_dict[src]['models'].append(model_name)
                else:
                    ppi_result_dict[src] = {'true': true, 
                                            'pred': [pred],
                                            'models': [model_name]}

all_predicted_ppi_list = []
none_predicted_ppi_list = []

for src, value in ppi_result_dict.items():
    if len(value['models']) == len(data_result_dict["string"]):
        s_p = list(set(value['pred']))
        if len(s_p) == 1:
            if value['true'] == s_p[0]:
                all_predicted_ppi_list.append([src, s_p[0]])
            else:
                none_predicted_ppi_list.append([src, s_p[0]])

with open('all_predicted_ppi_list.txt', 'w+') as fout:
    for i in all_predicted_ppi_list:
        fout.write(', '.join(i) + '\n')

with open('none_predicted_ppi_list.txt', 'w+') as fout:
    for i in none_predicted_ppi_list:
        fout.write(', '.join(i) + '\n')

