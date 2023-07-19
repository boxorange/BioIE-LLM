import os
from datetime import datetime
import numpy as np
from typing import List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_metrics_and_save_results(
        src, 
        pred, 
        true, 
        task, 
        labels, 
        output_dir,
        batch_size,
        n_shots,
        test_sample_size,
        task_prompt,
        data_name,
        kegg_data_type,
        num_of_indra_classes,
        exec_time,
    ):
    pred = np.array(pred)
    true = np.array(true)
    
    macro_p, macro_r, macro_f, _  = precision_recall_fscore_support(true, pred, average='macro')
    micro_p, micro_r, micro_f, _  = precision_recall_fscore_support(true, pred, average='micro')
    weighted_p, weighted_r, weighted_f, _  = precision_recall_fscore_support(true, pred, average='weighted')
    
    task_output_dir = os.path.join(output_dir, task)
    
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    
    # get current date and time
    current_datetime = str(datetime.now())

    with open(os.path.join(task_output_dir, task + "_result_" + current_datetime + ".txt"), "w+") as fout:
        if data_name == "kegg":
            fout.write(f">> N-shots: {n_shots}, Batch size: {batch_size}, Test sample size: {test_sample_size}, Data type: {kegg_data_type}, Execution time: {exec_time}\n")
        elif data_name == "indra":
            fout.write(f">> N-shots: {n_shots}, Batch size: {batch_size}, Test sample size: {test_sample_size}, Number of classes: {num_of_indra_classes}, Execution time: {exec_time}\n")
        else:
            fout.write(f">> N-shots: {n_shots}, Batch size: {batch_size}, Test sample size: {test_sample_size}, Execution time: {exec_time}\n")
        
        fout.write(f">> Task prompt:\n{task_prompt}\n")
        fout.write("--------------------------------------------------------------------\n")
        
        fout.write(f">> (macro)    Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f:.4f}\n")
        fout.write(f">> (micro)    Precision: {micro_p:.4f}, Recall: {micro_r:.4f}, F1: {micro_f:.4f}\n")
        fout.write(f">> (weighted) Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f}, F1: {weighted_f:.4f}\n")
        fout.write("====================================================================\n")
        
        # store the source item in the query to be used in entity_relation task for STRING, KEGG. 04/12/2023
        if src != None:
            fout.write("Src, Pred, True:\n")
            fout.write("********************************************************************\n")
            for s, p, t in zip(src, pred.tolist(), true.tolist()):
                fout.write(s + ", " + p + ", " + t + "\n")
        else:
            fout.write("Pred, True:\n")
            fout.write("********************************************************************\n")
            for p, t in zip(pred.tolist(), true.tolist()):
                fout.write(p + ", " + t + "\n")
    
    if labels != None:
        labels = [x.lower() for x in labels]
        labels = np.array(labels)
        
        # remove values not in labels. 05/23/2023
        filtered_pred = pred[np.isin(pred, labels)]
        no_label_idx = np.argwhere(~np.isin(pred, labels)).ravel()
        filtered_true = np.delete(true, no_label_idx)
        
        cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


        if len(labels) > 3:
            dpi = 500
            xticks_rotation = 65
        else:
            dpi = 300
            xticks_rotation = "horizontal"
        
        
        cm_disp.plot(xticks_rotation=xticks_rotation)
        
        plt.tight_layout()
        cm_disp.figure_.savefig(os.path.join(task_output_dir, task + "_confusion_matrix_" + current_datetime + ".png"), dpi=dpi, pad_inches=5)
        
        # debug
        if len(no_label_idx) > 0:
            print('>> List of generated texts not in labels')
            for p, t in zip(np.take(pred, no_label_idx), np.take(true, no_label_idx)):
                print(p, t)
    

"""
from datasets import load_metric

def compute_metrics_hf(pred, true):
    # metrics ref: https://github.com/huggingface/datasets/tree/master/metrics
    
    a_m = load_metric("accuracy")
    p_m = load_metric("precision")
    r_m = load_metric("recall")
    f_m = load_metric("f1")
    
    '''
    # Remove ignored labels.
    # For ChemProt, ignore false labels. "CPR:false": "id": 0
    # For DDI, ignore false labels. "DDI-false": "id": 0
    # For TACRED, ignore no relation labels. "no_relation": "id": 0
    if any(x == dataset_name for x in ['ChemProt_BLURB', 'DDI_BLURB', 'TACRED']):
        cleaned_pred_true = [(p, t) for (p, t) in zip(pred, true) if t != 0]
        pred = [x[0] for x in cleaned_pred_true]
        true = [x[1] for x in cleaned_pred_true]

    if any(x == dataset_name for x in ["GAD_BLURB", "EU-ADR_BioBERT"]):
        average = "binary"
    else:
        average = "micro"
    '''
    
    a = a_m.compute(predictions=pred, references=true)
    p = p_m.compute(predictions=pred, references=true, average=average)
    r = r_m.compute(predictions=pred, references=true, average=average)
    f = f_m.compute(predictions=pred, references=true, average=average)
    
    return {"accuracy": a["accuracy"], "precision": p["precision"], "recall": r["recall"], "f1": f["f1"]}
"""


def _get_row(data, label):
    """
    ref: https://github.com/lavis-nlp/spert/blob/master/spert/evaluator.py

    """
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)
    

def _print_results(per_type: List, micro: List, macro: List, types: List):
    """
    ref: https://github.com/lavis-nlp/spert/blob/master/spert/evaluator.py
    
    """
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)


def compute_metrics_spert(true, pred, types, print_results: bool = False):
    """
    This is from SpERT code. https://github.com/lavis-nlp/spert/blob/master/spert/evaluator.py
    
    """
    #labels = [t.index for t in types]
    per_type = precision_recall_fscore_support(true, pred, labels=types, average=None, zero_division=0)
    micro = precision_recall_fscore_support(true, pred, labels=types, average='micro', zero_division=0)[:-1]
    macro = precision_recall_fscore_support(true, pred, labels=types, average='macro', zero_division=0)[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    return [m * 100 for m in micro + macro]


