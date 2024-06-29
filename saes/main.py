import sys
import os
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))

import torch 
from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt

from sae import SAEAnthropic, SAEPretrainedProbes
from utils import load_pre_trained_gpt, load_dataset, load_datasets_automatic

device='cuda' if torch.cuda.is_available() else 'cpu'

def training_dataset_sweep():
    GPT_probe=load_pre_trained_gpt()
    saes=[]
    _, smd_evaluation_dataset =load_datasets_automatic(train_size=1, test_size=1000)
    for train_size in [1e3, 1e4, 1e5, 1e6, 1e7]:
        train_dataset,test_dataset = load_datasets_automatic(train_size=int(train_size), test_size=1000)
        saes.append(SAEAnthropic(gpt=GPT_probe, feature_ratio=2, sparsity_coefficient=10, window_start_trim=4, window_end_trim=4))
        saes[-1].train_model(train_dataset, test_dataset, report_every_n_steps=100000)
        saes[-1].compute_all_smd(smd_evaluation_dataset)
        print(saes[-1].model_specs_to_string())

def evaluate_pretrained_probes(save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metrics = ["smd", "auroc"]
    thresholds = [2.0, 0.9]
    _, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    gpts_for_probing = {layer:load_pre_trained_gpt(probe_layer=layer) for layer in range(1, 8+1)}
    probes = {layer: SAEPretrainedProbes(gpts_for_probing[layer], probe_layer=layer, window_start_trim=4, window_end_trim=4) for layer in range(1, 8+1)}
    for layer in range(1, 8+1):
        probes[layer].compute_all_smd(test_dataset)
        probes[layer].compute_all_aurocs(test_dataset)

        layer_str = f"Layer {layer} classification metrics:\n"

        best_smds = probes[layer].classifier_smds.max(dim=0).values.flatten().detach().numpy()
        best_aurocs = probes[layer].classifier_aurocs.max(dim=0).values.flatten().detach().numpy()

        for i, array in enumerate([best_smds, best_aurocs]):
            average = round(probes[layer].average_classifier_score(metric_name=f"classifier_{metrics[i]}s"), 3)
            above_threshold = probes[layer].num_classifier_above_threshold(metric_name=f"classifier_{metrics[i]}s", threshold=thresholds[i])
            layer_str += f"Average {metrics[i]}: {average}\nNumber of classifiers with {metrics[i]} > {thresholds[i]}: {above_threshold}\n"

            bins = np.linspace(min(array), max(array), 75) # fixed number of bins

            plt.xlim([0, max(array)+0.5])

            plt.hist(array, bins=bins, alpha=0.5)
            plt.title(f'{metrics[i]} distribution')
            plt.xlabel(metrics[i])
            plt.ylabel('count')

            if save_dir:
                plt.savefig(f"{save_dir}/layer_{layer}_{metrics[i]}s_distribution.png")
                plt.close()
            else:
                plt.show()

        layer_str += "\n"
        if save_dir:
            with open(save_dir+"/classifier_metrics.txt", "a") as f:
                f.write(layer_str)
        else:
            print(layer_str)



if __name__=="__main__":

    #training_dataset_sweep()
    evaluate_pretrained_probes(save_dir="probe_evals")