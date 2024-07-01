import torch
import os
import math
import numpy as np
from matplotlib import pyplot as plt

from utils import load_datasets_automatic


def plot_smd_auroc_distributions(sae, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    _, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)

    sae.compute_all_smd(test_dataset)
    sae.compute_all_aurocs(test_dataset)

    best_smds = sae.classifier_smds.max(dim=0).values.flatten().detach().numpy()
    best_aurocs = sae.classifier_aurocs.max(dim=0).values.flatten().detach().numpy()

    output_str = "Classification metrics:\n"

    metrics = ["smd", "auroc"]
    thresholds = [2.0, 0.9]
    for i, array in enumerate([best_smds, best_aurocs]):
        average = round(sae.average_classifier_score(metric_name=f"classifier_{metrics[i]}s"), 3)
        above_threshold = sae.num_classifier_above_threshold(metric_name=f"classifier_{metrics[i]}s", threshold=thresholds[i])
        output_str += f"Average {metrics[i]}: {average}\nNumber of classifiers with {metrics[i]} > {thresholds[i]}: {above_threshold}\n"

        bins = np.linspace(min(array), max(array), 75) # fixed number of bins

        plt.xlim([0, max(array)+0.5])

        plt.hist(array, bins=bins, alpha=0.5)
        plt.title(f'{metrics[i]} distribution')
        plt.xlabel(metrics[i])
        plt.ylabel('count')

        if save_dir:
            plt.savefig(f"{save_dir}/{metrics[i]}s_distribution.png")
            plt.close()
        else:
            plt.show()

    output_str += "\n"
    if save_dir:
        with open(save_dir+"/classifier_metrics.txt", "a") as f:
            f.write(output_str)
    else:
        print(output_str)

