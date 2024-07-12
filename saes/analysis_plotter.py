import torch
import os
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import diptest
import re

from utils import load_datasets_automatic
from board_states import get_board_states

device='cuda' if torch.cuda.is_available() else 'cpu'

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


def plot_accuracies(sae_location, save_location=None):
    with open(sae_location, 'rb') as f:
        sae = torch.load(f, map_location=device)
    class_names=["Enemy", "Blank", "Own"]
    fig, axes = plt.subplots(1,3, constrained_layout=True)

    for i in range(3):
        data=sae.classifier_smds[:,:,i].max(dim=0).values
        data=data.reshape((8,8))
        im=axes[i].imshow(data, vmin=0, vmax=2)
        axes[i].set_title(f"SMD of {class_names[i]}")
    
    fig.suptitle("SMD of classification by target class")
    fig.colorbar(im)
    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()

def tricolor_plot(sae):
    x=sae.classifier_smds.max(dim=0).values.reshape((8,8,3))
    x=torch.clamp(x, min=0, max=2)/2
    plt.imshow(x)
    plt.title("All classification accuracies (white is SMD=2)\nRed=Enemy, Blue=Empty, Green=Own")
    return

def plot_feature_activations(sae, feature_number, board_position, dataset, separate_at_0=True):
    residual_streams, hidden_layers, reconstructed_residual_streams=sae.catenate_outputs_on_dataset(dataset, batch_size=8, include_loss=False)
    scores=hidden_layers[:,:, feature_number].flatten().detach().numpy()
    board_states= get_board_states(dataset)
    board_states=sae.trim_to_window(board_states)
    labels=board_states[:,:, board_position].flatten()
    class_names=["enemy", "empty", "own"]
    enemy_empty_own_scores=[scores[np.where(labels==class_number)] for class_number in range(3)]

    if separate_at_0:
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 20]})
        ax=axs[1]
        axs[0].set_xticks([0],[0])
        for spine in axs[0].spines.values():
            spine.set_visible(False)
        enemy_empty_own_zero_scores=[class_scores[np.where(class_scores==0)] for class_scores in enemy_empty_own_scores]
        enemy_empty_own_zero_frequencies=[len(class_scores)/len(scores) for class_scores in enemy_empty_own_zero_scores]
        enemy_empty_own_nonzero_scores=[class_scores[np.where(class_scores!=0)] for class_scores in enemy_empty_own_scores]
        bottom=0
        axs[0].set_ylabel("Frequency")
        ax.set_ylabel(" ")

        for class_number, class_name in enumerate(class_names):
            sns.kdeplot(enemy_empty_own_nonzero_scores[class_number], fill=True, label=class_name, ax=axs[1])
            axs[0].bar(x=0, bottom=bottom, height=enemy_empty_own_zero_frequencies[class_number])
            bottom+=enemy_empty_own_zero_frequencies[class_number]
    else:
        fig, ax=plt.subplots()
        ax.set_ylabel("Frequency")
        sns.kdeplot(enemy_empty_own_scores, fill=True, label=class_names)

    ax.set_xlabel("Feature Activation")
    plt.title(f'Feature {feature_number} activations against Position {board_position} contents')
    plt.legend()
    save_file_name=f"analysis_results/hist_contents_feat_{feature_number}_pos_{board_position}.png"
    fig.savefig(save_file_name)

def show_best_feature(sae, position_index, piece_class):
    feature_to_use=sae.classifier_smds.max(dim=0).indices[position_index][piece_class]
    _, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    plot_feature_activations(sae, feature_to_use, position_index, test_dataset)

def compute_feature_dip_scores(sae):
    '''
    given an sae, computes the dip scores for all features
    dip scores measure bimodality. A score >0.01 is somewhat bimodal, and >0.02 is "clearly" bimodal
    '''
    train_dataset, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    residual_streams, hidden_layers, reconstructed_residual_streams=sae.catenate_outputs_on_dataset(test_dataset, batch_size=8, include_loss=False)
    hidden_layers=hidden_layers.transpose(0,2).flatten(start_dim=1)
    dip_scores=[]
    for feature_scores in hidden_layers:
        dip = diptest.dipstat(feature_scores.detach().numpy())
        dip_scores.append(dip)
    return torch.tensor(dip_scores)

def plot_many_saes(prefix, dir="trained_models"):
    suffix=".txt"
    smd_expression='Number of SMD>2'
    sparsity_expression='k (sparsity)'
    num_features_expression='Number of features'

    names=[]
    smds=[]
    sparsities=[]
    num_features_list=[]

    for file in os.listdir(dir):
        if not file.startswith(prefix) or not file.endswith(suffix):
            continue
        contents=open(f"{dir}/{file}", 'r').read()
        smd_line=[line for line in contents.split("\n") if line.strip().startswith(smd_expression)]
        if smd_line:
            smd=float(smd_line[0].split(": ")[1])
        sparsity_line=[line for line in contents.split("\n") if line.strip().startswith(sparsity_expression)]
        if sparsity_line:
            sparsity=int(sparsity_line[0].split(": ")[1])
        num_features_line=[line for line in contents.split("\n") if line.strip().startswith(num_features_expression)]
        if num_features_line:
            num_features=int(num_features_line[0].split(": ")[1])

        names.append(file)
        smds.append(smd)
        sparsities.append(sparsity)
        num_features_list.append(num_features)

    unique_sparsities=sorted(list(set(sparsities)))
    unique_num_features=sorted(list(set(num_features_list)))
    data=np.zeros((len(unique_sparsities), len(unique_num_features)))
    for smd, sparsity, num_feat in zip(smds, sparsities, num_features_list):
        data[unique_sparsities.index(sparsity), unique_num_features.index(num_feat)]=smd
    bar_width=.5/len(unique_sparsities)
    for i, sparsity in enumerate(unique_sparsities):
        plt.bar(np.arange(len(unique_num_features))+bar_width*i, data[i], bar_width, label=sparsity)
    plt.xticks(ticks=range(len(unique_num_features)), labels=unique_num_features)
    plt.xlabel("Num Features")
    plt.savefig("Comparison.jpg")
    return

if __name__=="__main__":
    pass