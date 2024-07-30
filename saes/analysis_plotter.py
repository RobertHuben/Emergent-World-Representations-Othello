import torch
import os
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils import load_datasets_automatic
from board_states import get_board_states
from probes import Gated_Probe

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

#Creates a folder of 64 plots (one per board position) at save_location
def plot_top_2_features_per_board_position(probe_location: str, save_location: str):
    dataset, _ = load_datasets_automatic(1000, 1)
    probe = torch.load(probe_location, map_location=device)
    probe_name = probe_location.split("/")[-1][:-4]
    save_dir = f"{save_location}/{probe_name}_top_2_feature_plots"
    os.makedirs(save_dir, exist_ok=True)
    residual_streams, hidden_layers, reconstructed_residual_streams=probe.model_to_probe.sae.catenate_outputs_on_dataset(dataset, batch_size=8, include_loss=False)
    top_2_features = torch.topk(probe.feature_choice, k=2, dim=-1).indices
    board_states= get_board_states(dataset)
    board_states=probe.model_to_probe.sae.trim_to_window(board_states)
    class_names=np.array(["Enemy", "Empty", "Own"])
    for board_position in range(64):
        first_feature_index = top_2_features[board_position, 0]
        second_feature_index = top_2_features[board_position, 1]
        first_feature_activations = hidden_layers[:,:, first_feature_index].flatten().detach().numpy()
        second_feature_activations = hidden_layers[:,:, second_feature_index].flatten().detach().numpy()
        labels=board_states[:,:, board_position].flatten()

        scatter = plt.scatter(x=first_feature_activations, y=second_feature_activations, s=5, c=labels)
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        plt.legend(handles, class_names, loc="upper right", title=f"Position {board_position} contents")
        plt.xlabel(f"First Feature (#{first_feature_index}) Activation")
        plt.ylabel(f"Second Feature (#{second_feature_index}) Activation")
        plt.title(f'Activations of top 2 features for predicting board position {board_position} contents')
        
        save_file_name=f"position_{board_position}.png"
        plt.savefig(f"{save_dir}/{save_file_name}")
        plt.close()

def show_best_feature(sae, position_index, piece_class):
    feature_to_use=sae.classifier_smds.max(dim=0).indices[position_index][piece_class]
    _, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    plot_feature_activations(sae, feature_to_use, position_index, test_dataset)

if __name__=="__main__":
    sae=torch.load("trained_models/top_k_sae_k_is_100.pkl", map_location=device)
    show_best_feature(sae, position_index=1, piece_class=0)

