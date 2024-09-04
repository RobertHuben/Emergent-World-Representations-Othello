import torch
import os
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import diptest
import re

from saes.utils import load_datasets_automatic
from saes.analysis_metrics import compute_feature_frequency
from saes.board_states import get_board_states
from saes.board_states import get_board_states
from saes.probes import Gated_Probe

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

def plot_top_3_features_per_board_position(probe_location: str, board_position: int):
    dataset, _ = load_datasets_automatic(104, 1)
    probe = torch.load(probe_location, map_location=device)
    residual_streams, hidden_layers, reconstructed_residual_streams=probe.model_to_probe.sae.catenate_outputs_on_dataset(dataset, batch_size=8, include_loss=False)
    top_3_features = torch.topk(probe.feature_choice, k=3, dim=-1).indices
    board_states= get_board_states(dataset)
    board_states=probe.model_to_probe.sae.trim_to_window(board_states)
    class_names=np.array(["Enemy", "Empty", "Own"])

    first_feature_index = top_3_features[board_position, 0]
    second_feature_index = top_3_features[board_position, 1]
    third_feature_index = top_3_features[board_position, 2]
    first_feature_activations = hidden_layers[:,:, first_feature_index].flatten().detach().numpy()
    second_feature_activations = hidden_layers[:,:, second_feature_index].flatten().detach().numpy()
    third_feature_activations = hidden_layers[:,:, third_feature_index].flatten().detach().numpy()
    labels=board_states[:,:, board_position].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(xs=first_feature_activations, ys=second_feature_activations, zs=third_feature_activations, s=5, c=labels)
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    ax.legend(handles, class_names, loc="upper right", title=f"Position {board_position} contents")
    ax.set_xlabel(f"First Feature (#{first_feature_index}) Activation")
    ax.set_ylabel(f"Second Feature (#{second_feature_index}) Activation")
    ax.set_zlabel(f"Third Feature (#{third_feature_index}) Activation")
    ax.set_title(f'Activations of top 3 features for predicting board position {board_position} contents')
    
    plt.show()

def show_best_feature(sae, position_index, piece_class):
    feature_to_use=sae.classifier_smds.max(dim=0).indices[position_index][piece_class]
    _, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    plot_feature_activations(sae, feature_to_use, position_index, test_dataset)

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

def plot_frequencies_vs_classifier_quality(sae):
    frequencies=compute_feature_frequency(sae)
    frequencies=torch.sort(frequencies).values

    good_classifier_indices=torch.nonzero(sae.classifier_smds>2)[:,0].tolist()

    plt.scatter(range(len(frequencies)), torch.sort(frequencies).values)
    plt.scatter(good_classifier_indices, torch.sort(frequencies).values[good_classifier_indices])
    return

def board_state_frequency_vs_smd(sae, include_aurocs=False, sae_title=None):
    smds=sae.classifier_smds.max(dim=0).values
    board_state_frequencies=torch.zeros(smds.shape)
    _, dataset = load_datasets_automatic(train_size=1, test_size=1000)
    _, hidden_layers, __=sae.catenate_outputs_on_dataset(dataset, include_loss=False)

    board_states= get_board_states(dataset)
    board_states=sae.trim_to_window(board_states)
    hidden_layers=hidden_layers.flatten(end_dim=-2)
    board_states=board_states.flatten(end_dim=-2)
    game_not_ended_mask=board_states[:,0]>-100
    hidden_layers=hidden_layers[game_not_ended_mask]
    board_states=board_states[game_not_ended_mask]
    
    for piece_class in range(3):
        board_state_frequencies[:, piece_class]=(board_states==piece_class).mean(dim=0, dtype=float)
    if include_aurocs:
        sae.compute_all_aurocs(dataset)
        best_aurocs=sae.classifier_aurocs.max(dim=0).values
        plt.scatter(board_state_frequencies, best_aurocs)
    else:
        plt.scatter(board_state_frequencies, smds)
    plt.xlabel("Board State Feature Frequency")
    plt.ylabel("Best SMD")
    plot_title="Frequency vs SMD for 64x3 board states and piece classes"
    save_location="analysis_results/frequency_vs_smd"
    if sae_title:
        reduced_sae_title=sae_title.split("/")[-1].split(".")[0]
        plot_title+=f"\nSAE: {reduced_sae_title}"
        save_location+="_"+reduced_sae_title
    plt.title(plot_title)
    plt.savefig(f"{save_location}.png")
    plt.close()

def plot_features_used_vs_accuracy(data:dict, save_name=None):
    fig, (main, full_features) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [15, 1]})
    for sae_type, data_list in data.items():
        data_list.sort()
        features = [datum[0] for datum in data_list]
        accuracies = [datum[1] for datum in data_list]
        """ outlier_num_features = []
        outlier_accuracies = []
        for i, num_features in enumerate(features):
            if num_features > 10:
                outlier_num_features.append(num_features)
                features.pop(i)
                outlier_accuracies.append(accuracies.pop(i))
        full_features.scatter(outlier_num_features, outlier_accuracies, label=sae_type) """
        main.plot(features, accuracies, "-o", label=sae_type)
        full_features.plot(features, accuracies, "-o", label=sae_type)
    main.spines.right.set_visible(False)
    full_features.spines.left.set_visible(False)
    main.set(xlabel="Number of features used", ylabel="Probe Accuracy")
    main.set_xlim(1.5, 5.2)
    plt.xticks(np.arange(1024, 1025, step=1))
    full_features.set_xlim(1023.85, 1024.15)
    full_features.tick_params(left=False)
    fig.suptitle("Accuracy of L1-choice linear probes trained on various SAEs")
    main.legend(title="SAE Types")
    if save_name:
        plt.savefig(f"analysis_results/{save_name}.png")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

if __name__=="__main__":    
    sae_locations=[ 
                    # 'trained_models/07_10_top_k_sae_1024_features_100_sparsity.pkl.pkl',
                    # 'trained_models/07_10_top_k_sae_1024_features_150_sparsity.pkl.pkl',
                    # 'trained_models/07_11_top_k_sae_1024_features_200_sparsity.pkl.pkl',
                    # 'trained_models/07_12_saeAnthropic_1024_features_4_sparsity.pkl',
                    'trained_models/07_30_saeAnthropic_layer_3_1024_features_1_sparsity.pkl',
                    'trained_models/07_30_saeAnthropic_layer_4_1024_features_1_sparsity.pkl',
                    'trained_models/07_30_saeAnthropic_layer_5_1024_features_1_sparsity.pkl',
    ]
    for sae_location in sae_locations:
        sae=torch.load(sae_location, map_location=device)
        board_state_frequency_vs_smd(sae, include_aurocs=False, sae_title=sae_location)
    plt.show()