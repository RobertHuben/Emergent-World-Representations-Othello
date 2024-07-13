import sys
import os
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))

import torch 
from tqdm import tqdm

from saes.sae_template import SAEPretrainedProbes
from saes.architectures import SAEAnthropic, Leaky_Topk_SAE, Gated_SAE
from utils import load_pre_trained_gpt, load_dataset, load_datasets_automatic
from analysis_plotter import plot_smd_auroc_distributions
from train import train_and_test_sae, test_train_params, train_probe
from probes import SAEforProbing

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
    gpts_for_probing = {layer:load_pre_trained_gpt(probe_layer=layer) for layer in range(1, 8+1)}
    probes = {layer: SAEPretrainedProbes(gpts_for_probing[layer], probe_layer=layer, window_start_trim=4, window_end_trim=4) for layer in range(1, 8+1)}
    for layer in range(1, 8+1):
        layer_dir = f"{save_dir}/layer_{layer}"
        plot_smd_auroc_distributions(probes[layer], save_dir=layer_dir)

def leaky_topk_training_sweep(k_list:list, epsilon_list:list, mode_list:list, num_features_list=[1024], layer=3):
    gpt = load_pre_trained_gpt(probe_layer=layer)
    for k in k_list:
        for epsilon in epsilon_list:
            for num_features in num_features_list:
                for mode in mode_list:
                    sae = Leaky_Topk_SAE(gpt, num_features, epsilon, k, suppression_mode=mode)
                    if num_features_list == [1024]:
                        suffix=""
                    else:
                        suffix=f"_features={num_features}"
                    sae_name = f"leaky_topk_k={k}_epsilon={epsilon}_{mode}{suffix}"
                    print(f"\nBeginning training of {sae_name}.")
                    train_and_test_sae(sae, sae_name)

def gated_training_sweep(sparsity_coeff_list:list, type_list:list, num_features_list=[1024], layer=3):
    gpt = load_pre_trained_gpt(probe_layer=layer)
    for coeff in sparsity_coeff_list:
        for type in type_list:
            for num_features in num_features_list:
                if type == "standard":
                    no_aux_loss = False
                elif type == "tied_weights_no_aux_loss":
                    no_aux_loss = True
                sae = Gated_SAE(gpt, num_features, coeff, no_aux_loss=no_aux_loss)
                if num_features_list == [1024]:
                    suffix=""
                else:
                    suffix=f"_features={num_features}"
                sae_name = f"gated_{type}_coeff={coeff}{suffix}"
                print(f"\nBeginning training of {sae_name}.")
                train_and_test_sae(sae, sae_name)

if __name__=="__main__":

    #training_dataset_sweep()
    #evaluate_pretrained_probes(save_dir="probe_evals")
    #leaky_topk_training_sweep(k_list=[75, 100], epsilon_list=[0.005], mode_list=["absolute"])
    #gated_training_sweep([60, 100, 120, 150], ["standard"])

    #sae = torch.load("trained_models/07_13_test_sae.pkl", map_location=device)
    sae = torch.load("trained_models/for_analysis/07_09_gated_tied_weights_no_aux_loss_coeff=1.5.pkl", map_location=device)
    sae_to_probe = SAEforProbing(sae)
    train_probe(sae_to_probe, "test_sae_probe")