import sys
import os
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))

import torch 
from tqdm import tqdm

from sae_template import SAEPretrainedProbes
from architectures import SAEAnthropic, Leaky_Topk_SAE, Gated_SAE, P_Annealing_SAE, Smoothed_L0_SAE
from utils import load_pre_trained_gpt, load_dataset, load_datasets_automatic
from analysis_plotter import plot_smd_auroc_distributions
from train import train_and_test_sae, test_train_params, train_probe, L1_Choice_Trainer
from probes import ProbeDataset, LinearProbe, L1_Sparse_Probe, Without_Topk_Sparse_Probe, Leaky_Topk_Probe, K_Annealing_Probe, Pre_Chosen_Features_Gated_Probe, L1_Gated_Probe, K_Annealing_Gated_Probe, Constant_Probe, SAEforProbing, move_list_to_state_list
from train import TrainingParams
from utils import load_probe_datasets_automatic

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

def L1_choice_probe_sweep(sae_locations:list, coeff_lists:list, train_size=500000):
    train_dataset, test_dataset = load_probe_datasets_automatic(train_size=train_size, test_size=1000)
    for n, sae_location in enumerate(sae_locations):
        sae = torch.load(sae_location, map_location=device)
        sae_name = sae_location.split('/')[-1][:-4]
        sae_to_probe = SAEforProbing(sae)
        for sparsity_coeff in coeff_lists[n]:
            probe_name = f"L1_choice_probe_coeff={sparsity_coeff}__sae={sae_name}"
            print(f"Beginning training of {probe_name}")
            trainer = L1_Choice_Trainer(sae_to_probe, probe_name, train_dataset, test_dataset, sparsity_coeff=sparsity_coeff)
            trainer.train()


if __name__=="__main__":

    #training_dataset_sweep()
    #evaluate_pretrained_probes(save_dir="probe_evals")
    #leaky_topk_training_sweep(k_list=[75, 100], epsilon_list=[0.005], mode_list=["absolute"])
    #gated_training_sweep([60, 100, 120, 150], ["standard"])

    #sae_location = "trained_models/for_analysis/07_09_gated_tied_weights_no_aux_loss_coeff=1.5.pkl"
    #sae_location = "07_09_gated_tied_weights_no_aux_loss_coeff=1.5.pkl"
    
    
    #test a constant probe
    """ sae = torch.load(sae_location, map_location=device)
    sae_to_probe = SAEforProbing(sae)
    training_params = TrainingParams(num_train_data=500000)
    probe = Constant_Probe(sae_to_probe, input_dim=1024)
    train_probe(probe, "constant_probe", train_params=training_params, eval_after=True) """

    """ layer = 3
    coeff = 1.7
    num_features = 1024
    gpt = load_pre_trained_gpt(probe_layer=layer)
    sae = SAEAnthropic(gpt, num_features, coeff)
    sae_name = f"anthropic_sae_coeff={coeff}_features={num_features}"
    print(f"\nBeginning training of {sae_name}.")
    train_and_test_sae(sae, sae_name) """
    
    """ k = 100
    sae = Leaky_Topk_SAE(gpt, num_features, epsilon=0, k=k)
    sae_name = f"topk_sae_k={100}_features={num_features}_layer={layer}"
    print(f"\nBeginning training of {sae_name}.")
    train_and_test_sae(sae, sae_name) """

    """ gpt = load_pre_trained_gpt(probe_layer=3)
    coeffs = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    anneal_proportions = [0.2, 0.4, 0.6, 0.8]
    for coeff in coeffs:
        for anneal_prop in anneal_proportions:
            sae = P_Annealing_SAE(gpt, 1024, coeff, anneal_prop)
            sae_name = f"p_anneal_coeff={coeff}_anneal={anneal_prop}"
            print(f"Beginning training of {sae_name}")
            train_and_test_sae(sae, sae_name) """
    
    gpt = load_pre_trained_gpt(probe_layer=3)
    coeffs = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    epsilon = 0.01
    deltas = [2.5, 5, 10, 20]
    for coeff in coeffs:
        for delta in deltas:
            sae = Smoothed_L0_SAE(gpt, 1024, coeff, epsilon, delta)
            sae_name = f"smoothed_L0_coeff={coeff}_delta={delta}_epsilon={epsilon}"
            print(f"Beginning training of {sae_name}")
            train_and_test_sae(sae, sae_name)

    
    """ test_train_size = 1000
    full_train_size = 500000
    sae_filenames = os.listdir("trained_models/for_analysis")
    above = [42, 48, 54]
    below = [18, 15, 12]
    coeffs = [[],[],[],[],[]]
    for n, filename in enumerate(sae_filenames):
        if "gated" in filename:
            coeffs[n].extend(below)
        else:
            coeffs[n].extend(above)
        if "top" in filename:
            coeffs[n].extend(below)
    sae_locations = [f"trained_models/for_analysis/{filename}" for filename in sae_filenames]
    L1_choice_probe_sweep(sae_locations, coeffs, train_size=full_train_size) """

    """ sae_filenames = os.listdir("trained_models/for_analysis")
    sae_locations = [f"trained_models/for_analysis/{filename}" for filename in sae_filenames]
    train_dataset, test_dataset = load_probe_datasets_automatic(500000, 1000)
    for n, sae_filename in enumerate(sae_filenames):
        sae_name = sae_filename[:-4]
        sae = torch.load(sae_locations[n], map_location=device)
        sae_to_probe = SAEforProbing(sae)
        probe = LinearProbe(sae_to_probe, input_dim=1024, layer_to_probe="hidden")
        train_probe(probe, f"linear_probe_layer=hidden_sae={sae_name}", TrainingParams(num_epochs=6), dataset_pair=(train_dataset, test_dataset)) """