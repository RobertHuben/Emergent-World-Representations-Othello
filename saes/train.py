import os
import torch
from typing import Union
from datetime import datetime

from EWOthello.mingpt.model import GPTforProbing

from saes.sae_template import SAETemplate
from saes.probes import LinearProbe, ProbeDataset, SAEforProbing, L1_Sparse_Probe, Pre_Chosen_Features_Gated_Probe
from saes.utils import load_datasets_automatic, load_probe_datasets_automatic
from saes.sae_template import SAETemplate
from saes.probes import LinearProbe, ProbeDataset
from saes.utils import load_datasets_automatic

class TrainingParams:

    def __init__(self, lr=5e-4, num_train_data=1000000, num_test_data=1000, num_epochs=1, report_every_n_data=50000, compute_smd=True, compute_aurocs=False, compute_f1_scores=True):
        self.lr=lr
        self.num_train_data=num_train_data
        self.num_test_data=num_test_data
        self.num_epochs = num_epochs
        self.report_every_n_data=report_every_n_data
        self.compute_smd=compute_smd
        self.compute_aurocs=compute_aurocs
        self.compute_f1_scores=compute_f1_scores

default_train_params=TrainingParams()
test_train_params=TrainingParams(num_test_data=100, report_every_n_data=500)

def train_and_test_sae(sae:SAETemplate, save_name:str, train_params:TrainingParams=default_train_params, save_dir="trained_models", print_results=True):
    '''
    Does a single standardized experiment where you train and evaluate the given sae
    Inputs:
        sae - an untrained sparse autoencoder archecture
        save_name - the place to write the reports to, should be a string without a suffix
        train_params - a TrainingParams object
        print_results - whether you want results to be printed to the command line after the training
    Outputs:
        Generates a file at MM_DD_save_name.txt and writes model_specs_to_string() to it, after all training
        Saves the trained model to MM_DD_save_name.pkl
        And returns the trained model
    todo: make the default training params argument the generic training params object
    '''
    train_dataset, test_dataset = load_datasets_automatic(train_size=train_params.num_train_data, test_size=train_params.num_test_data)
    sae.train_model(train_dataset, test_dataset, learning_rate=train_params.lr, report_every_n_data=train_params.report_every_n_data)
    if train_params.compute_smd:
        sae.compute_all_smd(test_dataset)
    if train_params.compute_aurocs:
        sae.compute_all_aurocs(test_dataset)
    if train_params.compute_f1_scores:
        sae.compute_all_f1_vectorized(test_dataset)
    this_message=sae.model_specs_to_string(test_dataset)
    if print_results:
        print(this_message)
    date_prefix=datetime.today().strftime("%m_%d")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f"{save_dir}/{date_prefix}_{save_name}.txt", 'w') as f:
        print(this_message, file=f)
    torch.save(sae, f"{save_dir}/{date_prefix}_{save_name}.pkl")
    return sae

def train_probe(probe:LinearProbe, save_name:str, train_params:TrainingParams=default_train_params, dataset_pair=None, save_dir="trained_probes", eval_after=True):
    if dataset_pair:
        train_dataset, test_dataset = dataset_pair
    else:
        train_dataset, test_dataset = load_probe_datasets_automatic(train_size=train_params.num_train_data, test_size=train_params.num_test_data)
    probe.train_model(train_dataset, test_dataset, num_epochs=train_params.num_epochs, learning_rate=train_params.lr, report_every_n_data=train_params.report_every_n_data)

    date_prefix=datetime.today().strftime("%m_%d")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(probe, f"{save_dir}/{date_prefix}_{save_name}.pkl")

    if eval_after:
        probe.after_training_eval(test_dataset, f"{save_dir}/{date_prefix}_{save_name}_eval.txt")
    return probe

class L1_Choice_Trainer:
    def __init__(self, sae_to_probe:SAEforProbing, save_name:str, train_dataset, test_dataset, save_dir="trained_probes", L1_probe=None, sparsity_coeff=None, num_L1_epochs=2, num_choice_epochs=4, bound=0.01, bound_type="absolute"):
        self.sae_to_probe = sae_to_probe
        self.save_name = save_name
        self.save_dir = save_dir
        self.choice_epochs = num_choice_epochs
        self.L1_epochs = num_L1_epochs
        self.bound = bound
        self.bound_type = bound_type
        if L1_probe == None:
            self.L1_probe = L1_Sparse_Probe(sae_to_probe, sparsity_coeff)
            self.L1_probe_trained = False
        else:
            self.L1_probe = L1_probe
            self.L1_probe_trained = True

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_L1_probe(self):
        train_probe(self.L1_probe, f"{self.save_name}_L1_probe", TrainingParams(num_epochs=self.L1_epochs), (self.train_dataset, self.test_dataset), self.save_dir, eval_after=False)

        abs_weights = torch.abs(self.L1_probe.linear.weight)
        top5_features = torch.topk(abs_weights, k=5, dim=1).indices
        top5_weights = self.L1_probe.linear.weight.gather(1, top5_features)
        date_prefix=datetime.today().strftime("%m_%d")
        with open(f"{self.save_dir}/{date_prefix}_{self.save_name}_L1_probe_eval.txt", 'w') as f:
            f.write(f"\nTop 5 features by board position and class:\n{top5_features.reshape((8, 8, 3, 5))}\n")
            f.write(f"\nTop 5 weights by board position and class:\n{top5_weights.reshape((8, 8, 3, 5))}")

    def initialize_choice_probe(self):
        self.chosen_features_list = []
        initial_bias = self.L1_probe.linear.bias
        initial_weights = []
        sparse_weights = self.L1_probe.linear.weight.reshape(64, 3, -1)
        num_features_chosen = 0
        for position in range(64):
            max_abs_feature_weights = torch.abs(sparse_weights[position]).max(dim=0).values
            if self.bound_type == "absolute":
                bound = self.bound
            elif self.bound_type == "relative":
                max_abs_weight = torch.max(torch.abs(sparse_weights[position]))
                bound = max_abs_weight * self.bound
            feature_indices = torch.nonzero(max_abs_feature_weights >= bound).flatten()
            self.chosen_features_list.append(feature_indices.flatten())
            initial_weights.append(sparse_weights[position, :, feature_indices])
            num_features_chosen += feature_indices.numel()

        print(f"Average number of features chosen per position: {num_features_chosen/64}")
        self.choice_probe = Pre_Chosen_Features_Gated_Probe(self.sae_to_probe, self.chosen_features_list, initial_weights=initial_weights, initial_bias=initial_bias)
        
    def train_choice_probe(self):
        train_probe(self.choice_probe, f"{self.save_name}_choice_probe", TrainingParams(num_epochs=self.choice_epochs), (self.train_dataset, self.test_dataset), self.save_dir)

    def train(self):
        if not self.L1_probe_trained:
            print("\nTraining L1 probe.\n")
            self.train_L1_probe()

        self.initialize_choice_probe()

        print("\nTraining choice probe.\n")
        self.train_choice_probe()

