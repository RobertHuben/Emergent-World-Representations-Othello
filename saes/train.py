import torch
from typing import Union
from sae_template import SAETemplate
from probes import LinearProbe, ProbeDataset, SAEforProbing, L1_Sparse_Probe, Pre_Chosen_Features_Gated_Probe
from EWOthello.mingpt.model import GPTforProbing
from utils import load_datasets_automatic, load_probe_datasets_automatic
from datetime import datetime
import os

class TrainingParams:

    def __init__(self, lr=5e-4, num_train_data=1000000, num_test_data=1000, num_epochs=1, report_every_n_data=50000):
        self.lr=lr
        self.num_train_data=num_train_data
        self.num_test_data=num_test_data
        self.num_epochs = num_epochs
        self.report_every_n_data=report_every_n_data

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
    sae.compute_all_smd(test_dataset)
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

def train_probe(probe:LinearProbe, save_name:str, train_params:TrainingParams=default_train_params, save_dir="trained_probes", eval_after=True):
    train_dataset, test_dataset = load_probe_datasets_automatic(train_size=train_params.num_train_data, test_size=train_params.num_test_data)
    probe.train_model(train_dataset, test_dataset, num_epochs=train_params.num_epochs, learning_rate=train_params.lr, report_every_n_data=train_params.report_every_n_data)

    date_prefix=datetime.today().strftime("%m_%d")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(probe, f"{save_dir}/{date_prefix}_{save_name}.pkl")

    if eval_after:
        probe_test_dataset = ProbeDataset(test_dataset)
        probe.after_training_eval(probe_test_dataset, f"{save_dir}/{date_prefix}_{save_name}_eval.txt")
    return probe

class L1_Choice_Trainer:
    def __init__(self, sae_to_probe:SAEforProbing, save_name:str, save_dir="trained_probes", L1_probe=None, sparsity_coeff=None, L1_training_params=TrainingParams(num_train_data=500000, num_epochs=2), choice_training_params=TrainingParams(num_train_data=500000, num_epochs=4), init_with_L1=True, bound=0.01, bound_type="absolute"):
        self.sae_to_probe = sae_to_probe
        self.save_name = save_name
        self.save_dir = save_dir
        self.choice_training_params = choice_training_params
        self.L1_training_params = L1_training_params
        self.init_with_L1 = init_with_L1
        self.bound = bound
        self.bound_type = bound_type
        if L1_probe == None:
            self.L1_probe = L1_Sparse_Probe(sae_to_probe, sparsity_coeff)
            self.L1_probe_trained = False
        else:
            self.L1_probe = L1_probe
            self.L1_probe_trained = True

    def train_L1_probe(self):
        train_dataset, test_dataset = load_datasets_automatic(train_size=self.L1_training_params.num_train_data, test_size=self.L1_training_params.num_test_data)
        self.L1_probe.train_model(train_dataset, test_dataset, learning_rate=self.L1_training_params.lr, report_every_n_data=self.L1_training_params.report_every_n_data)
        date_prefix=datetime.today().strftime("%m_%d")
        torch.save(self.L1_probe, f"{self.save_dir}/{date_prefix}_{self.save_name}_L1_probe.pkl")

        abs_weights = torch.abs(self.L1_probe.linear.weight)
        top5_features = torch.topk(abs_weights, k=5, dim=1).indices
        top5_weights = self.L1_probe.linear.weight.gather(1, top5_features)
        with open(f"{self.save_dir}/{date_prefix}_{self.save_name}_L1_probe_eval.txt", 'w') as f:
            f.write(f"\nTop 5 features by board position and class:\n{top5_features.reshape((8, 8, 3, 5))}\n")
            f.write(f"\nTop 5 weights by board position and class:\n{top5_weights.reshape((8, 8, 3, 5))}")

    def initialize_choice_probe(self):
        self.chosen_features_list = []
        if self.init_with_L1:
            initial_bias = self.L1_probe.linear.bias
            initial_weights = []
        sparse_weights = self.L1_probe.linear.weight.reshape(64, 3, -1)
        for position in range(64):
            max_abs_feature_weights = torch.abs(sparse_weights[position]).max(dim=0).values
            if self.bound_type == "absolute":
                bound = self.bound
            elif self.bound_type == "relative":
                max_abs_weight = torch.max(torch.abs(sparse_weights[position]))
                bound = max_abs_weight * self.bound
            feature_indices = torch.nonzero(max_abs_feature_weights >= bound).flatten()
            self.chosen_features_list.append(feature_indices.flatten())
            if self.init_with_L1:
                initial_weights.append(sparse_weights[position, :, feature_indices])

        if self.init_with_L1:
            self.choice_probe = Pre_Chosen_Features_Gated_Probe(self.sae_to_probe, self.chosen_features_list, initial_weights=initial_weights, initial_bias=initial_bias)
        else:
            self.choice_probe = Pre_Chosen_Features_Gated_Probe(self.sae_to_probe, self.chosen_features_list)

    def train_choice_probe(self):
        train_probe(self.choice_probe, self.save_name, self.choice_training_params, self.save_dir, eval_after=True)

    def train(self):
        if not self.L1_probe_trained:
            print("\nTraining L1 probe.\n")
            self.train_L1_probe()

        self.initialize_choice_probe()

        print("\nTraining choice probe.\n")
        self.train_choice_probe()

