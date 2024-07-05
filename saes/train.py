import torch
from sae_template import SAETemplate
from utils import load_datasets_automatic
from datetime import datetime


class TrainingParams:

    def __init__(self, lr=5e-4, num_train_data=1000000, num_test_data=1000, report_every_n_data=50000):
        self.lr=lr
        self.num_train_data=num_train_data
        self.num_test_data=num_test_data
        self.report_every_n_data=report_every_n_data

default_train_params=TrainingParams()

def train_and_test_sae(sae:SAETemplate, save_name:str, train_params:TrainingParams=default_train_params, print_results=True):
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
    with open(f"{date_prefix}_{save_name}.txt", 'w') as f:
        print(this_message, file=f)
    torch.save(sae, f"{date_prefix}_{save_name}.pkl")
    return sae


