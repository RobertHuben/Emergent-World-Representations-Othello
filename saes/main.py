import sys
import os
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))#"C:/Users/andre/Code/othello_gpt_saes/Emergent-World-Representations-Othello")

import torch 
from tqdm import tqdm

from sae import SAEAnthropic
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

def evaluate_pretrained_probes():
    gpt = load_pre_trained_gpt()
    sae = SAEPretrainedProbes(gpt, probe_layer=3, window_start_trim=4, window_end_trim=4)
    test_dataset = load_dataset(split_fraction=.95, use_first_half_of_split=False, entries_limit=1000)
    #sae.compute_all_aurocs(test_dataset)
    sae.compute_all_smd(test_dataset)
    print(sae.model_specs_to_string())


if __name__=="__main__":

    #training_dataset_sweep()
    evaluate_pretrained_probes()