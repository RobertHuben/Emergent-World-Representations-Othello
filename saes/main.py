import torch 
from tqdm import tqdm

from sae import SAEAnthropic
from utils import load_pre_trained_gpt, load_dataset

device='cuda' if torch.cuda.is_available() else 'cpu'

def training_dataset_sweep():
    for entries_limit in [1e3, 1e4,1e5,1e6]:
        GPT_probe=load_pre_trained_gpt()
        print("Beginning training process. It may take a moment to load the datasets...")
        train_dataset = load_dataset(split_fraction=.95, use_first_half_of_split=True, entries_limit=int(entries_limit))
        test_dataset = load_dataset(split_fraction=.95, use_first_half_of_split=False, entries_limit=1000)

        sae=SAEAnthropic(gpt=GPT_probe, feature_ratio=2, sparsity_coefficient=10, window_start_trim=4, window_end_trim=4)

        print("SAE initialized, proceeding to train!")
        sae.train_model(train_dataset, test_dataset, report_every_n_steps=500)

        print("Finished SAE training, proceeding to evaluate classifiers!")
        sae.compute_all_smd(sae, test_dataset)
        print(sae.model_specs_to_string())

if __name__=="__main__":
    training_dataset_sweep()