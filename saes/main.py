import torch 
from tqdm import tqdm

from sae import SAEAnthropic
from utils import load_pre_trained_gpt, load_dataset

device='cuda' if torch.cuda.is_available() else 'cpu'

if __name__=="__main__":
    probe_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
    probe_layer = 3
    GPT_probe=load_pre_trained_gpt(probe_path=probe_path, probe_layer=probe_layer)

    print("Beginning training process. It may take a moment to load the datasets...")
    train_dataset = load_dataset(split_fraction=.95, use_first_half_of_split=True, entries_limit=1000000)
    test_dataset = load_dataset(split_fraction=.95, use_first_half_of_split=False, entries_limit=1000)

    sae=SAEAnthropic(gpt=GPT_probe, feature_ratio=2, sparsity_coefficient=10, window_start_trim=4, window_end_trim=4)

    print("SAE initialized, proceeding to train!")
    sae.train_model(train_dataset, test_dataset, report_every_n_steps=500)

    print("Finished SAE training, proceeding to evaluate classifiers!")
    sae.compute_all_aurocs(sae, test_dataset)
    good_classifier_count=sae.num_high_accuracy_classifiers()
    print(f"Number of high-accuracy aurocs: {good_classifier_count}")
    
    
    save_results_location= "best_aurocs.pkl"
    torch.save(sae.classifier_aurocs, save_results_location)
    print(f"Saved results to {save_results_location}")