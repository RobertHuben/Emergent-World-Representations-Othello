import torch
from EWOthello.data.othello import get
from EWOthello.mingpt.model import GPTConfig, GPTforProbing
from EWOthello.mingpt.dataset import CharDataset
from probes import ProbeDataset
from architectures import SAEDummy
import random
import pickle
import os
from copy import copy
from tqdm import tqdm
import itertools

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_datasets_automatic(train_size:int,test_size:int, shuffle_seed=1) -> CharDataset:
    '''
    creates a test and train dataset of the given sizes.
    train_size and test_size must both be positive
    maximum dataset size: ~23M
    '''
    num_datasets_to_load=(test_size+train_size)//100000 + 1
    othello = get(ood_num=-1, data_root=None, num_preload=num_datasets_to_load) # 11 corresponds to over 1 million games

    random.seed(shuffle_seed)
    random.shuffle(othello.sequences)
    train_othello, test_othello=copy(othello), copy(othello)
    train_othello.sequences=othello.sequences[:train_size]
    test_othello.sequences=othello.sequences[train_size:train_size+test_size]
    return CharDataset(train_othello), CharDataset(test_othello)

def load_probe_datasets_automatic(train_size:int, test_size:int, shuffle_seed=1):
    print("Loading games...")
    data_dir="EWOthello/data/othello_synthetic_with_board_states"
    num_datasets_to_load = (test_size+train_size)//100000 + 1
    games = []
    filenames = os.listdir(data_dir)
    bar = tqdm(filenames[:num_datasets_to_load])
    for filename in bar:
        with open(f"{data_dir}/{filename}", "rb") as handle:
            g = pickle.load(handle)
            games.extend(g)
    
    print("Deduplicating...")
    games.sort()
    games = [k for k, _ in itertools.groupby(games)]
    print(f"Deduplicating finished with {len(games)} games left")

    random.seed(shuffle_seed)
    random.shuffle(games)
    train_games = games[:train_size]
    test_games = games[train_size:train_size+test_size]
    return ProbeDataset(train_games), ProbeDataset(test_games)


def load_dataset(split_fraction=1, use_first_half_of_split=True, entries_limit=False, shuffle_seed=1) -> CharDataset:
    othello = get(ood_num=-1, data_root=None, num_preload=11) # 11 corresponds to over 1 million games
    random.seed(shuffle_seed)
    random.shuffle(othello.sequences)
    split_index=int(split_fraction*len(othello))
    othello.sequences=othello.sequences[:split_index] if use_first_half_of_split else othello.sequences[split_index:]
    if entries_limit and entries_limit<len(othello.sequences):
        othello.sequences=othello.sequences[:entries_limit]
    dataset=CharDataset(othello)
    return dataset

def load_pre_trained_gpt(probe_path="EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/", probe_layer:int=3):
    """
    loads the model at probe_path and wires it to run through probe_layer
    """
    n_layer = int(probe_path[-5:-4])
    n_head = int(probe_path[-3:-2])
    mconf = GPTConfig(61, 59, n_layer=n_layer, n_head=n_head, n_embd=512)
    GPT_probe = GPTforProbing(mconf, probe_layer)
    
    GPT_probe.load_state_dict(torch.load(probe_path + f"GPT_Synthetic_{n_layer}Layers_{n_head}Heads.ckpt", map_location=device))
    GPT_probe.eval()
    return GPT_probe

def find_residual_stream_mean_and_stdev():
    probe_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
    probe_layer = 3
    GPT_probe=load_pre_trained_gpt(probe_path=probe_path, probe_layer=probe_layer)
    sae=SAEDummy(gpt=GPT_probe, window_start_trim=4, window_end_trim=4)

    game_dataset = load_dataset(entries_limit=10000)
    torch.manual_seed(1)

    losses, residual_streams, hidden_layers, reconstructed_residual_streams = sae.catenate_outputs_on_dataset(game_dataset, batch_size=8)
    residual_streams=residual_streams.flatten(end_dim=-2)
    residual_stream_mean=residual_streams.mean(dim=0)
    centered_residual_streams=residual_streams-residual_stream_mean
    norms=centered_residual_streams.norm(dim=1)
    average_residual_stream_norm=norms.mean()
    
    torch.save(residual_stream_mean, "saes/model_params/residual_stream_mean.pkl")
    torch.save(average_residual_stream_norm, "saes/model_params/average_residual_stream_norm.pkl")