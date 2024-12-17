import random
import pickle
import os
import torch
import itertools
import zipfile
from copy import copy
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from nnsight import NNsight

from EWOthello.data.othello import get, chess_get
from EWOthello.mingpt.model import GPTConfig, GPTforProbing, AnyGPTforProbing
from EWOthello.mingpt.dataset import CharDataset

from saes.probe_datasets import ProbeDataset, ProbeDatasetPrecomputed

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_datasets_automatic(train_size:int, test_size:int, shuffle_seed=1, game="othello") -> CharDataset:
    '''
    creates a test and train dataset of the given sizes.
    train_size and test_size must both be positive
    maximum dataset size: ~23M
    '''
    if game == "othello":
        num_datasets_to_load=(test_size+train_size)//100000 + 1
        game_data = get(ood_num=-1, data_root=None, num_preload=num_datasets_to_load) # 11 corresponds to over 1 million games
    elif game == "chess":
        game_data = chess_get(data_root=None, num_data=test_size+train_size)
    
    random.seed(shuffle_seed)
    random.shuffle(game_data.sequences)
    train_game_data, test_game_data=copy(game_data), copy(game_data)
    train_game_data.sequences=game_data.sequences[:train_size]
    test_game_data.sequences=game_data.sequences[train_size:train_size+test_size]
    return CharDataset(train_game_data, game=game), CharDataset(test_game_data, game=game)


def load_probe_datasets_automatic(train_size:int, test_size:int, shuffle_seed=1, mode="precomputed"):
    if mode == "precomputed":
        data_dir="EWOthello/data/othello_synthetic_with_board_states"
    else:
        data_dir="EWOthello/data/othello_synthetic"
    total_data_needed = round((test_size+train_size) * 1.001)
    games = []
    filenames = os.listdir(data_dir)
    print("Collecting, unzipping, and loading data files...")
    pickle_files = []
    zip_files = []
    for filename in filenames:
        if filename[-4:] == ".pkl":
            pickle_files.append(filename)
        elif filename[-4:] == ".zip":
            zip_files.append(filename)
    print("Loading previously-unzipped files...")
    finished_loading = False
    for filename in pickle_files:
        with open(f"{data_dir}/{filename}", "rb") as handle:
            g = pickle.load(handle)
            games.extend(g)
        print(f"\r{len(games)} games loaded out of {total_data_needed}", end="")
        if len(games) >= total_data_needed:
            finished_loading = True
            break
    if not finished_loading:
        print("\nUnzipping and loading zipped files...")
        for filename in zip_files:
            with zipfile.ZipFile(f"{data_dir}/{filename}","r") as zip_ref:
                zip_ref.extractall(data_dir)
            with open(f"{data_dir}/{filename[:-4]}.pkl", "rb") as handle:
                g = pickle.load(handle)
                games.extend(g)
            print(f"\r{len(games)} games loaded out of {total_data_needed}", end="")
            if len(games) >= total_data_needed:
                finished_loading = True
                break
    
    print("\nDeduplicating...")
    games.sort()
    games = [k for k, _ in itertools.groupby(games)]
    print(f"Deduplicating finished with {len(games)} games left")

    random.seed(shuffle_seed)
    random.shuffle(games)
    test_games = games[:test_size]
    if finished_loading:
        train_games = games[test_size:train_size+test_size]
    else:
        train_games = games[test_size:]

    if mode == "precomputed":
        return ProbeDatasetPrecomputed(train_games), ProbeDatasetPrecomputed(test_games)
    else:
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

othello_gpt_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
#todo: figure out how to load ChessGPT

def load_pre_trained_gpt(probe_path=None, game="othello", probe_layer:int=3):
    """
    loads the corresponding model and wires it to run through probe_layer
    """

    if game == "othello":
        if probe_path == None:
            probe_path = othello_gpt_path
        n_layer = int(probe_path[-5:-4])
        n_head = int(probe_path[-3:-2])
        mconf = GPTConfig(61, 59, n_layer=n_layer, n_head=n_head, n_embd=512)
        othello_probe = GPTforProbing(mconf, probe_layer)
        
        othello_probe.load_state_dict(torch.load(probe_path + f"GPT_Synthetic_{n_layer}Layers_{n_head}Heads.ckpt", map_location=device))
        othello_probe.eval()
        GPT_probe = AnyGPTforProbing(gpt=othello_probe, game=game)

    elif game == "chess":
        full_model = GPT2LMHeadModel.from_pretrained("adamkarvonen/RandomWeights8LayerChessGPT2").to(device)
        nnsight_model = NNsight(full_model).to(device)
        chess_probe = nnsight_model.transformer.h[probe_layer]
        GPT_probe = AnyGPTforProbing(gpt=chess_probe, game=game, output_size=512)
    return GPT_probe


@torch.inference_mode()
def vectorized_f1_score(scores:torch.tensor, labels:torch.tensor, thresholds:torch.tensor) -> torch.tensor:
    '''
    inputs:
        scores: float tensor of shape (N, K) where N is the number of items in your dataset and K is the number of input scores each has
        labels: binary tensor of shape (N)
        thresholds: float tensor of shape (T)
    returns:
        f1_scores: tensor of shape (K, T), where the (k,t)th entry is the F1 score of using score k at threshold t
    '''
    while len(scores.shape)<3:
        scores=scores.unsqueeze(-1)
    labels=labels.to(device=device, dtype=torch.float32)
    indicated_positives=(scores>=thresholds.unsqueeze(0).unsqueeze(0)).to(device=device, dtype=torch.float32) #shape (N,K,T)
    true_positives=torch.tensordot(indicated_positives,labels, dims=([0],[0]))
    false_positives=torch.tensordot(indicated_positives,(1-labels), dims=([0],[0]))
    false_negatives=torch.tensordot(1-indicated_positives,labels, dims=([0],[0]))
    f1_scores=(2*true_positives)/(2*true_positives+false_negatives+false_positives)
    masked_f1_scores=torch.where((true_positives+false_negatives+false_positives)==0, 0, f1_scores)
    return masked_f1_scores