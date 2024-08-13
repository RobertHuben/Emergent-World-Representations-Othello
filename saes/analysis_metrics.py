import torch
from utils import load_datasets_automatic
import diptest


def compute_feature_dip_scores(sae):
    '''
    given an sae, computes the dip scores for all features
    dip scores measure bimodality. A score >0.01 is somewhat bimodal, and >0.02 is "clearly" bimodal
    '''
    train_dataset, test_dataset = load_datasets_automatic(train_size=1, test_size=1000)
    residual_streams, hidden_layers, reconstructed_residual_streams=sae.catenate_outputs_on_dataset(test_dataset, batch_size=8, include_loss=False)
    hidden_layers=hidden_layers.transpose(0,2).flatten(start_dim=1)
    dip_scores=[]
    for feature_scores in hidden_layers:
        dip = diptest.dipstat(feature_scores.detach().numpy())
        dip_scores.append(dip)
    return torch.tensor(dip_scores)

@torch.inference_mode()
def calculate_mcs_between_two_saes(sae_1, sae_2, eps=1e-6):
    '''
    computes the maximum cosine similarity between all of sae_1's features and sae_2's features
    the maximum is
    '''
    sae_1_encoder=sae_1.encoder
    sae_2_encoder=sae_2.encoder
    sae_1_encoder_directions=sae_1_encoder/(sae_1_encoder.norm(dim=0)+eps)
    sae_2_encoder_directions=sae_2_encoder/(sae_2_encoder.norm(dim=0)+eps)
    cosine_similarities=sae_1_encoder_directions.T @sae_2_encoder_directions
    mcs=cosine_similarities.max(dim=0).values
    return mcs

@torch.inference_mode()
def compute_feature_frequency(sae):
    _, dataset = load_datasets_automatic(train_size=1, test_size=1000)
    _, hidden_layers, __=sae.catenate_outputs_on_dataset(dataset, include_loss=False)
    frequencies=(hidden_layers.flatten(end_dim=-2)>0).mean(dim=0, dtype=float)
    return frequencies
