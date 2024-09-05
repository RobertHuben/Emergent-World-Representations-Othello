import torch

from saes.utils import load_pre_trained_gpt, load_dataset
from saes.architectures import SAEDummy

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