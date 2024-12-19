import torch

from saes.utils import load_pre_trained_gpt, load_datasets_automatic
from saes.architectures import SAEDummy

def find_residual_stream_mean_and_stdev(game="othello", probe_layer=3):
    probe_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
    GPT_probe=load_pre_trained_gpt(probe_path=probe_path, game=game, probe_layer=probe_layer)
    sae=SAEDummy(gpt=GPT_probe)

    game_dataset, _ = load_datasets_automatic(game=game, train_size=10000, test_size=1)
    torch.manual_seed(1)

    residual_streams, hidden_layers, reconstructed_residual_streams = sae.catenate_outputs_on_dataset(game_dataset, batch_size=8)
    residual_streams=residual_streams.flatten(end_dim=-2)
    residual_stream_mean=residual_streams.mean(dim=0)
    centered_residual_streams=residual_streams-residual_stream_mean
    norms=centered_residual_streams.norm(dim=1)
    average_residual_stream_norm=norms.mean()
    
    torch.save(residual_stream_mean, f"saes/model_params/{GPT_probe.game}/residual_stream_mean.pkl")
    torch.save(average_residual_stream_norm, f"saes/model_params/{GPT_probe.game}/average_residual_stream_norm.pkl")