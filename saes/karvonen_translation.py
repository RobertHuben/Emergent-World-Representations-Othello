import json
import torch
from sae_template import SAETemplate
from EWOthello.mingpt.model import AnyGPTforProbing
from saes.utils import load_datasets_automatic, load_pre_trained_gpt
from dictionary import GatedAutoEncoder

device='cuda' if torch.cuda.is_available() else 'cpu'

def test_karvonen_sae_coverage(autoencoder_path):
    layer = 5 #un-hardcode this later
    gpt = load_pre_trained_gpt(probe_layer=layer)
    sae = KarvonenSAE(gpt, num_features=512, autoencoder_path=autoencoder_path)
    sae.to(device)

    train_dataset, test_dataset = load_datasets_automatic(train_size=1, test_size=1000, game=sae.gpt.game)
    sae.compute_all_f1_vectorized(test_dataset)
    cov = sae.compute_coverage()
    print(f"Coverage: {cov}") 

class KarvonenSAE(SAETemplate):
    def __init__(self, gpt:AnyGPTforProbing, num_features:int, autoencoder_path:str, window_start_trim=0, window_end_trim=0):
        super().__init__(gpt=gpt, num_features=num_features, window_start_trim=window_start_trim, window_end_trim=window_end_trim)
        self.sae = get_ae(autoencoder_path)

    def forward(self, residual_stream, compute_loss=False):
        reconstructed_residual_stream, hidden_layer = self.sae.forward(residual_stream, output_features=True)
        loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

def get_ae(autoencoder_path):
    autoencoder_model_path = f"{autoencoder_path}/ae.pt"
    autoencoder_config_path = f"{autoencoder_path}/config.json"

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    config_args = []
    for k, v in config["trainer"].items():
        if k not in ["trainer_class", "sparsity_penalty"]:
            if not (config["trainer"]["trainer_class"] == "TrainerTopK" and k == "lr"):
                if isinstance(v, str) and k != "dict_class":
                    config_args.append(k + "=" + "'" + v + "'")
                else:
                    config_args.append(k + "=" + str(v))
    config_str = ", ".join(config_args)

    # rangell: this is a super hacky way to get the correct dictionary class from the config
    #ae_class = eval(config["trainer"]["trainer_class"] + f"({config_str})").ae.__class__
    ae_class = GatedAutoEncoder
    if "k" in config["trainer"]:
        ae = ae_class.from_pretrained(
            autoencoder_model_path, k=config["trainer"]["k"], device=device
        )
    else:
        ae = ae_class.from_pretrained(autoencoder_model_path, device=device)
    ae = ae.to(device)

    return ae