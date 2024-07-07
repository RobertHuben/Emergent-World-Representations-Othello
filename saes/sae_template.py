import torch 
from torch.nn import functional as F
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from EWOthello.mingpt.probe_model import BatteryProbeClassification
from EWOthello.mingpt.dataset import CharDataset
from board_states import get_board_states

logger = logging.getLogger(__name__)
device='cuda' if torch.cuda.is_available() else 'cpu'

class SAETemplate(torch.nn.Module, ABC):
    '''
    abstract base class that defines the SAE contract
    '''

    def __init__(self, gpt:GPTforProbing, num_features:int, window_start_trim:int=4, window_end_trim:int=8):
        super().__init__()
        self.gpt=gpt
        self.num_features=num_features
        for param in self.gpt.parameters():
            #freezes the gpt model  
            param.requires_grad=False 
        self.window_start_trim=window_start_trim
        self.window_end_trim=window_end_trim
        self.num_data_trained_on=0
        self.classifier_aurocs=None
        self.classifier_smds=None
        try:
            self.residual_stream_mean=torch.load("saes/model_params/residual_stream_mean.pkl", map_location=device)
            self.average_residual_stream_norm=torch.load("saes/model_params/average_residual_stream_norm.pkl", map_location=device)
        except:
            self.residual_stream_mean=torch.zeros((1))
            self.average_residual_stream_norm=torch.ones((1))
            logger.warning("Please ensure the correct files are in saes/model_params/residual_stream_mean.pkl and saes/model_params/average_residual_stream_norm.pkl!")

    def trim_to_window(self, input, offset=0):
        '''
        trims the tensor input from shape (n_batch, l_window, d_model) to (n_batch, l_window - window_start_trim - window_end_trim, d_model)'''
        window_length=input.shape[1]
        return input[:, (self.window_start_trim+offset):(window_length-self.window_end_trim+offset+1), :]

    def forward_on_tokens(self, token_sequences, compute_loss=False):
        '''
        runs the SAE on a token sequence

        in particular:
            1. takes the intermediate layer of the gpt model on this token sequence
            2. trims it to the right part of the context window
            3. Normalizes it by subtracting the model mean and dividing by the scale factor
            4. Runs the SAE on that residual stream
        '''
        raw_residual_stream=self.gpt(token_sequences)
        trimmed_residual_stream=self.trim_to_window(raw_residual_stream)
        normalized_residual_stream=(trimmed_residual_stream-self.residual_stream_mean)/self.average_residual_stream_norm
        loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.forward(normalized_residual_stream, compute_loss=compute_loss)
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def catenate_outputs_on_dataset(self, dataset:CharDataset, batch_size=8, include_loss=False):
        '''
        runs the model on the entire dataset, one batch at a time, catenating the outputs
        '''
        losses=[]
        residual_streams=[]
        hidden_layers=[]
        reconstructed_residual_streams=[]
        test_dataloader=iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for test_input, test_labels in test_dataloader:
            test_input=test_input.to(device)
            loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.forward_on_tokens(test_input, compute_loss=include_loss)
            if include_loss:
                losses.append(loss)
            residual_streams.append(residual_stream)
            hidden_layers.append(hidden_layer)
            reconstructed_residual_streams.append(reconstructed_residual_stream)
        residual_streams=torch.cat(residual_streams)
        hidden_layers=torch.cat(hidden_layers)
        reconstructed_residual_streams=torch.cat(reconstructed_residual_streams)
        if include_loss:
            losses=torch.stack(losses)
            return losses, residual_streams, hidden_layers, reconstructed_residual_streams
        else:
            return residual_streams, hidden_layers, reconstructed_residual_streams

    def print_evaluation(self, train_loss, eval_dataset:CharDataset, step_number="N/A"):
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset, include_loss=True)
        test_loss=losses.mean()
        l0_sparsity=self.compute_l0_sparsity(hidden_layers)
        dead_features=self.count_dead_features(hidden_layers)
        print_message=f"Train loss, test loss, l0 sparsity, dead features after {self.num_data_trained_on} training games: {train_loss.item():.2f}, {test_loss:.2f}, {l0_sparsity:.1f}, {dead_features:.0f}"
        tqdm.write(print_message)

    def compute_l0_sparsity(self, hidden_layers):
        active_features=hidden_layers>0
        sparsity_per_entry=active_features.sum()/hidden_layers[..., 0].numel()
        return sparsity_per_entry

    def count_dead_features(self, hidden_layers):
        active_features=hidden_layers>0
        dead_features=torch.all(torch.flatten(active_features, end_dim=-2), dim=0)
        num_dead_features=dead_features.sum()
        return num_dead_features

    def reconstruction_error(self, residual_stream, reconstructed_residual_stream):
        reconstruction_l2=torch.norm(reconstructed_residual_stream-residual_stream, dim=-1)
        reconstruction_loss=(reconstruction_l2**2).mean()
        return reconstruction_loss

    def train_model(self, train_dataset:CharDataset, eval_dataset:CharDataset, batch_size=64, num_epochs=1, report_every_n_data=500, learning_rate=1e-3, fixed_seed=1337):
        '''
        performs a training loop on self, with printed evaluations
        '''
        if fixed_seed:
            torch.manual_seed(fixed_seed)
        self.to(device)
        self.train()
        optimizer=torch.optim.AdamW(self.parameters(), lr=learning_rate)
        step=0
        report_on_batch_number=report_every_n_data//batch_size

        self.training_prep(train_dataset=train_dataset, eval_dataset=eval_dataset, batch_size=batch_size, num_epochs=num_epochs)

        print(f"Beginning model training on {device}!")

        for epoch in range(num_epochs):
            train_dataloader=iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
            print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_data} games.")
            
            for input_batch, label_batch in tqdm(train_dataloader):
                input_batch=input_batch.to(device)
                step+=1
                self.num_data_trained_on+=len(input_batch)
                optimizer.zero_grad(set_to_none=True)
                loss, residual_stream, hidden_layer, reconstructed_residual_stream= self.forward_on_tokens(input_batch, compute_loss=True)
                loss.backward()
                optimizer.step()

                self.after_step_update(hidden_layer=hidden_layer, step=step)

                if step % report_on_batch_number==0:
                    self.print_evaluation(loss, eval_dataset, step_number=step)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_dataset, step_number="Omega")
        self.eval()

    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        '''
        for anything additional that needs to be done before training starts
        '''
        return
    
    def after_step_update(self, hidden_layer=None, step=None):
        '''
        for anything additional that needs to be done after each training step
        '''
        return

    def model_specs_to_string(self, eval_dataset=None):
        '''
        returns a string representation of the model
        '''
        information=[f"Model type: {type(self)}", 
                     f"Number of features: {self.num_features}", 
                     f"Number of parameters: {sum([param.numel() for param in self.parameters()])}", 
                     f"Number of games trained on: {self.num_data_trained_on}"]
        information.extend(self.report_model_specific_features())
        information.extend([f"Classifications:",
                            f"    Number of SMD>2 classifiers (None=not evaluated): {self.num_classifier_above_threshold(metric_name='classifier_smds', threshold=2)}",
                            f"    Average classifer SMD (None=not evaluated): {self.average_classifier_score(metric_name='classifier_smds')}",
                            f"    Number of AUROC>.9 classifiers (None=not evaluated): {self.num_classifier_above_threshold()}",
                            f"    Average classifer AUROC (None=not evaluated): {self.average_classifier_score()}",
                            ])
        if eval_dataset:
            losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset, include_loss=True)
            test_loss=losses.mean()
            l0_sparsity=self.compute_l0_sparsity(hidden_layers)
            dead_features=self.count_dead_features(hidden_layers)
            reconstruction_error=self.reconstruction_error(residual_streams, reconstructed_residual_streams)
            information.extend([
                f"Results of evaluation on {len(eval_dataset)} games ({residual_streams.shape[0]*residual_streams.shape[1]} activations):", 
                f"    Loss: {test_loss:.3f}", 
                f"    Reconstruction Loss: {reconstruction_error:.3f}",
                f"    L0 Sparsity: {l0_sparsity:.1f}", 
                f"    Dead features: {dead_features:.0f}", ])
        return "\n".join(information)

    def report_model_specific_features(self):
        '''
        returns a list of strings, describing features specific to the type of SAE, such as hyperparameters
        '''
        return ["No model-specific features"]

    @abstractmethod
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        if compute_loss is False, loss is None
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        pass

    @torch.inference_mode()
    def compute_all_aurocs(self, evaluation_dataset:DataLoader, alternate_players=True):
        '''
        computes aurocs of each sae feature on the entire evaluation_dataset
        returns a shape (N,64,3) tensor, where N is the number of features
        alters the self state by writing the value of self.number_of_high_quality_classifiers
        '''
        _, hidden_layers, __=self.catenate_outputs_on_dataset(evaluation_dataset, include_loss=False)
        board_states= get_board_states(evaluation_dataset,alternate_players=alternate_players)
        board_states=self.trim_to_window(board_states)
        hidden_layers=hidden_layers.flatten(end_dim=-2)
        board_states=board_states.flatten(end_dim=-2)
        game_not_ended_mask=board_states[:,0]>-100
        hidden_layers=hidden_layers[game_not_ended_mask]
        board_states=board_states[game_not_ended_mask]
        aurocs=torch.zeros((hidden_layers.shape[1], board_states.shape[1], 3))
        for i, feature_activation in tqdm(enumerate(hidden_layers.transpose(0,1))):
            for j, board_position in enumerate(board_states.transpose(0,1)):
                for k, piece_class in enumerate([0,1,2]):
                    is_target_piece=board_position==piece_class
                    ended_game_mask= board_position>-100
                    metric = BinaryAUROC()
                    metric.update(feature_activation[ended_game_mask], is_target_piece[ended_game_mask].int())
                    this_auroc=float(metric.compute())
                    this_auroc_rectified=max(this_auroc, 1-this_auroc)
                    aurocs[i,j,k]=this_auroc_rectified
        self.classifier_aurocs=aurocs

    @torch.inference_mode()
    def compute_all_smd(self, evaluation_dataset:DataLoader, alternate_players=True):
        '''
        computes aurocs of each sae feature on the entire evaluation_dataset
        returns a shape (N,64,3) tensor, where N is the number of features
        alters the self state by writing the value of self.number_of_high_quality_classifiers
        '''
        _, hidden_layers, __=self.catenate_outputs_on_dataset(evaluation_dataset, include_loss=False)
        board_states= get_board_states(evaluation_dataset,alternate_players=alternate_players)
        board_states=self.trim_to_window(board_states)
        hidden_layers=hidden_layers.flatten(end_dim=-2)
        board_states=board_states.flatten(end_dim=-2)
        game_not_ended_mask=board_states[:,0]>-100
        hidden_layers=hidden_layers[game_not_ended_mask]
        board_states=board_states[game_not_ended_mask]
        standardized_mean_distances=torch.zeros((hidden_layers.shape[1], board_states.shape[1], 3))
        for i, feature_activation in tqdm(enumerate(hidden_layers.transpose(0,1))):
            feature_stdev=feature_activation.std()
            if feature_stdev<1e-10:
                continue
            for j, board_position in enumerate(board_states.transpose(0,1)):
                for k, piece_class in enumerate([0,1,2]):
                    if j in [27,28,35,36] and k==1:
                        #center pieces are never empty
                        continue
                    is_target_piece=board_position==piece_class
                    first_mean=feature_activation[is_target_piece].mean()
                    second_mean=feature_activation[~ is_target_piece].mean()
                    smd=torch.abs(first_mean-second_mean)/feature_stdev
                    standardized_mean_distances[i,j,k]=smd
        self.classifier_smds=standardized_mean_distances

    def num_classifier_above_threshold(self, metric_name="classifier_aurocs", threshold=.9):
        '''
        returns the number of board state features which are well-classified (the named metric is above threshold)
        if self.metric_name is None, will return None instead
        supported choices for metric_name: "classifier_aurocs", "classifier_smds"
        '''
        metric=getattr(self, metric_name)
        if metric is None:
            return None
        best_scores=metric.max(dim=0).values
        return int((best_scores>threshold).sum())

    def average_classifier_score(self, metric_name="classifier_aurocs"):
        '''
        returns the classifer accuracy (of the named metric) averaged over all positions/pieces
        if self.metric_name is None, will return None instead
        supported choices for metric_name: "classifier_aurocs", "classifier_smds"

        if self.classifier aurocs is computed, returns the classifer accuracy averaged over all positions/pieces
        if self.classifier aurocs is not computed, returns None
        '''
        metric=getattr(self, metric_name)
        if metric is None:
            return None
        best_scores=metric.max(dim=0).values
        return float(torch.mean(best_scores))
    
class SAEPretrainedProbes(SAETemplate):
    def __init__(self, gpt: GPTforProbing, probe_layer: int, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, window_start_trim, window_end_trim)
        self.gpt.to(device)

        residual_stream_size=gpt.pos_emb.shape[-1]
        probe = BatteryProbeClassification(device, probe_class=3, num_task=64, input_dim=residual_stream_size)
        probe_path = f"EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/linearProbe_Map_New_8L8H_GPT_Layer{probe_layer}.ckpt"
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        self.probe = probe.to(device)

    def forward(self, residual_stream, compute_loss=False):
        logits = self.probe.proj(residual_stream)
        loss = None
        return loss, residual_stream, logits, residual_stream

