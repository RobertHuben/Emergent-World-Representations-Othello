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
        print_message=f"Train loss, test loss, l0 sparsity, dead features after {step_number} steps: {train_loss.item():.2f}, {test_loss:.2f}, {l0_sparsity:.1f}, {dead_features:.0f}"
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

    def train_model(self, train_dataset:CharDataset, eval_dataset:CharDataset, batch_size=64, num_epochs=1, report_every_n_steps=500, learning_rate=1e-3, fixed_seed=1337):
        '''
        performs a training loop on self, with printed evaluations
        '''
        if fixed_seed:
            torch.manual_seed(fixed_seed)
        self.to(device)
        self.train()
        optimizer=torch.optim.AdamW(self.parameters(), lr=learning_rate)
        step=0
        report_on_batch_number=report_every_n_steps//batch_size

        self.training_prep(train_dataset=train_dataset, eval_dataset=eval_dataset, batch_size=batch_size, num_epochs=num_epochs)

        print(f"Beginning model training on {device}!")
        for epoch in range(num_epochs):
            train_dataloader=iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
            print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_steps} steps.")
            
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
    def __init__(self, gpt: GPTforProbing, probe_layer: int):
        super().__init__(gpt)
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

class SAEAnthropic(SAETemplate):

    def __init__(self, gpt:GPTforProbing, num_features:int, sparsity_coefficient:float, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.sparsity_coefficient=sparsity_coefficient
        residual_stream_size=gpt.pos_emb.shape[-1]
        decoder_initial_value=torch.randn((self.num_features, residual_stream_size))
        decoder_initial_value=decoder_initial_value/decoder_initial_value.norm(dim=0) # columns of norm 1
        decoder_initial_value*=decoder_initialization_scale # columns of norm decoder_initial_value
        self.encoder=torch.nn.Parameter(torch.clone(decoder_initial_value).transpose(0,1).detach())
        self.encoder_bias=torch.nn.Parameter(torch.zeros((self.num_features)))
        self.decoder=torch.nn.Parameter(decoder_initial_value)
        self.decoder_bias=torch.nn.Parameter(torch.zeros((residual_stream_size)))


    def forward(self, residual_stream, compute_loss=False):
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        if compute_loss:
            loss = self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream)
        else:
            loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream
    
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)
        sparsity_loss= self.sparsity_loss_function(hidden_layer)*self.sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss

    def activation_function(self, encoder_output):
        return F.relu(encoder_output)

    def sparsity_loss_function(self, hidden_layer):
        decoder_row_norms=self.decoder.norm(dim=1)
        return torch.mean(hidden_layer*decoder_row_norms)
    
    def report_model_specific_features(self):
        return [f"Sparsity loss coefficient: {self.sparsity_coefficient}"]

class SAEDummy(SAETemplate):
    '''
    "SAE" whose hidden layer and reconstruction is just the unchanged residual stream
    '''

    def __init__(self, gpt:GPTforProbing, num_features=1024):
        super().__init__(gpt=gpt, num_features=num_features)
        self.to(device)

    def forward(self, residual_stream, compute_loss=False):
        return None, residual_stream,residual_stream,residual_stream

#supported variants: mag_in_aux_loss, relu_only
class Gated_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, no_aux_loss=False, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.b_gate = self.encoder_bias #just renaming to make this more clear
        self.r_mag = torch.nn.Parameter(torch.randn(self.hidden_layer_size,))
        self.b_mag = torch.nn.Parameter(torch.randn(self.hidden_layer_size,))
        self.no_aux_loss = no_aux_loss

    def forward(self, residual_stream, compute_loss=False):
        if self.no_aux_loss:
            encoder = F.normalize(self.encoder, p=2, dim=1)
        else:
            encoder = self.encoder
        encoding = residual_stream @ encoder
        if self.no_aux_loss:
            hidden_layer = (F.relu(encoding + self.b_gate) - self.b_gate) * torch.exp(self.r_mag) + self.b_mag
        else:
            hidden_layer_before_gating = encoding * torch.exp(self.r_mag) + self.b_mag
            hidden_layer = ((encoding + self.b_gate) > 0) * hidden_layer_before_gating
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1)
        reconstructed_residual_stream = hidden_layer @ normalized_decoder + self.decoder_bias

        if compute_loss:
            reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)

            hidden_layer_without_gating_or_mag = F.relu(encoding+self.b_gate)
            sparsity_loss = self.sparsity_loss_function(hidden_layer_without_gating_or_mag)*self.sparsity_coefficient

            if self.no_aux_loss:
                auxiliary_loss = 0.0
            else:
                reconstruction_without_gating = hidden_layer_without_gating_or_mag @ normalized_decoder.detach() + self.decoder_bias.detach() #seriously, this doesn't use r_mag or b_mag????
                auxiliary_loss = self.reconstruction_error(residual_stream, reconstruction_without_gating)
            loss = reconstruction_loss + sparsity_loss + auxiliary_loss
        else:
            loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream
    
    def sparsity_loss_function(self, gated_activations):
        return torch.mean(gated_activations)

class ActivationQueue:
    def __init__(self, length):
        self.list = []
        self.length = length

    def add(self, activations):
        self.list.insert(0, activations)
        while len(self.list) > self.length:
            self.list.pop()
    
    def sparsity_coefficient_factor(self, last_p, next_p):
        list_as_tensor = torch.stack(self.list).to(device)
        return torch.sum(list_as_tensor**last_p) / torch.sum(list_as_tensor**next_p)

class P_Annealing_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, anneal_start: int, p_end=0.2, queue_length=10, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.p = 1
        self.anneal_start = anneal_start
        self.p_end = p_end
        self.queue = ActivationQueue(queue_length)
    
    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.p_step = (1 - self.p_end)/(num_steps - self.anneal_start)
        return
    
    def after_step_update(self, hidden_layer=None, step = None):
        if self.anneal_start - step <= self.queue.length:
            self.queue.add(hidden_layer.detach())
        if step >= self.anneal_start:
            next_p = self.p - self.p_step
            self.sparsity_coefficient *= self.queue.sparsity_coefficient_factor(self.p, next_p)
            self.p = next_p
        return
    
    def sparsity_loss_function(self, hidden_layer):
        return torch.norm(hidden_layer, p=self.p, dim=-1).sum() / hidden_layer.numel()

class Smoothed_L0_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, epsilon: float, delta: float):
        super().__init__(gpt, num_features, sparsity_coefficient)
        self.epsilon = epsilon
        self.delta = delta

    def sparsity_loss_function(self, hidden_layer):
        decoder_row_norms=self.decoder.norm(dim=1)
        normalized_hidden_layer = hidden_layer*decoder_row_norms #does doing this just like in SAEAnthropic make sense?
        functions = [CallableConstant(0.0), CallableConstant(1.0)]
        transitions = [{"x":self.epsilon, "epsilon":self.epsilon, "delta":self.delta, "focus":"left"}]
        return torch.mean(smoothed_piecewise(normalized_hidden_layer, functions, transitions))
    
class Without_TopK_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, k: int, p: int):
        super().__init__(gpt, num_features, sparsity_coefficient)
        self.k = k
        self.p = p

    def sparsity_loss_function(self, hidden_layer):
        decoder_row_norms=self.decoder.norm(dim=1)
        normalized_hidden_layer = hidden_layer*decoder_row_norms #does doing this just like in SAEAnthropic make sense?
        top_k_indices = torch.topk(torch.abs(hidden_layer), self.k, dim=-1).indices #should we find topk from hidden_layer or normalized_hidden_layer?
        top_k_mask = torch.ones(hidden_layer.shape).to(device).scatter_(-1, top_k_indices, 0)
        without_top_k = normalized_hidden_layer * top_k_mask
        return torch.mean(torch.norm(without_top_k, p=self.p, dim=-1))

    
# class No_Sparsity_Loss_SAE(SAETemplate):
#     def __init__(self, gpt: GPTforProbing, num_features: int, window_start_trim: int, window_end_trim: int):
#         super().__init__(gpt, num_features, 0.0, window_start_trim, window_end_trim)
#     def sparsity_loss_function(self, hidden_layer):
#         return 0.0
    
class Leaky_Topk_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, num_features: int, epsilon: float, k:int, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.epsilon = epsilon
        self.k=k
        residual_stream_size=gpt.pos_emb.shape[-1]
        decoder_initial_value=torch.randn((self.num_features, residual_stream_size))
        decoder_initial_value=decoder_initial_value/decoder_initial_value.norm(dim=0) # columns of norm 1
        decoder_initial_value*=decoder_initialization_scale # columns of norm decoder_initial_value
        self.encoder=torch.nn.Parameter(torch.clone(decoder_initial_value).transpose(0,1).detach())
        self.encoder_bias=torch.nn.Parameter(torch.zeros((self.num_features)))
        self.decoder=torch.nn.Parameter(decoder_initial_value)
        self.decoder_bias=torch.nn.Parameter(torch.zeros((residual_stream_size)))

    def activation_function(self, encoder_output):
        kth_value = torch.topk(torch.abs(encoder_output), k=self.k).values.min(dim=-1).values
        return suppress_lower_activations(encoder_output, kth_value, epsilon=self.epsilon)
    
    def forward(self, residual_stream):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        return residual_stream, hidden_layer, reconstructed_residual_stream

    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        return self.reconstruction_error(residual_stream, reconstructed_residual_stream)

    def report_model_specific_features(self):
        return [f"k (sparsity): {self.k}", f"Epsilon (leakyness): {self.epsilon}"]

class Dimension_Reduction_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, num_features: int, start_index: int, start_proportion: float, end_proportion: float, epsilon: float):
        super().__init__(gpt, num_features)
        self.start_index = start_index
        self.start_proportion = start_proportion
        self.end_proportion = end_proportion
        self.epsilon = epsilon
        self.activation_f = reduce_dimensions_activation(Expand(self.start_index, self.start_proportion, self.end_proportion), max_n = self.num_features, epsilon=self.epsilon)

    def activation_function(self, encoder_output):
        return self.activation_f(encoder_output)
    
class CallableConstant(object):
    def __init__(self, constant): self.constant = constant
    def __call__(self, input):
        if torch.is_tensor(input):
            return self.constant * torch.ones(input.shape).to(device)
        else:
            return torch.Tensor((self.constant,)).to(device)

class CallableList(object):
    def __init__(self, list): self.list = torch.Tensor(list).to(device)
    def __call__(self, index):
        if torch.is_tensor(index):
            index = index.int()
            return self.list[index] 
        else:
            assert isinstance(index, int) or index == int(index), f"Input {index} is not an int."
            assert int(index) in range(len(self.list)), f"Input {index} is out of range."
            return self.list[int(index)]

class Expand(CallableList):
    def __init__(self, start_index, start_p, end_p, max_n=1024):
        expand = [10,10] #start off with any values, doesn't matter
        finished_expanding = False
        for n in range(2, max_n+1):
            if n < start_index:
                a_n = (1-start_p) * expand[n-1] + start_p * expand[n-1]*n/(n-1)
                expand.append(a_n)
            else:
                a_n1 = 2*expand[n-1] - expand[n-2]
                a_n2 = (1-end_p) * expand[n-1] + end_p * expand[n-1]*n/(n-1)
                if a_n1 <= a_n2:
                    a_n = a_n1
                else:
                    a_n = a_n2
                    if not finished_expanding:
                        print(f"Expanded from {start_index} to {n}")
                        finished_expanding = True
                expand.append(a_n)
        super().__init__(expand)

class reduce_dimensions_activation(object):
    def __init__(self, a, max_n = 1024, epsilon=0.1):
        self.epsilon = epsilon
        if isinstance(a, list):
            a = CallableList(a)
        else:
            assert callable(a), "a is not a list or function"

        tolerance = 0.001
        for n in range(2, max_n + 1):
            assert a(n-1) <= a(n) <= a(n-1)*n/(n-1), f"a({n}) is not between a({n-1}) and {n}/{n-1} * a({n-1})."
            if n != 2:
                assert a(n) - a(n-1) <= a(n-1) - a(n-2) + tolerance, f"Difference between a({n}) and a({n-1}) is greater than the previous difference."

        self.a = a

    def __call__(self, t):        
        remaining_mask = torch.ones(t.shape).to(device)
        while True:
            n = torch.sum(remaining_mask, dim=-1)
            n_or_2 = torch.maximum(n, 2*torch.ones(n.shape).to(device))
            bound_constant = 1 - self.a(n_or_2-1)/self.a(n_or_2)
            new_remaining = 1*(torch.abs(t)*remaining_mask > torch.unsqueeze(torch.sum(torch.abs(t)*remaining_mask, dim=-1) * bound_constant, dim=-1))
            finished_mask = torch.logical_or(torch.eq(remaining_mask, new_remaining), torch.unsqueeze(torch.eq(n, torch.ones(n.shape).to(device)), dim=-1)) #finished if, for each set of activations, either no updates this loop or n = 1
            if torch.sum(~finished_mask) == 0:
                break
            remaining_mask = new_remaining

        k = torch.sum(remaining_mask, dim=-1)
        k_or_plus_1_or_2 = torch.maximum(torch.unsqueeze(k, dim=-1) + 1-remaining_mask, 2*torch.ones(t.shape).to(device))
        bound_constant = 1 - self.a(k_or_plus_1_or_2-1)/self.a(k_or_plus_1_or_2)
        bound = (torch.unsqueeze(torch.sum(torch.abs(t)*remaining_mask, dim=-1), dim=-1) + torch.abs(t) * (1 - remaining_mask)) * bound_constant
        return k, suppress_lower_activations(t, bound, epsilon=self.epsilon, inclusive=False, mode="absolute")


def suppress_lower_activations(t, bound, epsilon, inclusive=True, mode="absolute"):
    if torch.is_tensor(bound) and bound.numel() != 1:
        while bound.dim() < t.dim():
            bound = torch.unsqueeze(bound, -1)
    above_mask = (torch.abs(t) >= bound) if inclusive else (torch.abs(t) > bound)
    above_only = t * above_mask
    below_only = t * (~above_mask)
    if mode == "absolute":
        return above_only + epsilon/bound * below_only
    elif mode == "relative":
        return above_only + epsilon * below_only

def smoothed_piecewise(input, functions, transitions):
    assert len(functions) == len(transitions) + 1, "Incorrect number of transitions for number of functions given."
    for i in range(len(transitions)-1):
        assert transitions[i]["x"] < transitions[i+1]["x"], "Transition list not sorted by x-value in ascending order."
    sig = torch.nn.Sigmoid()
    sum = functions[0](input) #first add in the initial function
    for i, t in enumerate(transitions): #then at each transition we will subtract off the previous function and add on the next function
        g = functions[i]
        h = functions[i+1]
        if "focus" in t:
            if t["focus"] == "right":
                t["x"] = t["x"] - t["delta"]
                n = torch.log(abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"]))/t["epsilon"] - 1)/t["delta"]
            else:
                assert t["focus"] == "left", "Unrecognized focus for a transition (must be either right or left)."
                t["x"] = t["x"] + t["delta"]
                n = torch.log(abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))/t["epsilon"] - 1)/t["delta"]
        else:
            left_and_right = torch.stack((abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"])), abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))))
            n = torch.log(torch.max(left_and_right, dim=0).values/t["epsilon"] - 1)/t["delta"]
        sum += sig(n*(input-t["x"])) * h(input) - sig(n*(input-t["x"])) * g(input)
    return sum