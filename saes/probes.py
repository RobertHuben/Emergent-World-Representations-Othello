import sys
import os

from torch.nn.modules import Module
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))

from abc import abstractmethod
import torch
from torch.nn import functional as F
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from EWOthello.mingpt.dataset import CharDataset
from EWOthello.data.othello import OthelloBoardState
from sae_template import SAETemplate
from architectures import suppress_lower_activations

device='cuda' if torch.cuda.is_available() else 'cpu'

class ProbeDataset(Dataset):
    def __init__(self, game_dataset:CharDataset):
        self.game_dataset = game_dataset
        self.enemy_own_modifier = np.concatenate([np.ones((1,64))*(-1)**i for i in range(59)],axis=0)
    
    def __len__(self):
        return len(self.game_dataset)

    def __getitem__(self, index):
        x, _ = self.game_dataset[index]
        tbf = [self.game_dataset.itos[_] for _ in x.tolist()]
        valid_until = tbf.index(-100) if -100 in tbf else 999

        # Get the board state vectors
        a = OthelloBoardState()
        board_state = a.get_gt(tbf[:valid_until], "get_state")
        board_state = (np.array(board_state) - 1.0) * self.enemy_own_modifier[:valid_until, :] + 1.0
        board_state = torch.tensor(board_state, dtype=torch.float32).to(device)
        if valid_until < len(tbf):
            padding = -100*torch.ones(len(tbf)-valid_until, 64).to(device)
            board_state = torch.cat((board_state, padding), 0)
        return x, board_state.to(dtype=int)

class SAEforProbing(torch.nn.Module):
    def __init__(self, sae:SAETemplate):
        super().__init__()
        self.sae = sae
        self.output_dim = sae.num_features

    def forward(self, input, layer_to_probe="hidden"):
        if layer_to_probe == "residual":
            return self.sae.gpt(input)
        else:
            loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.sae.forward_on_tokens(input, compute_loss=False)
            if layer_to_probe == "hidden":
                return hidden_layer
            elif layer_to_probe == "reconstruction":
                return reconstructed_residual_stream

class LinearProbe(torch.nn.Module):
    def __init__(self, model_to_probe:torch.nn.Module, input_dim:int, layer_to_probe="hidden"):
        super().__init__()
        self.model_to_probe = model_to_probe
        self.layer_to_probe = layer_to_probe
        self.linear = torch.nn.Linear(input_dim, 64*3)
        self.num_data_trained_on=0
        self.accuracy = None
        self.accuracy_by_board_position = None

        for param in model_to_probe.parameters():
            param.requires_grad=False

    def forward_on_tokens(self, token_sequences, targets=None):
        if isinstance(self.model_to_probe, SAEforProbing):
            activations = self.model_to_probe(token_sequences, self.layer_to_probe)
            if self.layer_to_probe != "residual":
                targets = self.model_to_probe.sae.trim_to_window(targets)
        else:
            activations = self.model_to_probe(token_sequences)
        loss, logits = self.forward(activations, targets)
        logits = logits.view((logits.shape[0], logits.shape[1], 64, 3))
        return loss, logits, targets
    
    def forward(self, activations, targets):
        logits = self.linear(activations)
        loss = self.loss(logits, targets)
        return loss, logits
    
    def loss(self, logits, targets):
        return F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), ignore_index=-100)
    
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

        train_probe_dataset = ProbeDataset(train_dataset)
        eval_probe_dataset = ProbeDataset(eval_dataset)

        self.training_prep(train_dataset=train_dataset, batch_size=batch_size, num_epochs=num_epochs)

        print(f"Beginning probe training on {device}!")

        for epoch in range(num_epochs):
            train_dataloader=iter(torch.utils.data.DataLoader(train_probe_dataset, batch_size=batch_size, shuffle=True))
            print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_data} games.")
            batch_count=0 #for testing
            for input_batch, label_batch in tqdm(train_dataloader):
                batch_count += 1 #for testing
                input_batch=input_batch.to(device)
                label_batch=label_batch.to(device)
                step+=1
                self.num_data_trained_on+=len(input_batch)
                optimizer.zero_grad(set_to_none=True)
                loss, logits, targets = self.forward_on_tokens(input_batch, label_batch)
                loss.backward()
                optimizer.step()

                if step % report_on_batch_number==0:
                    self.print_evaluation(loss, eval_probe_dataset, step_number=step)

                self.after_step_update(step=step)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_probe_dataset, step_number="Omega")
        self.eval()

    def training_prep(self, train_dataset=None, batch_size=None, num_epochs=None):
        '''
        for anything additional that needs to be done before training starts
        '''
        return
    
    def after_step_update(self, step=None):
        '''
        for anything additional that needs to be done after each training step
        '''
        return
    
    def catenate_outputs_on_dataset(self, dataset:ProbeDataset, batch_size=8):
        '''
        runs the model on the entire dataset, one batch at a time, catenating the outputs
        '''
        losses=[]
        logits_list=[]
        targets_list=[]
        test_dataloader=iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for test_input, test_labels in test_dataloader:
            test_input=test_input.to(device)
            test_labels = test_labels.to(device)
            loss, logits, targets = self.forward_on_tokens(test_input, test_labels)
            losses.append(loss)
            logits_list.append(logits)
            targets_list.append(targets)
        losses=torch.stack(losses)
        logits=torch.stack(logits_list)
        targets=torch.stack(targets_list)
        return losses, logits, targets

    def compute_accuracy(self, logits, targets):
        predictions = torch.argmax(logits, dim=-1, keepdim=False)
        hits = predictions == targets
        self.accuracy_by_board_position = hits.reshape(-1, 64).sum(dim=0) / (targets != -100).reshape(-1,64).sum(dim=0)
        self.accuracy_by_board_position = self.accuracy_by_board_position.view(8,8)
        self.accuracy = self.accuracy_by_board_position.mean()

    def print_evaluation(self, train_loss, eval_dataset:ProbeDataset, step_number="N/A"):
        if not isinstance(eval_dataset, ProbeDataset):
            eval_dataset = ProbeDataset(eval_dataset)
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        test_loss=losses.mean()
        self.compute_accuracy(logits, targets)
        print_message=f"Train loss, test loss, accuracy after {self.num_data_trained_on} training games: {train_loss.item():.2f}, {test_loss:.3f}, {self.accuracy:.4f}"
        tqdm.write(print_message)

    def after_training_eval(self, eval_dataset:ProbeDataset, save_location:str):
        if not isinstance(eval_dataset, ProbeDataset):
            eval_dataset = ProbeDataset(eval_dataset)
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        self.compute_accuracy(logits, targets)
        abs_weights = torch.abs(self.linear.weight)
        top4_features = torch.topk(abs_weights, k=4, dim=1).indices
        top4_weights = self.linear.weight.gather(1, top4_features)
        with open(save_location, 'a') as f:
            f.write(f"Average accuracy: {self.accuracy}\n")
            f.write(f"Accuracies by board position:\n {self.accuracy_by_board_position}\n")
            f.write(f"\nTop 4 features by board position and class:\n{top4_features.reshape((8, 8, 3, 4))}\n")
            f.write(f"\nTop 4 weights by board position and class:\n{top4_weights.reshape((8, 8, 3, 4))}")

class Constant_Probe(LinearProbe):
    def __init__(self, model_to_probe: Module, input_dim: int):
        super().__init__(model_to_probe, input_dim)
        self.zeros = torch.zeros((input_dim, 64*3)).to(device)

    def forward(self, activations, targets):
        logits = activations @ self.zeros + self.linear.bias
        loss = self.loss(logits, targets)
        return loss, logits

class L1_Sparse_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, sparsity_coeff: float, normalize=False):
        input_dim = model_to_probe.sae.num_features
        super().__init__(model_to_probe, input_dim)
        self.sparsity_coeff = sparsity_coeff
        self.normalize = normalize

    def forward(self, activations, targets):
        if self.normalize:
            normalized_weight = F.normalize(self.linear.weight, p=2, dim=1) #normalize rows, so that L1 term increases sparsity rather than just decreasing all weights
            logits = activations @ normalized_weight.transpose(0, 1) + self.linear.bias
        else: #don't normalize?
            normalized_weight = self.linear.weight
            logits = self.linear(activations)
        accuracy_loss = super().loss(logits, targets)
        sparsity_loss = torch.abs(normalized_weight).mean()
        loss = accuracy_loss + self.sparsity_coeff * sparsity_loss
        return loss, logits
    
class Without_Topk_Sparse_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, k: int, sparsity_coeff: float):
        input_dim = model_to_probe.sae.num_features
        super().__init__(model_to_probe, input_dim)
        self.sparsity_coeff = sparsity_coeff
        self.k = k

    def forward(self, activations, targets):
        #normalized_weight = F.normalize(self.linear.weight, p=2, dim=1) #normalize rows, so that sparsity loss term increases sparsity rather than just decreasing all weights
        #logits = activations @ normalized_weight.transpose(0, 1) + self.linear.bias
        normalized_weight = self.linear.weight #don't normalize?
        logits = self.linear(activations) #don't normalize?
        accuracy_loss = super().loss(logits, targets)
        
        top_k_indices = torch.topk(torch.abs(normalized_weight), self.k, dim=1).indices
        top_k_mask = torch.ones(normalized_weight.shape).to(device).scatter_(1, top_k_indices, 0)
        without_top_k = normalized_weight * top_k_mask
        sparsity_loss = torch.abs(without_top_k).mean()

        loss = accuracy_loss + self.sparsity_coeff * sparsity_loss
        return loss, logits
      
class Leaky_Topk_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, k: int, epsilon: float):
        input_dim = model_to_probe.sae.num_features
        super().__init__(model_to_probe, input_dim)
        self.k = k
        self.epsilon = epsilon

    def forward(self, activations, targets):
        #normalized_weight = F.normalize(self.linear.weight, p=2, dim=1) #normalize rows, so that model can't just decrease all weights to epsilon or below
        normalized_weight = self.linear.weight #don't normalize?
        kth_value = torch.topk(torch.abs(normalized_weight), k=self.k, dim=1).values.min(dim=1).values
        suppressed_weights = suppress_lower_activations(normalized_weight, kth_value, epsilon=self.epsilon, mode="absolute")
        logits = activations @ suppressed_weights.transpose(0, 1) + self.linear.bias
        loss = super().loss(logits, targets)
        return loss, logits
    
class K_Annealing_Probe(Leaky_Topk_Probe):
    def __init__(self, model_to_probe: SAEforProbing, epsilon: float, k_start: int, before_anneal_proportion: float, k_end: int, after_anneal_proportion: float):
        assert before_anneal_proportion + after_anneal_proportion <= 1, "Negative time given for annealing!"
        super().__init__(model_to_probe, k_start, epsilon)
        self.k_start = k_start
        self.before_anneal_proportion = before_anneal_proportion
        self.k_end = k_end
        self.after_anneal_proportion = after_anneal_proportion

    def training_prep(self, train_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.before_anneal_steps = round(num_steps*self.before_anneal_proportion)
        self.after_anneal_steps = round(num_steps*self.after_anneal_proportion)
        self.anneal_steps = num_steps - self.before_anneal_steps - self.after_anneal_steps
        self.a = self.model_to_probe.sae.num_features - self.k_end + 0.5
        self.b = math.log(0.5/self.a)/self.anneal_steps
        self.c = self.k_end - 0.5
        return
    
    def after_step_update(self, step=None):
        if step > self.before_anneal_steps:
            if step == self.before_anneal_steps+1:
                print("\nStarting annealing now.\n")
            if step <= self.before_anneal_steps + self.anneal_steps:
                self.k = round(self.a*math.exp(self.b*(step-self.before_anneal_steps)) + self.c)
            else:
                if step == self.before_anneal_steps + self.anneal_steps + 1:
                    print("\nAnnealing finished.\n")
                self.k = self.k_end
        return
    
class Gated_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, init_type="ones"):
        Module.__init__(self)
        self.model_to_probe = model_to_probe
        self.num_data_trained_on=0
        self.accuracy = None
        self.accuracy_by_board_position = None

        for param in model_to_probe.parameters():
            param.requires_grad=False

        num_features = self.model_to_probe.sae.num_features
        if init_type == "ones":
            self.feature_choice = torch.nn.Parameter(torch.ones((64, num_features)))
        elif init_type == "zeros":
            self.feature_choice = torch.nn.Parameter(torch.zeros((64, num_features)))
        elif init_type == "random":
            self.feature_choice = torch.nn.Parameter(torch.abs(torch.randn((64, num_features))))
        self.weight = torch.nn.Parameter(torch.randn((64, 3, num_features)))
        self.bias = torch.nn.Parameter(torch.zeros(64, 3))

    def forward(self, activations, targets):
        activated_feature_choice = self.feature_choice_activation()
        activations_chosen = activated_feature_choice * activations.unsqueeze(-2)
        logits = torch.einsum("ijk,...ik->...ij", self.weight, activations_chosen) + self.bias

        sparsity_loss = self.sparsity_loss(activated_feature_choice)
        accuracy_loss = super().loss(logits, targets)
        loss = accuracy_loss + sparsity_loss
        return loss, logits
    
    @abstractmethod
    def feature_choice_activation(self):
        pass
    
    @abstractmethod
    def sparsity_loss(self, activated_feature_choice):
        pass

    def after_training_eval(self, eval_dataset:ProbeDataset, save_location:str):
        if not isinstance(eval_dataset, ProbeDataset):
            eval_dataset = ProbeDataset(eval_dataset)
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        self.compute_accuracy(logits, targets)
        activated_feature_choice = self.feature_choice_activation()
        combined_weights = activated_feature_choice.unsqueeze(-2) * self.weight
        top4_features = torch.topk(torch.abs(combined_weights), k=4, dim=-1).indices
        top4_weights = combined_weights.gather(-1, top4_features)
        top4_feature_choices = torch.topk(torch.abs(activated_feature_choice), k=4, dim=-1).indices
        top4_feature_choice_weights = activated_feature_choice.gather(-1, top4_feature_choices)
        with open(save_location, 'a') as f:
            f.write(f"Average accuracy: {self.accuracy}\n")
            f.write(f"Accuracies by board position:\n {self.accuracy_by_board_position}\n")
            f.write(f"\nTop 4 features by board position and class:\n{top4_features.reshape((8, 8, 3, 4))}\n")
            f.write(f"\nTop 4 weights by board position and class:\n{top4_weights.reshape((8, 8, 3, 4))}\n")
            f.write(f"\nTop 4 features choices by board position:\n{top4_feature_choices.reshape((8, 8, 4))}\n")
            f.write(f"\nTop 4 chosen weights by board position:\n{top4_feature_choice_weights.reshape((8, 8, 4))}")

class L1_Gated_Probe(Gated_Probe):
    def __init__(self, model_to_probe: SAEforProbing, sparsity_coeff: float, init_type="ones"):
        super().__init__(model_to_probe, init_type)
        self.sparsity_coeff = sparsity_coeff

    def feature_choice_activation(self):
        return F.relu(self.feature_choice)
    
    def sparsity_loss(self, activated_feature_choice):
        return self.sparsity_coeff * activated_feature_choice.mean() #note that all elements of activated_feature_choice are already non-negative

class K_Annealing_Gated_Probe(Gated_Probe, K_Annealing_Probe):
    def __init__(self, model_to_probe: SAEforProbing, epsilon: float, k_start: int, before_anneal_proportion: float, k_end: int, after_anneal_proportion: float, init_type="ones"):
        super().__init__(model_to_probe, init_type)
        self.epsilon = epsilon
        self.k = k_start
        self.k_start = k_start
        self.before_anneal_proportion = before_anneal_proportion
        self.k_end = k_end
        self.after_anneal_proportion = after_anneal_proportion

    def feature_choice_activation(self):
        kth_value = torch.topk(F.relu(self.feature_choice), k=self.k, dim=1).values.min(dim=1).values
        return suppress_lower_activations(self.feature_choice, kth_value, epsilon=self.epsilon, mode="absolute")

    def sparsity_loss(self, activated_feature_choice):
        return 0.0