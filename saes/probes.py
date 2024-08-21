import sys
import os

from torch.nn.modules import Module
sys.path.append(os.path.join( os.path.dirname ( __file__), os.path.pardir))

from abc import abstractmethod
import torch
from torch.nn import functional as F
import numpy as np
import pickle
import math
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset
from EWOthello.mingpt.dataset import CharDataset
from EWOthello.data.othello import OthelloBoardState
from sae_template import SAETemplate
from architectures import suppress_lower_activations

device='cuda' if torch.cuda.is_available() else 'cpu'

eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
def move_list_to_state_list(move_list:list):
    board = np.zeros((8, 8))
    board[3, 4] = 1
    board[3, 3] = -1
    board[4, 3] = 1
    board[4, 4] = -1

    state_list = []
    forfeited_move = False
    color = 1
    for move in move_list:
        r, c = move // 8, move % 8
        assert board[r, c] == 0, "Illegal move!  There's already a piece there."
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if board[cur_r, cur_c] == 0:
                    break
                elif board[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:  # means one hand forfeited move (unless this is an illegal move)
            forfeited_move = True
            color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if board[cur_r, cur_c] == 0:
                        break
                    elif board[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        assert len(tbf) > 0, "Illegal move!  No flipped pieces."

        for ff in tbf:
            board[ff[0], ff[1]] *= -1
        board[r, c] = color
        state_list.append((board+1).flatten().tolist())

        color *= -1

    return state_list, forfeited_move

enemy_own_modifier = np.concatenate([np.ones((1,64))*(-1)**i for i in range(60)],axis=0)

class ProbeDataset(Dataset):
    def __init__(self, games:list):
        self.games = games
        self.computed_data = [False] * len(games)
        
        self.max_game_length = max([len(game_seq) for game_seq in games])
        chars = sorted(list(set(list(itertools.chain.from_iterable(games)))) + [-100])
        self.char_to_index = {ch: i for i, ch in enumerate(chars)}

        self.num_forfeited_moves = 0

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        datum = self.computed_data[index]
        if datum:
            move_indices, state_seq = datum
        else:
            move_seq = self.games[index]
            game_length = len(move_seq)
            state_seq, forfeited_move = move_list_to_state_list(move_seq)
            state_seq = ((np.array(state_seq) - 1.0) * enemy_own_modifier[:game_length, :] + 1.0).tolist()
            if forfeited_move:
                self.num_forfeited_moves += 1

            if game_length < self.max_game_length:
                padding_length = self.max_game_length - game_length
                move_seq += [-100] * padding_length
                state_seq += [[-100] * 64 for i in range(padding_length)]
            move_indices = [self.char_to_index[char] for char in move_seq]
            self.computed_data[index] = (move_indices, state_seq)

        return torch.tensor(move_indices[:-1], dtype=torch.long), torch.tensor(state_seq[:-1], dtype=torch.long) #I don't know why these datatypes, just copying previous code

class ProbeDatasetPrecomputed(Dataset):
    def __init__(self, games:list):
        game_sequences = [game[0] for game in games]
        
        max_game_length = max([len(game_seq) for game_seq in game_sequences])
        chars = sorted(list(set(list(itertools.chain.from_iterable(game_sequences)))) + [-100])
        self.char_to_index = {ch: i for i, ch in enumerate(chars)}

        self.data = []
        for game in games:
            game_seq, game_states = game[0], game[1]
            game_length = len(game_seq)
            if game_length < max_game_length:
                padding_length = max_game_length - game_length
                game_seq += [-100] * padding_length
                game_states += [[-100] * 64 for i in range(padding_length)]
            game_indices = [self.char_to_index[char] for char in game_seq]
            self.data.append((game_indices[:-1], game_states[:-1])) #I think it is correct to take 1 off the end, since we shouldn't be predicting anything from the last move
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long) #I don't know why these datatypes, just copying previous code

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
    
    def train_model(self, train_dataset:ProbeDataset, eval_dataset:ProbeDataset, batch_size=64, num_epochs=1, report_every_n_data=500, learning_rate=1e-3, fixed_seed=1337):
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

        self.training_prep(train_dataset=train_dataset, batch_size=batch_size, num_epochs=num_epochs)

        print(f"Beginning probe training on {device}!")

        for epoch in range(num_epochs):
            train_dataloader=iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
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
                    self.print_evaluation(loss, eval_dataset, step_number=step)

                self.after_step_update(step=step)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_dataset, step_number="Omega")
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
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        test_loss=losses.mean()
        self.compute_accuracy(logits, targets)
        print_message=f"Train loss, test loss, accuracy after {self.num_data_trained_on} training games: {train_loss.item():.2f}, {test_loss:.3f}, {self.accuracy:.4f}"
        tqdm.write(print_message)

    def after_training_eval(self, eval_dataset:ProbeDataset, save_location:str, weight=None):
        if weight == None:
            weight = self.linear.weight
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        self.compute_accuracy(logits, targets)
        abs_weights = torch.abs(weight)
        top5_features = torch.topk(abs_weights, k=5, dim=1).indices
        top5_weights = weight.gather(1, top5_features)
        with open(save_location, 'a') as f:
            f.write(f"Average accuracy: {self.accuracy}\n")
            f.write(f"Accuracies by board position:\n {self.accuracy_by_board_position}\n")
            f.write(f"\nTop 5 features by board position and class:\n{top5_features.reshape((8, 8, 3, 5))}\n")
            f.write(f"\nTop 5 weights by board position and class:\n{top5_weights.reshape((8, 8, 3, 5))}")

class Constant_Probe(LinearProbe):
    def __init__(self, model_to_probe: Module, input_dim: int):
        super().__init__(model_to_probe, input_dim)
        self.zeros = torch.zeros((input_dim, 64*3)).to(device)

    def forward(self, activations, targets):
        logits = activations @ self.zeros + self.linear.bias
        loss = self.loss(logits, targets)
        return loss, logits

class L1_Sparse_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, sparsity_coeff: float):
        input_dim = model_to_probe.sae.num_features
        super().__init__(model_to_probe, input_dim)
        self.sparsity_coeff = sparsity_coeff

    def forward(self, activations, targets):
        logits = self.linear(activations)
        accuracy_loss = super().loss(logits, targets)
        sparsity_loss = torch.abs(self.linear.weight).mean()
        loss = accuracy_loss + self.sparsity_coeff * sparsity_loss
        return loss, logits
    
    #for evaluating the weights for use in a choice probe
    """ def print_evaluation(self, train_loss, eval_dataset: ProbeDataset, step_number="N/A"):
        super().print_evaluation(train_loss, eval_dataset, step_number)
        abs_weights = torch.abs(self.linear.weight.reshape((64, 3, -1)))
        
        max_weights = torch.max(abs_weights.reshape(64, -1), dim=-1).values
        max_feature_weights = abs_weights.max(dim=1).values
        num_features_chosen = torch.sum(max_feature_weights >= (max_weights.unsqueeze(-1) * 0.01))

        top5_weights = torch.topk(abs_weights.reshape(8, 8, 3, -1), k=5, dim=-1)
        with open("L1_training_top_weights.txt", "a") as f:
            f.write(f"Number of features chosen after {step_number} steps: {num_features_chosen}; Average per position: {num_features_chosen/64}\n")
            f.write(f"Top features after {step_number} steps:\n{top5_weights.indices}\n\n")
            f.write(f"Top weights after {step_number} steps:\n{top5_weights.values}\n\n")
 """

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
    
class Pre_Chosen_Features_Gated_Probe(LinearProbe):
    #chosen_features_list is a list containing 64 lists containing feature indices
    def __init__(self, model_to_probe: SAEforProbing, chosen_features_list: list, initial_weights=None, initial_bias=None):
        Module.__init__(self)
        self.model_to_probe = model_to_probe
        self.layer_to_probe = "hidden"
        self.num_data_trained_on=0
        self.accuracy = None
        self.accuracy_by_board_position = None

        for param in model_to_probe.parameters():
            param.requires_grad=False

        max_features_per_position = max([feature_indices.shape[0] for feature_indices in chosen_features_list])
        self.features_to_use_mask = torch.ones(64, 1, max_features_per_position).to(device)
        padded_features_list = []

        for position, feature_indices in enumerate(chosen_features_list):
            num_unique_indices = feature_indices.shape[0]
            places_to_pad = max_features_per_position - num_unique_indices
            padded_features_list.append(torch.cat((feature_indices, torch.zeros(places_to_pad).to(device))))
            self.features_to_use_mask[position, 0, num_unique_indices:] = torch.zeros(places_to_pad).to(device)
            if initial_weights:
                initial_weights[position] = torch.cat((initial_weights[position], torch.zeros((3, places_to_pad)).to(device)), dim=1)            
        self.indices = torch.stack(padded_features_list).type(torch.int)

        if initial_weights:
            self.weight = torch.nn.Parameter(torch.stack(initial_weights))
        else:
            self.weight = torch.nn.Parameter(torch.randn((64, 3, max_features_per_position)).to(device) * self.features_to_use_mask)
        if initial_bias != None:
            self.bias = torch.nn.Parameter(initial_bias.reshape((64, 3)))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(64, 3))

    def forward(self, activations, targets):
        chosen_activations = activations[:, :, self.indices]
        #test
        """ for batch in range(activations.shape[0]):
            for move in range(activations.shape[1]):
                for position in range(64):
                    for index in range(self.weight.shape[-1]):
                        assert torch.Tensor.isclose(chosen_activations[batch, move, position, index], activations[batch, move, self.indices[position, index]]) """
        weight = self.weight*self.features_to_use_mask
        logits = torch.einsum("ijk,...ik->...ij", weight, chosen_activations) + self.bias
        #test
        """ for batch in range(activations.shape[0]):
            for move in range(activations.shape[1]):
                for position in range(64):
                    for clas in range(3):
                        assert torch.abs(logits[batch, move, position, clas] - torch.sum(self.weight[position, clas] * chosen_activations[batch, move, position])) < 0.001 """
        loss = super().loss(logits, targets)
        return loss, logits
    
    def after_training_eval(self, eval_dataset:ProbeDataset, save_location:str):
        if not isinstance(eval_dataset, ProbeDataset):
            eval_dataset = ProbeDataset(eval_dataset)
        losses, logits, targets=self.catenate_outputs_on_dataset(eval_dataset)
        self.compute_accuracy(logits, targets)
        features_used_per_position = (self.indices.reshape((8, 8, -1)) + 1) * self.features_to_use_mask.reshape((8, 8 ,-1)).type(torch.int) - 1
        num_features_used_per_position = self.features_to_use_mask.sum(dim=-1).reshape((8, 8))
        with open(save_location, 'a') as f:
            f.write(f"Average accuracy: {self.accuracy}; Average number of features used per position: {num_features_used_per_position.mean()}\n")
            f.write(f"Accuracies by board position:\n {self.accuracy_by_board_position}\n")
            f.write(f"Number of features used by board position:\n {num_features_used_per_position}\n")
            f.write(f"\nFeatures used by each board position:\n{features_used_per_position}\n")
            f.write(f"\nWeights by board position and class:\n{self.weight.reshape((8, 8, 3, -1))}")

class Gated_Probe(LinearProbe):
    def __init__(self, model_to_probe: SAEforProbing, init_type="ones"):
        Module.__init__(self)
        self.model_to_probe = model_to_probe
        self.layer_to_probe = "hidden"
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

    def feature_choice_activation(self): #todo: try sigmoid instead of relu
        kth_value = torch.topk(F.relu(self.feature_choice), k=self.k, dim=1).values.min(dim=1).values
        return suppress_lower_activations(F.relu(self.feature_choice), kth_value, epsilon=self.epsilon, mode="absolute")

    def sparsity_loss(self, activated_feature_choice):
        return 0.0