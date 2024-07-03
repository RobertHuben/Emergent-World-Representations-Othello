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
from sae_template import SAETemplate

logger = logging.getLogger(__name__)
device='cuda' if torch.cuda.is_available() else 'cpu'

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
        self.r_mag = torch.nn.Parameter(torch.randn(num_features,))
        self.b_mag = torch.nn.Parameter(torch.randn(num_features,))
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
        top_k_indices = torch.topk(hidden_layer, self.k, dim=-1).indices #should we find topk from hidden_layer or normalized_hidden_layer?
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
        kth_value = torch.topk(F.relu(encoder_output), k=self.k).values.min(dim=-1).values
        return suppress_lower_activations(encoder_output, kth_value, epsilon=self.epsilon)
    
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        if compute_loss:
            loss = self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream)
        else:
            loss = None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

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
        t = F.relu(t)
        remaining_mask = torch.ones(t.shape).to(device)
        while True:
            n = torch.sum(remaining_mask, dim=-1)
            n_or_2 = torch.maximum(n, 2*torch.ones(n.shape).to(device))
            bound_constant = 1 - self.a(n_or_2-1)/self.a(n_or_2)
            new_remaining = 1*(t*remaining_mask > torch.unsqueeze(torch.sum(t*remaining_mask, dim=-1) * bound_constant, dim=-1))
            finished_mask = torch.logical_or(torch.eq(remaining_mask, new_remaining), torch.unsqueeze(torch.eq(n, torch.ones(n.shape).to(device)), dim=-1)) #finished if, for each set of activations, either no updates this loop or n = 1
            if torch.sum(~finished_mask) == 0:
                break
            remaining_mask = new_remaining

        k = torch.sum(remaining_mask, dim=-1)
        k_or_plus_1_or_2 = torch.maximum(torch.unsqueeze(k, dim=-1) + 1-remaining_mask, 2*torch.ones(t.shape).to(device))
        bound_constant = 1 - self.a(k_or_plus_1_or_2-1)/self.a(k_or_plus_1_or_2)
        bound = (torch.unsqueeze(torch.sum(t*remaining_mask, dim=-1), dim=-1) + t * (1 - remaining_mask)) * bound_constant
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