import torch 
from torch.nn import functional as F
import logging
import numpy as np

from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from saes.sae_template import SAETemplate

logger = logging.getLogger(__name__)
device='cuda' if torch.cuda.is_available() else 'cpu'

class SAEAnthropic(SAETemplate):

    def __init__(self, gpt:GPTforProbing, num_features:int, sparsity_coefficient:float, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.sparsity_coefficient=sparsity_coefficient
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

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
        decoder_column_norms=self.decoder.norm(dim=1)
        return torch.mean(hidden_layer*decoder_column_norms)
    
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

class MultiHeadedTopKSAE(SAETemplate):

    def __init__(self, gpt:GPTforProbing, num_features:int, sparsity:int, num_heads:int, decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.sparsity=sparsity
        self.num_heads=num_heads
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

    def activation_function(self, encoder_output):
        activations = F.relu(encoder_output)
        attention_by_head=activations.reshape((activations.shape[0],activations.shape[1], self.num_heads, self.num_features//self.num_heads))
        kth_value = torch.topk(attention_by_head, k=self.sparsity//self.num_heads).values.min(dim=-1).values
        masked_activations=suppress_lower_activations(attention_by_head, kth_value, epsilon=0, mode='relative')
        return masked_activations.reshape(activations.shape)
    
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
        loss= self.reconstruction_error(residual_stream, reconstructed_residual_stream) if compute_loss else None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"Number of heads: {self.num_heads}", f"Sparsity (total): {self.sparsity}"]

#supported variants: mag_in_aux_loss, relu_only
#setting no_aux_loss=True implements a gated sae in a different way from the paper that makes more sense to me
#currently uses tied weights only
#to try: untied weights original version, as well as using sigmoid instead of step function for training to avoid aux_loss
class Gated_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, no_aux_loss=False, sigmoid_act_coeff=False, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.b_gate = self.encoder_bias #just renaming to make this more clear
        self.r_mag = torch.nn.Parameter(torch.zeros((num_features)))
        self.b_mag = torch.nn.Parameter(torch.zeros((num_features)))
        self.no_aux_loss = no_aux_loss
        self.sigmoid_act_coeff = sigmoid_act_coeff

    def forward(self, residual_stream, compute_loss=False):
        if self.no_aux_loss:
            encoder = F.normalize(self.encoder, p=2, dim=1)
        else:
            encoder = self.encoder
        encoding = (residual_stream - self.decoder_bias) @ encoder
        if self.no_aux_loss:
            if self.sigmoid_act_coeff:
                features_to_use = torch.sigmoid(self.sigmoid_act_coeff*(encoding + self.b_gate))
                hidden_layer = F.relu(features_to_use * encoding * torch.exp(self.r_mag) + self.b_mag)
            else:
                features_to_use = F.relu(encoding + self.b_gate)
                hidden_layer = F.relu(features_to_use * torch.exp(self.r_mag) + self.b_mag) #is b_mag really necessary here?
        else:
            hidden_layer_before_gating = F.relu(encoding * torch.exp(self.r_mag) + self.b_mag)
            hidden_layer = ((encoding + self.b_gate) > 0) * hidden_layer_before_gating
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1)
        reconstructed_residual_stream = hidden_layer @ normalized_decoder + self.decoder_bias

        if compute_loss:
            reconstruction_loss=self.reconstruction_error(residual_stream, reconstructed_residual_stream)

            if self.no_aux_loss:
                hidden_layer_without_gating_or_mag = features_to_use
            else:
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
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, anneal_proportion: float, p_end=0.2, queue_length=10, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, sparsity_coefficient, decoder_initialization_scale)
        self.p = 1
        self.anneal_proportion = anneal_proportion
        self.p_end = p_end
        self.queue = ActivationQueue(queue_length)
    
    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.anneal_start = round(num_steps*(1-self.anneal_proportion))
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
        return (hidden_layer**self.p).mean()
    
class Gated_P_Annealing_SAE(P_Annealing_SAE, Gated_SAE):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, anneal_proportion: float, p_end=0.2, queue_length=10, no_aux_loss=False, decoder_initialization_scale=0.1):
        P_Annealing_SAE.__init__(self, gpt, num_features, sparsity_coefficient, anneal_proportion, p_end, queue_length, decoder_initialization_scale)
        Gated_SAE.__init__(self, gpt, num_features, sparsity_coefficient, no_aux_loss=no_aux_loss)

    def forward(self, residual_stream, compute_loss=False):
        return Gated_SAE.forward(self, residual_stream, compute_loss)

    def sparsity_loss_function(self, hidden_layer):
        return P_Annealing_SAE.sparsity_loss_function(self, hidden_layer)

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
    
    def report_model_specific_eval_results(self, hidden_layers=None):
        return [f"    Average activations over epsilon: {torch.sum(hidden_layers > self.epsilon)/hidden_layers[..., 0].numel():.1f}"]
    
class Gated_Smoothed_L0_SAE(Smoothed_L0_SAE, Gated_SAE):
    def __init__(self, gpt: GPTforProbing, num_features: int, sparsity_coefficient: float, epsilon:float, delta:float, no_aux_loss=False):
        Smoothed_L0_SAE.__init__(self, gpt, num_features, sparsity_coefficient, epsilon, delta)
        Gated_SAE.__init__(self, gpt, num_features, sparsity_coefficient, no_aux_loss=no_aux_loss)

    def forward(self, residual_stream, compute_loss=False):
        return Gated_SAE.forward(self, residual_stream, compute_loss)

    def sparsity_loss_function(self, hidden_layer):
        return Smoothed_L0_SAE.sparsity_loss_function(self, hidden_layer)
    
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

#suppression_mode can be "relative" or "absolute"
class Leaky_Topk_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, num_features: int, epsilon: float, k:int, suppression_mode="relative", decoder_initialization_scale=0.1):
        super().__init__(gpt=gpt, num_features=num_features)
        self.epsilon = epsilon
        self.k=k
        self.suppression_mode = suppression_mode
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

    def activation_function(self, encoder_output):
        activations = F.relu(encoder_output)
        kth_value = torch.topk(activations, k=self.k).values.min(dim=-1).values
        return suppress_lower_activations(activations, kth_value, epsilon=self.epsilon, mode=self.suppression_mode)
    
    def forward(self, residual_stream, compute_loss=False):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (loss, residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        normalized_encoder = F.normalize(self.encoder, p=2, dim=1) #normalize columns
        normalized_decoder = F.normalize(self.decoder, p=2, dim=1) #normalize columns
        hidden_layer=self.activation_function((residual_stream - self.decoder_bias) @ normalized_encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ normalized_decoder + self.decoder_bias
        loss= self.reconstruction_error(residual_stream, reconstructed_residual_stream) if compute_loss else None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def report_model_specific_features(self):
        return [f"k (sparsity): {self.k}", f"Epsilon (leakyness): {self.epsilon}"]

    def post_copying_update(self, original_sae, new_feature_indices):
        '''
        if there are fewer features than k, make k equal the number of features
        '''
        if len(new_feature_indices)<self.k:
            self.k=len(new_feature_indices)

class K_Annealing_Leaky_Topk_SAE(Leaky_Topk_SAE):
    def __init__(self, gpt: GPTforProbing, num_features: int, epsilon: float, k_start: int, anneal_start: int, k_end: int, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features, epsilon, k_start, decoder_initialization_scale)
        self.k_start = k_start
        self.anneal_start = anneal_start
        self.k_end = k_end
        self.k_continuous = k_start

    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.k_step = (1 - self.k_end)/(num_steps - self.anneal_start)
        return
    
    def after_step_update(self, hidden_layer=None, step=None):
        if step >= self.anneal_start:
            self.k_continuous += self.k_step
            self.k = round(self.k_continuous)
        return
    
    def report_model_specific_eval_results(self, hidden_layers=None):
        [f"    Average activations over epsilon: {torch.mean(hidden_layers > self.epsilon):.1f}"]
    

class Epsilon_Annealing_Leaky_Topk_SAE(Leaky_Topk_SAE):
    def __init__(self, gpt: GPTforProbing, num_features: int, k: int, epsilon_start: int, epsilon_end: int, decoder_initialization_scale=0.1, annealing_mode='linear'):
        assert annealing_mode in ['linear', 'exponential']
        super().__init__(gpt, num_features, epsilon=epsilon_start, k=k, decoder_initialization_scale=decoder_initialization_scale)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.annealing_mode=annealing_mode

    def training_prep(self, train_dataset=None, eval_dataset=None, batch_size=None, num_epochs=None):
        num_steps = len(train_dataset) * num_epochs / batch_size
        self.epsilon_step_linear = (self.epsilon_end-self.epsilon_start)/num_steps
        self.epsilon_step_exponential = (self.epsilon_end/self.epsilon_start)**(1/num_steps)
    
    def after_step_update(self, hidden_layer=None, step=None):
        if self.annealing_mode=='linear':
            self.epsilon+=self.epsilon_step_linear
        elif self.annealing_mode=='exponential':
            self.epsilon*=self.epsilon_step_exponential
    
    # def report_model_specific_eval_results(self, hidden_layers=None):
    #     [f"    Average activations over epsilon: {float(torch.mean(hidden_layers > self.epsilon)):.1f}"]
    
class Random_Leaky_Topk_SAE(Leaky_Topk_SAE):
    '''
    Currently supports poisson and normal distribution for k
    During training, each forward pass uses a random value of k from the distribution
    After training, forward uses k_mean for k by default, but you can override this by providing a value for k_eval
    '''
    def __init__(self, gpt: GPTforProbing, num_features: int, epsilon: float, k_mean: int, distribution="poisson", k_std=None, decoder_initialization_scale=0.1):
        assert distribution in ["poisson", "normal"], "Distribution not recognized.  Only supports poisson and normal distributions."
        if distribution == "normal":
            assert k_std, "Need to input a standard deviation to use a normal distribution."
        super().__init__(gpt, num_features, epsilon, k_mean, decoder_initialization_scale)
        self.k_mean = k_mean
        self.distribution = distribution
        self.k_std = k_std
        self.rng = np.random.default_rng()

    def forward(self, residual_stream, compute_loss=False, k_eval=None):
        if self.training:
            if self.distribution == "poisson":
                self.k = self.rng.poisson(self.k_mean)
            elif self.distribution == "normal":
                self.k = round(self.rng.normal(self.k_mean, self.k_std))
                if self.k < 1:
                    self.k = 1
                elif self.k > self.num_features:
                    self.k = self.num_features
        elif k_eval:
            self.k = k_eval
        return super().forward(residual_stream, compute_loss)
    
    def eval(self):
        self.k = self.k_mean
        return super().eval()
    
    def report_model_specific_eval_results(self, hidden_layers=None):
        [f"    Average activations over epsilon: {torch.mean(hidden_layers > self.epsilon):.1f}"]

class Top_L1_Proportion_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, num_features: int, L1_proportion_to_remove: float, decoder_initialization_scale=0.1):
        super().__init__(gpt, num_features)
        self.proportion = L1_proportion_to_remove
        self.encoder, self.encoder_bias, self.decoder, self.decoder_bias=self.create_linear_encoder_decoder(decoder_initialization_scale)

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
        loss= self.reconstruction_error(residual_stream, reconstructed_residual_stream) if compute_loss else None
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream
    
    def activation_function(self, encoder_output):
        activations = F.relu(encoder_output)
        sorted, indices = torch.sort(activations)
        sum_bounds = self.proportion*torch.sum(activations, dim=-1)
        partial_sums = torch.einsum("ij,...j->...i", self.lower_triangle_ones, sorted)
        selected_mask_out_of_order = partial_sums > sum_bounds.unsqueeze(-1)
        selected_mask = selected_mask_out_of_order.gather(-1, indices.argsort(-1))
        final = selected_mask * activations

        #for testing to make sure this is implemented correctly
        """ bound = torch.max((~selected_mask) * activations, dim=-1).values
        assert torch.equal(selected_mask, (activations > bound.unsqueeze(-1))) #this might not be true if there are identical activations
        assert (~(torch.sum(final, dim=-1) >= (1-self.proportion)*torch.sum(activations, dim=-1))).sum() == 0
        assert (~(torch.sum((~selected_mask) * activations, dim=-1) <= self.proportion*torch.sum(activations, dim=-1))).sum() == 0
 """
        return final

class Dimension_Reduction_SAE(SAEAnthropic):
    def __init__(self, gpt: GPTforProbing, num_features: int, start_index: int, start_proportion: float, end_proportion: float, epsilon: float):
        super().__init__(gpt, num_features)
        self.start_index = start_index
        self.start_proportion = start_proportion
        self.end_proportion = end_proportion
        self.epsilon = epsilon
        self.activation_f = reduce_dimensions_activation(Expand(self.start_index, self.start_proportion, self.end_proportion), max_n = self.num_features, epsilon=self.epsilon)

    def activation_function(self, encoder_output):
        return self.activation_f(encoder_output)
    
    #need to override loss function of SAEAnthropic, or else inherit from SAETemplate
    
    def report_model_specific_eval_results(self, hidden_layers=None):
        [f"    Average activations over epsilon: {torch.mean(hidden_layers > self.epsilon):.1f}"]
    
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
        bad_bound_mask = bound <= 0 #to make sure we don't divide by 0
        return above_only + (~bad_bound_mask)*epsilon/(bound+bad_bound_mask) * below_only
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
