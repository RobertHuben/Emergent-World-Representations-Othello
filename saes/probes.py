import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from EWOthello.mingpt.dataset import CharDataset
from EWOthello.data.othello import OthelloBoardState

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
        
        return x, torch.tensor(board_state, dtype=torch.float32)


class LinearProbe(torch.nn.Module):
    def __init__(self, model_to_probe:torch.nn.Module, input_dim:int):
        self.model_to_probe = model_to_probe
        self.linear = torch.nn.Linear(input_dim, 64*3)

        for param in model_to_probe.parameters():
            param.requires_grad=False

    def forward(self, inputs, targets=None):
        activations = self.model_to_probe(inputs)
        logits = self.linear(activations)
        if targets:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return loss, logits
    
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

        print(f"Beginning probe training on {device}!")

        for epoch in range(num_epochs):
            train_dataloader=iter(torch.utils.data.DataLoader(train_probe_dataset, batch_size=batch_size, shuffle=True))
            print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_data} games.")
            
            for input_batch, label_batch in tqdm(train_dataloader):
                input_batch=input_batch.to(device)
                label_batch=label_batch.to(device)
                step+=1
                self.num_data_trained_on+=len(input_batch)
                optimizer.zero_grad(set_to_none=True)
                loss, logits = self.forward(input_batch, label_batch)
                loss.backward()
                optimizer.step()

                if step % report_on_batch_number==0:
                    self.print_evaluation(loss, eval_probe_dataset, step_number=step)
        else:
            self.print_evaluation(train_loss=loss, eval_dataset=eval_probe_dataset, step_number="Omega")
        self.eval()
    
