import torch
from torch.utils.data import Dataset
import itertools
import numpy as np

device='cuda' if torch.cuda.is_available() else 'cpu'

eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
enemy_own_modifier = np.concatenate([np.ones((1,64))*(-1)**i for i in range(60)],axis=0)


class ProbeDataset(Dataset):
    def __init__(self, games:list):
        self.games = games
        self.computed_data = [False] * len(games)
        
        self.max_game_length = max([len(game_seq) for game_seq in games])
        chars = sorted(list(set(list(itertools.chain.from_iterable(games)))) + [-100])
        self.char_to_index = {ch: i for i, ch in enumerate(chars)}

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
