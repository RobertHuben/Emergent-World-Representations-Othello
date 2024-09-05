import torch
device='cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def vectorized_f1_score(scores:torch.tensor, labels:torch.tensor, thresholds:torch.tensor) -> torch.tensor:
    '''
    inputs:
        scores: float tensor of shape (N, K) where N is the number of items in your dataset and K is the number of input scores each has
        labels: binary tensor of shape (N)
        thresholds: float tensor of shape (T)
    returns:
        f1_scores: tensor of shape (K, T), where the (k,t)th entry is the F1 score of using score k at threshold t
    '''
    while len(scores.shape)<3:
        scores=scores.unsqueeze(-1)
    labels=labels.to(device=device, dtype=torch.float32)
    indicated_positives=(scores>=thresholds.unsqueeze(0).unsqueeze(0)).to(device=device, dtype=torch.float32) #shape (N,K,T)
    true_positives=torch.tensordot(indicated_positives,labels, dims=([0],[0]))
    false_positives=torch.tensordot(indicated_positives,(1-labels), dims=([0],[0]))
    false_negatives=torch.tensordot(1-indicated_positives,labels, dims=([0],[0]))
    f1_scores=(2*true_positives)/(2*true_positives+false_negatives+false_positives)
    masked_f1_scores=torch.where((true_positives+false_negatives+false_positives)==0, 0, f1_scores)
    return masked_f1_scores


