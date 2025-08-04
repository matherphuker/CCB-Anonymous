import torch
import torch.nn.functional as F
import math

def get_entropy_margin_confidence(probs, alpha=0.5, eps=1e-8):
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    max_entropy = math.log(probs.shape[-1])
    entropy_score = 1 - (entropy / max_entropy)  
    top2 = torch.topk(probs, k=2).values
    margin_score = torch.clamp(top2[0] - top2[1], 0, 1)
    confidence = alpha * entropy_score + (1 - alpha) * margin_score
    return confidence
