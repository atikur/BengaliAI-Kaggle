import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import recall_score

class Combined_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2, x3, y1, y2, y3, reduction='mean'):
        return 0.5*F.cross_entropy(x1,y1,reduction=reduction) + 0.25*F.cross_entropy(x2,y2,reduction=reduction) + \
          0.25*F.cross_entropy(x3,y3,reduction=reduction)

def get_component_recall(pred, actual):
    pred_lbl = torch.argmax(pred, dim=1).cpu().numpy()
    actual_lbl = actual.cpu().numpy()
    return recall_score(actual_lbl, pred_lbl, average='macro')

def macro_recall_multi(pred_graphemes, true_graphemes, pred_vowels,true_vowels,pred_consonants,true_consonants):
    recall_grapheme = get_component_recall(pred_graphemes, true_graphemes)
    recall_vowel = get_component_recall(pred_vowels, true_vowels)
    recall_consonant = get_component_recall(pred_consonants, true_consonants)
    
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2.0, 1.0, 1.0])

    return final_score

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets

# rate = 0.75
def ohem_loss(cls_pred, cls_target, rate=0.75):
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets

def cutmix_criterion(preds1,preds2,preds3, targets, use_ohem=False):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    
    if use_ohem:
        criterion = ohem_loss
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    l1 = lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)
    l2 = lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)
    l3 = lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)
    return  0.5 * l1 + 0.25 * l2 + 0.25 * l3 

def mixup_criterion(preds1,preds2,preds3, targets, use_ohem=False):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    
    if use_ohem:
        criterion = ohem_loss
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    l1 = lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)
    l2 = lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)
    l3 = lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)
    return  0.5 * l1 + 0.25 * l2 + 0.25 * l3

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)