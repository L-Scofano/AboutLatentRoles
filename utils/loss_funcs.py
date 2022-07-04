import torch

def MAD(batch_pred,batch_gt): 
    batch_pred = batch_pred.contiguous().view(-1,2)
    batch_gt = batch_gt.contiguous().view(-1,2)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def FAD(batch_pred,batch_gt): 
    batch_pred = torch.unsqueeze(batch_pred[:,-1],1).contiguous().view(-1,2)
    batch_gt = torch.unsqueeze(batch_gt[:,-1],1).contiguous().view(-1,2)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))
    