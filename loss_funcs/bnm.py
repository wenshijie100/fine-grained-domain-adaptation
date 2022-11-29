import torch

def BNM(src, tar):
    """ Batch nuclear-norm maximization, CVPR 2020.
    tar: a tensor, softmax target output.
    NOTE: this does not require source domain data.
    """
    #print(tar)
    #print("LEN:",len(tar))
    _, out, _ = torch.svd(tar)
    loss = -torch.mean(out)
    return loss