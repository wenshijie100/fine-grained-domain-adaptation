
import torch

def PERSONLOSS(src, tar):


    batch_size=int(len(src)/2)

    src_person_loss=torch.cosine_similarity(src[:batch_size],src[batch_size:],dim=1,eps=1e-8)
    #print(len(src_person_loss))


    tgt_person_loss=torch.cosine_similarity(tar[:batch_size],tar[batch_size:],dim=1,eps=1e-8)
    #print(len(tgt_person_loss))
    #print(torch.mean(src_person_loss),torch.mean(tgt_person_loss))

    return -(torch.mean(src_person_loss)+torch.mean(tgt_person_loss))

