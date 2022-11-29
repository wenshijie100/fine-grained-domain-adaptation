from torchvision import datasets, transforms
import torch
from collections import Counter
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
   
    def __init__(self, data_root, data_label,data_weight,data_person,data_item,train):
        self.data = data_root
        self.label = data_label
        self.weight=data_weight
        self.person=data_person
        self.item=data_item
        
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        weight = self.weight[index] 
        person = self.person[index] 
        item=  self.item[index] 
        return data, labels, weight,person,item
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        #int("AAA:",len(self.data))
        return len(self.data)

def load_data(datax,datay,dataw,datap,datai, batch_size, train, num_workers=0, **kwargs):
    
    #data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    
    data= GetLoader(datax,datay,dataw,datap,datai,train)
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = 4
    return data_loader, n_class

def load_data2(datax,datay, batch_size, train, num_workers=0, **kwargs):
    
    #data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    
    data= GetLoader(datax,datay,train)
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class =4
    return data_loader, n_class



def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    print("DATASET.SHAPE:",dataset,infinite_data_loader)
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))
        #self.batch_size=batch_size

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return  0
        # Always return 0