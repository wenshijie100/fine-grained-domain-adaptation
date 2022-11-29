from torchvision import datasets, transforms
import torch
from collections import Counter
import random

class Data_Bank:
    def __init__(self,X_train,Y_train,W_train,P_train,I_train):
        self.person_number=max(P_train)+1
        #print("MIN_PERSON:",min(P_train))
        self.X_data=[ [] for i in range(self.person_number) ]
        self.Y_data=[ [] for i in range(self.person_number) ]
        self.W_data=[ [] for i in range(self.person_number) ]
        self.P_data=[ [] for i in range(self.person_number) ]
        self.I_data=[ [] for i in range(self.person_number) ]
        for i in range(len(X_train)):
            #print(P_train[i],self.person_number)
            self.X_data[P_train[i]].append(X_train[i])
            self.Y_data[P_train[i]].append(Y_train[i])
            self.W_data[P_train[i]].append(W_train[i])

    def get_another(self,P,I):
        number=I 
        cnt=0
        while(number==I and cnt<3):
            cnt=cnt+1
            number=random.randint(0,len(self.X_data[P])-1)
        
        #print("C:",self.X_data[P][number])
        #print("B:",len(self.X_data[P][number]),len(self.P_data[P][number]),len(self.W_data[P][number])) 
        '''
        try:
            a,b,c=self.X_data[P][number],self.Y_data[P][number],self.W_data[P][number]

        except Exception as e:
            print(e.args)
            print("A:",P,number,self.person_number,len(self.X_data),len(self.X_data[P]),len(self.W_data[P]),len(self.Y_data[P]))  
            a=self.X_data[P][number]
            b=self.Y_data[P][number]
            c=self.W_data[P][number]
        '''
        return self.X_data[P][number],self.Y_data[P][number],self.W_data[P][number]


class GetTrainLoader(torch.utils.data.Dataset):
	
   
    def __init__(self, data_root, data_label,data_weight,data_person,data_item,train):
        self.data = data_root
        self.label = data_label
        self.weight=data_weight
        self.person=data_person
        self.item=data_item
        self.train=train
        self.databank=Data_Bank(data_root.copy(), data_label.copy(),data_weight.copy(),data_person.copy(),data_item.copy())
        
    
    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]
        weight = self.weight[index] 
        person = self.person[index] 
        item=  self.item[index] 
        data2,label2,weight2 = self.databank.get_another(person,item)
        return data, label, weight,person,item,data2,label2,weight2
   
    def __len__(self):
        #int("AAA:",len(self.data))
        return len(self.data)

class GetTestLoader(torch.utils.data.Dataset):
	
   
    def __init__(self, data_root, data_label,data_weight,data_person,data_item,train):
        self.data = data_root
        self.label = data_label
        self.weight=data_weight
        self.person=data_person
        self.item=data_item
        self.train=train
        
    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]
        weight = self.weight[index] 
        person = self.person[index] 
        item=  self.item[index] 
        return data, label, weight,person,item

    def __len__(self):
        #int("AAA:",len(self.data))
        return len(self.data)

def load_data_train(datax,datay,dataw,datap,datai, batch_size, train, num_workers=0, **kwargs):
    
    #data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    
    data= GetTrainLoader(datax,datay,dataw,datap,datai,train)
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = 4
    return data_loader, n_class


def load_data_test(datax,datay,dataw,datap,datai, batch_size, train, num_workers=0, **kwargs):
    
    #data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    
    data= GetTestLoader(datax,datay,dataw,datap,datai,train)
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = 4
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