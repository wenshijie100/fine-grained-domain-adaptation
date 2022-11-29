import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
import resnet_orginal
import numpy as np
import math
class TransferNet(nn.Module):
    def __init__(self, num_class,myPI, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, cl_weight=[1,1,1,1],**kwargs):

        super(TransferNet, self).__init__()
        print("LLL:",kwargs)
        self.num_class = num_class
        #self.base_network = backbones.get_backbone(base_net)
        self.base_network=resnet_orginal.resnet50()
    
        #print("SUMMART",self.base_network)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            print("BOTTLENECK_LIST:",bottleneck_list)
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class,
            "my_person_item": myPI
        }
        print("TYPE:",transfer_loss_args)
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        #self.criterion = torch.nn.CrossEntropyLoss()
        
       # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1,2,3])).float().cuda(),size_average=True)
        self.criterion = torch.nn.CrossEntropyLoss(size_average=True)
        #print("SUMMART",self.base_network)
        #model(data_source, data_target, label_source,person_source,item_source,weight_source,weight_target,person_target,item_target,mu_list)
    def forward(self, source, target, source_label,person_source,item_source,weight_source,weight_target,person_target,item_target,mu_list,e):
        cp_source,cp_target=source,target
        source = self.base_network(source)
        target = self.base_network(target)
        weight_source=np.array(weight_source)
        weight_target=np.array(weight_target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label.long())
        # transfer
        if math.isnan(clf_loss) == True:
            print("CP_source:")
            print(cp_source)
            print("SOURCE:")
            print(source)
            print("CLF:")
            print(source_clf)
            print("LABEL:")
            print(source_label.long())

        kwargs = {}
        
        if self.transfer_loss == "threemmd":
            kwargs['source_label'] = source_label
            kwargs['source_weight'] = torch.from_numpy(weight_source).cuda()
            kwargs['source_person'] = person_source
            kwargs['source_item'] = item_source
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            kwargs['target_weight'] = torch.from_numpy(weight_target).cuda()
            kwargs['target_person'] = person_target
            kwargs['target_item'] = item_target
            kwargs['mu_list']=mu_list
            kwargs['epoch']=e

        elif self.transfer_loss == "twommd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            kwargs['source_weight'] = torch.from_numpy(weight_source).cuda()
            kwargs['target_weight'] = torch.from_numpy(weight_target).cuda()
            kwargs['mu_list']=mu_list
        elif self.transfer_loss == "mmd":
            #kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            #kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
           
        elif self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "pmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            kwargs['source_weight'] = torch.from_numpy(weight_source).cuda()
            kwargs['target_weight'] = torch.from_numpy(weight_target).cuda()
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        else :
            print("LOSS TYPE ERROT")
        #print("SOURCE,TARGET",source, target)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass