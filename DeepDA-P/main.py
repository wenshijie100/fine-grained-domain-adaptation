import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from torchsummary import summary
import my_data_to_picture as mydatatopicture
import my_person_item as mypersonitem
import math
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--tname', type=str, required=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=50, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    print("JUST DO IT")
    #folder_src = os.path.join(args.data_dir, args.src_domain)
    #folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    if args.tname=="transfer+finetune":
        X_train,Y_train,W_train,P_train,I_train,L_train,\
        X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2,\
        X_test,Y_test,W_test,P_test,I_test,L_test,\
        X_test2,Y_test2,W_test2,P_test2,I_test2,L_test2\
        =mydatatopicture.run_finetune( args.src_domain, args.tgt_domain, args.tname)
        print("L_TEST:",len(L_test),len(L_test2),len(P_test2),len(P_test))
        print("read_finish!")
        print("DATA_PROFILE  ","train:",len(X_train),"train2:",len(X_train2),"test:",len(X_test),"test2:",len(X_test2))
        source_loader, n_class = data_loader.load_data(
            X_train,Y_train,W_train,P_train,I_train,args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("FINISH TRAIN")
        target_train_loader, T_class = data_loader.load_data(
            X_train2,Y_train2,W_train2,P_train2,I_train2, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("TT_class",T_class)
        target_test_loader, _ = data_loader.load_data(
            X_test,Y_test,W_test,P_test,I_test,args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
        
        finetune_source_loader, n_class = data_loader.load_data(
            X_test2,Y_test2,W_test2,P_test2,I_test2,args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("FINISH TRAIN")
        finetune_target_train_loader, T_class = data_loader.load_data(
            X_test,Y_test,W_test,P_test,I_test,args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("TT_class",T_class)
        finetune_target_test_loader, _ = data_loader.load_data(
            X_test,Y_test,W_test,P_test,I_test,args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
        myPI=mypersonitem.PersonItem(max(P_train)+1,max(I_train)+1,max(P_test)+1,max(I_test)+1)
        return source_loader, target_train_loader, target_test_loader, n_class,L_test,\
               finetune_source_loader, finetune_target_train_loader, finetune_target_test_loader, L_test2, myPI
          
    else :
        X_train,Y_train,W_train,P_train,I_train,L_train,\
        X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2,\
        X_test,Y_test,W_test,P_test,I_test,L_test\
        =mydatatopicture.run_experiment( args.src_domain, args.tgt_domain, args.tname)
        #print("read_finish!",max(W_train),max(W_test),np.mean(W_train),np.mean(W_test),min(W_train),min(W_test))
        '''
        Y_train=Y_train-1
        Y_train2=Y_train2-1
        Y_test=Y_test-1
        '''
        W_train=W_train/np.mean(W_train)
        W_train2=W_train2/np.mean(W_train2)
        W_test=W_test/np.mean(W_test)
        #print("read_finish!",max(W_train),max(W_test),np.mean(W_train),np.mean(W_test),min(W_train),min(W_test))
        print("DATA_PROFILE  ","train:",len(X_train),"train2:",len(X_train2),"test:",len(X_test))
        source_loader, n_class = data_loader.load_data(
            X_train,Y_train,W_train,P_train,I_train,args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("FINISH TRAIN")
        target_train_loader, T_class = data_loader.load_data(
            X_train2,Y_train2,W_train2,P_train2,I_train2, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
        #print("TT_class",T_class)
        target_test_loader, _ = data_loader.load_data(
            X_test,Y_test,W_test,P_test,I_test,args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
        #print("EEEEEEEE:",max(P_train),max(I_train),max(P_test),max(I_test))
        myPI=mypersonitem.PersonItem(max(P_train)+1,max(I_train)+1,max(P_test)+1,max(I_test)+1)
        return source_loader, target_train_loader, target_test_loader, n_class,L_test,_,_,_,_,myPI

def get_model(cl_weight,myPI,args):
    model = models.TransferNet( args.n_class ,myPI=myPI,transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,cl_weight=cl_weight, use_bottleneck=args.use_bottleneck).to(args.device)
    print(model)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    print("initial_lr",initial_lr)
    params = model.get_parameters(initial_lr=initial_lr)
    #print("params",params)
    #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    optimizer=torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    return optimizer

   

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler 

def get_finetune_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * 0.001 * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader,cl_weight, args):
    conf_matrix = torch.zeros(4,4)
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cl_weight)).float().to(args.device),size_average=True)
    #len_target_dataset = len(target_test_loader.dataset)
    len_target_dataset=0
    cnt=0
    
    with torch.no_grad():
        for data, target,weight,person,item in target_test_loader:
            #print("EFG",len(data),len(target),target,np.array(data).shape)
            data, target = data.to(args.device), target.to(args.device)
            cnt=cnt+1  
            s_output = model.predict(data)
            loss = criterion(s_output, target.long())
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
          
            confusion_matrix(pred,target,conf_matrix)
            len_target_dataset+=len(pred)
            #print(cnt)
            #print("PREDHA:",pred[:10])
           #print("TARGET:",target[:10])
    
    print("TTL",len_target_dataset)
    acc = 100. * correct / len_target_dataset
    print("correct:",correct,len_target_dataset)
    print(conf_matrix)
    conf_matrix=np.array(conf_matrix)
    cnt_f1(conf_matrix)
    return acc, test_loss.avg
best_person_f1_ma,best_person_f1_mi,best_person_recall,best_person_acc,best_f1_ma,best_f1_mi,best_recall,best_acc_mi,best_acc_ma=0,0,0,0,0,0,0,0,0
matrix_sample,matrix_person=[],[]

def test_person(model, target_test_loader, L_test,cl_weight,args):
    usepmmd=False
    if(args.transfer_loss=="pmmd" or args.transfer_loss=="threemmd"):
        print("USE PMMD")
        usepmmd=True
    pre_label=[]
    true_label=[]
    model.eval()
    test_loss = utils.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cl_weight)).float().to(args.device),size_average=True)
    #len_target_dataset = len(target_test_loader.dataset)
    len_target_dataset=0
    cnt=0
    n_person=len(L_test)

    label_true_person=[-1 for i in range(n_person)]
    label_pred_person=[-1 for i in range(n_person)]
    label_pred=[[] for i in range(n_person)]
    label_true=[[] for i in range(n_person)]
    weight_pred=[[] for i in range(n_person)]
    print("n_person",n_person)
    with torch.no_grad():
        
        cnt=0
        for data, target,weight,person,item in target_test_loader:
            cnt=cnt+1
            #if cnt%100==0:
               # print(cnt)
            #print(data)
            data, target = data.to(args.device), target.to(args.device)
            #print("CCC")
            s_output = model.predict(data)
            loss = criterion(s_output, target.long())
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            len_target_dataset+=len(pred)
           # print("PERSON:",person)
            #print("CNT:",pred)
            #print("CNT:",target)
            
            for i in range(len(pred)):
                #print("ERROR",person[i],n_person)
                label_true_person[person[i]]=int(target[i].cpu().numpy())
                label_pred[person[i]].append(int(pred[i].cpu().numpy()))
                label_true[person[i]].append(int(target[i].cpu().numpy()))
                weight_pred[person[i]].append(weight[i].cpu().numpy())
   # print(n_person)
   # print(label_true_person)
    #print("n_person",n_person)
    y_true=[]
    y_pred=[]
    #print("BBC")
    for i in range(n_person):
        #print(i,len(label_pred[i]))
        for j in range(len(label_pred[i])):
            y_true.append(label_true[i][j])
            y_pred.append(label_pred[i][j])
    #print("ABC:",len(y_true),len(y_pred))
    
    #print(y_pred) 
    test_f1=output(model,y_true, y_pred,"SAMPLE",args)
    '''
    for i in range(n_person):
        label_sum=0.0
        weight_sum=0.0
        t=[0 for i in range(args.n_class)]
        max_label=0
        for j in range(len(label_pred[i])):
            w=weight_pred[i][j]
            if usepmmd==False:
                w=1
            t[label_pred[i][j]]+=w
        label_pred_person[i]=np.argmax(t)
    y_true=label_true_person
    y_pred=label_pred_person
    '''
    #print("MIN:",min(y_true))
    #test_f1=output(model,y_true, y_pred,"PERSON",args)
    return  test_f1,test_loss.avg
def init():
    global best_person_f1_ma
    global best_person_f1_mi
    global best_person_recall
    global best_person_acc
    
    global best_f1_ma
    global best_f1_mi
    global best_recall
    global best_acc_ma
    global best_acc_mi
    
    best_person_f1_ma=0
    best_person_f1_mi=0
    best_person_recall=0
    best_person_acc=0
    
    best_f1_ma=0
    best_f1_mi=0
    best_recall=0
    best_acc_ma=0
    best_acc_mi=0

def input(args):
    path="./pth/"+str(args.src_domain)+"+"+str(args.tgt_domain)+"/"+str(args.config.split("/")[0])+"_"+str(args.transfer_loss)+"_"+str(args.tname)+"_"+str(args.seed)+"_"
    print("IN_PP:",path)
    model=torch.load(path+"F1.pth")
    return model
def load_transfer_to_finetune(args):
    path="./pth/"+str(args.src_domain)+"+"+str(args.tgt_domain)+"/"+str(args.config.split("/")[0])+"_"+str(args.transfer_loss)+"_"+"transfer"+"_"+str(args.seed)+"_"
    print("IN_PP:",path)
    model=torch.load(path+"F1.pth")
    return model


def output(model,y_true, y_pred,name,args):
    #print("MIN:",min(y_true))
    global best_person_f1_ma
    global best_person_f1_mi
    global best_person_recall
    global best_person_acc
    
    global best_f1_ma
    global best_f1_mi
    global best_recall
    global best_acc_ma
    global best_acc_mi

    global matrix_person
    global matrix_sample
    print(name)
    acc_ma=metrics.precision_score(y_true, y_pred, average='macro')
    recall=metrics.recall_score(y_true, y_pred, average='macro')
    f1_ma=2*acc_ma*recall/(acc_ma+recall)
    acc_mi=metrics.precision_score(y_true, y_pred, average='micro')
    recall=metrics.recall_score(y_true, y_pred, average='micro')
    f1_mi=2*acc_mi*recall/(acc_mi+recall)
   
    #print("ACC:",acc,"RECALL:",recall,"F1-W:",metrics.f1_score(y_true, y_pred, average='weighted'),"F1:",f1)
    path="./pth/"+str(args.src_domain)+"+"+str(args.tgt_domain)+"/"+str(args.config.split("/")[0])+"_"+str(args.transfer_loss)+"_"+str(args.tname)+"_"+str(args.seed)+"_"
    if name=='PERSON':
        print("PP:",path)
        if best_person_f1_ma<f1_ma:
            best_person_f1_ma=f1_ma
            matrix_person=confusion_matrix(y_true, y_pred)
        if best_person_f1_mi<f1_mi:
            best_person_f1_mi=f1_mi
            matrix_person=confusion_matrix(y_true, y_pred)
        '''
        if best_person_recall<recall:
            torch.save(model, path+"Recall.pth")
        best_person_f1=max( best_person_f1,2*acc*recall/(acc+recall))
        best_person_recall=max(best_person_recall,recall)
        best_person_acc=max(best_person_acc,acc)
        '''
        print("BEST_F1_PERSON_MI:",best_person_f1_mi,"BEST_F1_PERSON_MA:",best_person_f1_ma)
    if name=='SAMPLE':
       
        if best_f1_ma<f1_ma:
            best_f1_ma=f1_ma
            torch.save(model, path+"maF1.pth")
            matrix_person=confusion_matrix(y_true, y_pred)
        if best_f1_mi<f1_mi:
            best_f1_mi=f1_mi
            torch.save(model, path+"miF1.pth")
            matrix_sample=confusion_matrix(y_true, y_pred)

        if best_acc_ma<acc_ma:
            best_acc_ma=acc_ma
            
        if best_acc_mi<acc_mi:
            best_acc_mi=acc_mi
        '''
        best_f1=max(best_f1,2*acc*recall/(acc+recall))
        best_recall=max(best_recall,recall)
        best_acc=max(best_acc,acc)
        '''
        print("F1_MI:",f1_mi,"F1_MA:",f1_ma,"ACC_MI:",acc_mi,"F1_MA:",acc_ma)
        print("BEST_F1_MI:",best_f1_mi,"BEST_F1_MA:",best_f1_ma,"BEST_ACC_MI:",best_acc_mi,"BEST_F1_MA:",best_acc_ma)
    print(confusion_matrix(y_true, y_pred))
    return best_f1_ma
    '''
    print("ERROR:")
    print( max(y_pred),min(y_pred))
    print( max(y_true),min(y_true))
    '''

def train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, lr_scheduler,cl_weight, args):
  
    print(torch.__version__)
    print(torch.cuda.is_available())
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    print("LEN:",len_source_loader,len_target_loader)
    print("LOADER:",source_loader,target_train_loader)
    
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch ==0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)
    print("N:",n_batch)
    best_acc = 0
    stop = 0
    log = []
    test_person(model, target_test_loader,L_test,cl_weight,args)
    for e in range(1, args.n_epoch+1):
        print("trian begin")
        model.train()
        print("trian end")
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
            #print("ABC")
        #criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cl_weight)).float(),size_average=True)
        for i in range(n_batch):
            data_source, label_source,weight_source,person_source,item_source = next(iter_source) # .next()
            #print(label_source)
            #if i==0:
                 #print("ABC",len(data_source),len(label_source),label_source,np.array(data_source).shape,data_source[0][0][0],data_source[1][0][0])
            data_target, _,weight_target,person_target,item_target = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            mu_list=torch.from_numpy(np.array([1,1,1,1])).to(args.device)
            clf_loss, transfer_loss = model(data_source, data_target, label_source,person_source,item_source,weight_source,weight_target,person_target,item_target,mu_list,e)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            if math.isnan(loss) == True:
                break
            #print("LOSS",loss,clf_loss,args.transfer_loss_weight,transfer_loss)
          
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.9)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        if math.isnan(train_loss_total.avg) == True:
            break
        stop += 1
        #test_acc, test_loss = test(model, target_test_loader, args)
        test_acc,test_loss = test_person(model, target_test_loader,L_test,cl_weight,args)
        #info += ', test_loss {:4f}'.format(test_loss)
        #np_log = np.array(log, dtype=float)
        #np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
    
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        
        print(info)
    #print('Transfer result: {:.4f}'.format(best_acc))
    print(matrix_person)
    print(matrix_sample)
def train_finetune(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, lr_scheduler,cl_weight, args):
  
    print(torch.__version__)
    print(torch.cuda.is_available())
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch+1):
        model.train()
        print("#####################################")
        test_person(model, target_test_loader,L_test,cl_weight,args)
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        #criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(cl_weight)).float(),size_average=True)
        for i in range(n_batch):
            data_source, label_source,weight_source,_,_ = next(iter_source) # .next()
            #if i==0:
                 #print("ABC",len(data_source),len(label_source),label_source,np.array(data_source).shape,data_source[0][0][0],data_source[1][0][0])
            data_target, _,weight_target,person_source,item_source = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            
            clf_loss, transfer_loss = model(data_source, data_target, label_source,weight_source,weight_target)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            #print("LOSS",loss,clf_loss,args.transfer_loss_weight,transfer_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            #print("LRRRRRR:",lr_scheduler.get_last_lr()[0])
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        #test_acc, test_loss = test(model, target_test_loader, args)
        test_loss = test_person(model, target_test_loader,L_test,cl_weight,args)
        info += ', test_loss {:4f}'.format(test_loss)
        #np_log = np.array(log, dtype=float)
        #np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        '''
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        '''
        print(info)

def main():
    
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args.transfer_loss,args.seed,args.src_domain,args.tgt_domain,args.config)
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class,L_test,\
    finetune_source_loader, finetune_target_train_loader, finetune_target_test_loader, L_test2, myPI = load_data(args)
    print("CLASS:",n_class,len(L_test))
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    cl_weight=[1 for i in range(n_class)]
    model = get_model(cl_weight, myPI,args)
    optimizer = get_optimizer(model, args)
    #print("ABC",optimizer.state_dict()['param_groups'][0]['lr'])
    #print("args.lr_scheduler:",args.lr_scheduler)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    #print("SSS:", scheduler)
    if args.tname=="self":
        print("self_world")
        args.transfer_loss_weight=0
        train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        print("ARG:",args)
    elif args.tname=="transfer":
        print("transfer_world")
        train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        print("ARG:",args)
    elif args.tname=="transfer_test":
        print("transfer_test")
        #train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        model = input(args)
        test_person(model, target_test_loader, L_test,cl_weight,args)
        print("ARG:",args)
    else:
        print("finetune_world")
        #train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        #train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        model = input(args)
       
        #args.lr_scheduler=None
        optimizer = get_optimizer(model, args)
        if args.lr_scheduler :
            scheduler = get_finetune_scheduler(optimizer, args)
        else:
            scheduler = None
        
   
        '''
        lr_list=[]
        print("ABC",optimizer.state_dict()['param_groups'][0]['lr'])
        for epoch in range(100):
            optimizer.step()
            scheduler.step()
            
            # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            lr_list.append(scheduler.get_last_lr()[0])
        print("schedule_loss:",lr_list)
        '''

        #scheduler = None
        args.transfer_loss_weight=0
        test_person(model, finetune_target_test_loader, L_test,cl_weight,args)
        '''
        print("FINE_TUNE BEGIN!")
        for param in model.base_network.parameters():
	        param.requires_grad = False
        train_finetune(finetune_source_loader, finetune_target_train_loader, finetune_target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
        '''
        
        '''
        #train(source_loader, target_train_loader, target_test_loader,L_test2, model, optimizer, scheduler, cl_weight,args)
        print("ARG:",args)
        print("new_code")
        model=input(args)
        scheduler = get_scheduler(optimizer, args)
        args.transfer_loss_weight=0
        
        for param in model.base_network.parameters():
	        param.requires_grad = True
        train(finetune_source_loader, finetune_target_train_loader, finetune_target_test_loader,L_test2, model, optimizer, scheduler, cl_weight,args)
        print("ARG:",args)
        '''
    '''
        ans=[0,0,0,0,0,0]
        for i in range(1,11):
            init()
            train(source_loader, target_train_loader, target_test_loader,L_test, model, optimizer, scheduler, cl_weight,args)
            ans[0]+=best_person_f1
            ans[1]+=best_person_recall
            ans[2]+=best_f1
            ans[3]+=best_recall
            print("PERSON:",ans[0]/i,ans[1]/i,"SAMPLE:",ans[2]/i,ans[3]/i)
        print("ARG:",args)
    '''
    print("CL_weight",cl_weight)

if __name__ == "__main__":
    main()
