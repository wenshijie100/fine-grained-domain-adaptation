import numpy as np
from sklearn import svm
from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np
from collections import Counter
import sys
sys.path.append("..")
import my_person_item
def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    #print("SHAPE:",source_X.shape,target_X.shape)
    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)
    #C_list=[-3,-1,0,1,3]
    half_source, half_target = min(32,int(nb_source/2)), min(32,int(nb_target/2))
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return best_risk

def proxy_conflict(source_label,target_label,source_person,source_item,target_person,target_item,memory_bank):
    
    nb_source = np.shape(source_label)[0]
    nb_target = np.shape(target_label)[0]
    half_source, half_target = min(32,int(nb_source/2)), min(32,int(nb_target/2))
    cnt_t,cnt_s=0,0
    for i in range(half_source,nb_source):
        if memory_bank.get_mode_source(source_person[i])!=source_label[i]:
            cnt_s+=1
        memory_bank.update_source(source_person[i],source_item[i],source_label[i])
    for i in range(half_target,nb_target):
        if memory_bank.get_mode_target(target_person[i])!=target_label[i]:
            cnt_t+=1
        memory_bank.update_target(target_person[i],target_item[i],target_label[i])
    number_down=max(1,(nb_source-half_source+1)+(nb_target-half_target+1))
    number_up=cnt_s+cnt_t 
    #return 0
    #print("conflict",number_up,number_down)
    return float( number_up/number_down/4)
def estimate_mu(_X1, _Y1, _X2, _Y2,source_person,source_item,target_person,target_item,memory_bank,epoch):
    """
    Estimate value of mu using conditional and marginal A-distance.
    """
   # print("SSSSSSSSSSS",target_item)
   # print("SSSSSSSSSSS",source_item)
    adist_m = proxy_a_distance(_X1, _X2)
    Cs, Ct = np.unique(_Y1), np.unique(_Y2)
    #print("CS",Cs)
    #print("CT",Ct)
    
    C = np.intersect1d(Cs, Ct)
    epsilon = 1e-3
    list_adist_c = []
    tc = len(C)
    #print("C:",C,tc)
    for i in C:
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        #print("CCCCCCCC",ind_i)
        #print("DDDDDDDD",ind_j)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        if len(Xsi) <= 1 or len(Xtj) <= 1:
            tc -= 1
            continue
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    if tc < 1:
        return 0
    adist_c = sum(list_adist_c) / tc
    adist_p = proxy_conflict(_Y1,_Y2,source_person,source_item,target_person,target_item,memory_bank)
    #adist_p = 0
  
    if(adist_c+adist_m+adist_p<1e-7):
        return 0,1.0,0
    mu = adist_c / (adist_c + adist_m)
    '''
    if mu > 1:
        mu = 1
        print("MU ERROR!")
    if mu < epsilon:
        mu = 0
    '''
    #adist_p=0 
    adjust_sum= adist_c+adist_m+adist_p
    
    adist_c,adist_m,adist_p = adist_c/adjust_sum,adist_m/adjust_sum,adist_p/adjust_sum
    adist_c,adist_m,adist_p= memory_bank.update_para(adist_c,adist_m,adist_p)
    #return  mu,1-mu,0
    #if epoch<50:
        #return mu,1-mu,0
    if adist_p>0.005*epoch:
        lf=1-0.005*epoch
        return mu*lf,(1-mu)*lf,0.005*epoch
    return  adist_c,adist_m,adist_p
    #a,b,c=adist_c/adjust_sum,adist_m/adjust_sum,adist_p/adjust_sum
    #return a/(a+b)*(1+c),b/(a+b)*(1+c),-c
class Memory_Bank:
    def __init__(self,source_person_number,source_item_number,target_person_number,target_item_number):
        self.source_bank=[[-1 for i in range(source_item_number+1)] for j in range(source_person_number)]
        self.target_bank=[[-1 for i in range(target_item_number+1)] for j in range(target_person_number)]
        self.adjust_c=1.0
        self.adjust_m=0
        self.adjust_p=0
    
    def update_source(self,source_person,source_item,label):
        #print("SSSSSSS",source_person,source_item,len(self.source_bank),len(self.source_bank[0]))
        self.source_bank[source_person][source_item]=label
    def update_target(self,target_person,target_item,label):
        self.target_bank[target_person][target_item]=label
    def get_mode_source(self,source_person):
        #print("SSSSSSS",source_person,len(self.source_bank),len(self.source_bank[0]))
        t=self.source_bank[source_person]
        cnt=Counter(t)
        arr=np.array([cnt[0],cnt[1],cnt[2],cnt[3]])
        #print("src:",arr)
        return np.argmax(arr)

    def get_mode_target(self,target_person):
        t=self.target_bank[target_person]
        cnt=Counter(t)
        arr=np.array([cnt[0],cnt[1],cnt[2],cnt[3]])
        #print("tgt:",arr)
        return np.argmax(arr)
    def update_para(self,c,m,p):
        #self.adjust_c,self.adjust_m,self.adjust_p=0.99*self.adjust_c+0.01*c,0.99*self.adjust_m+0.01*m,0.99*self.adjust_p+0.01*p
        self.adjust_c,self.adjust_m,self.adjust_p=c,m,p
        
        return self.adjust_c,self.adjust_m,self.adjust_p

class THREEMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, 
                    gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(THREEMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class
        #print("KKKK",kwargs)
        myPI=kwargs['my_person_item']
        self.memory_bank =Memory_Bank(myPI.source_person_number,myPI.source_item_number,myPI.target_person_number,myPI.target_item_number)
    def forward(self, source, target, source_label,source_person,source_item,source_weight,target_weight,target_person,target_item, target_logits,mu_list,epoch):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
      
        else :
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            #print("weight_ss",weight_ss)
            weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]
            #print("SHAPE:",weight_ss.shape,weight_tt.shape)
            '''
            print("SW:",source_weight*source_weight)
            print("TW:",target_weight*target_weight)
            print("ST:",source_weight*target_weight)
            print("weight_ss:",weight_ss)
            print("weight_tt:",weight_tt)
            '''
            '''
            loss += 0.2*torch.sum( (weight_ss)*(source_weight*source_weight)* SS +\
            (weight_tt)* (target_weight*target_weight) * TT -\
             2 * (weight_st)*(source_weight*target_weight) * ST )
            '''
            
            #print("LOSS0:",loss)
            #adist_c,adist_m,adist_p =0.6,0.3,0.1

            adist_c,adist_m,adist_p=estimate_mu(source.cpu().detach().numpy(), source_label.cpu().detach().numpy(), \
            target.cpu().detach().numpy() ,target_logits.cpu().data.max(1)[1].numpy(), \
            source_person.cpu().detach().numpy(),source_item.cpu().detach().numpy(), \
            target_person.cpu().detach().numpy(),target_item.cpu().detach().numpy(),self.memory_bank,epoch)
            #print("MU",mu)
            #adist_p,adist_c,adist_m=0.3,0.4,0.3
            #adist_c,adist_m,adist_p=1.0,0.0,0.0
            #adist_c,adist_m,adist_p=0.45,0.45,0.1
            #print("loss rate:",adist_c , adist_m,adist_p)
            loss1 = adist_c* torch.sum( (weight_ss) * SS +\
            (weight_tt) * TT -\
             2 * (weight_st) * ST )
             
            #print("LOSS1:",loss)
            loss2= adist_p *1/batch_size/batch_size* torch.sum( ((source_weight*source_weight) * SS +\
             (target_weight*target_weight)*TT -\
             2 * (source_weight*target_weight) * ST ))
            '''
            loss2=(adist_p)*torch.sum( (weight_ss)*(source_weight*source_weight)* SS +\
            (weight_tt)* (target_weight*target_weight) * TT -\
             2 * (weight_st)*(source_weight*target_weight) * ST )
            '''
            #print("LOSS2:",loss)
            loss3=adist_m*1/batch_size/batch_size* torch.sum((SS+TT-2*ST))
            #print("LOSS3:",loss)
            
            #print(loss2)
            loss= loss1+loss2+loss3
            #loss= loss1+loss3
            #print(loss1,loss2,loss3,loss)
            #print("LOSS three:",loss1.float(),loss2.float(),loss3.float(),loss.float())
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            return loss
    
    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label] # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100 #can not divide zero
        '''
        print("ABC")
        print(source_label_onehot)
        print(source_label_sum )
        '''
        source_label_onehot = source_label_onehot / source_label_sum # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class): # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
                  
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st     
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


