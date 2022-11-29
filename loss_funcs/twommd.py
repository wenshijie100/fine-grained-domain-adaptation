import numpy as np
from sklearn import svm
from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np

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

    half_source, half_target = min(16,int(nb_source/2)), min(16,int(nb_target/2))
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
        #print("C:",C,test_risk)
        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def estimate_mu(_X1, _Y1, _X2, _Y2):
    """
    Estimate value of mu using conditional and marginal A-distance.
    """
    adist_m = proxy_a_distance(_X1, _X2)
    Cs, Ct = np.unique(_Y1), np.unique(_Y2)
    #print("CS",Cs)
    #print("CT",Ct)
    
    C = np.intersect1d(Cs, Ct)
    
    list_adist_c = []
    tc = len(C)
    if tc<1:
        return 0
    for i in C:
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        if len(Xsi) <= 1 or len(Xtj) <= 1:
            tc -= 1
            continue
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    if tc<1:
        return 0
    adist_c = sum(list_adist_c) / tc
    if(adist_c+adist_m<1e-7):
        return 0.5
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < 1e-7:
        mu = 0
    return mu




class TWOMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, 
                    gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(TWOMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class
    def forward(self, source, target, source_label, target_logits,source_weight,target_weight,mu_list):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
        elif self.kernel_type == 'rbf':
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

            #mu=0.5
            mu=estimate_mu(source.cpu().detach().numpy(), source_label.cpu().detach().numpy(), target.cpu().detach().numpy() ,target_logits.cpu().data.max(1)[1].numpy())
            #print("MU",mu)
            loss += (mu)*torch.sum( (weight_ss) * SS +\
            (weight_tt) * TT -\
             2 * (weight_st) * ST )
            
            #print("LOSS1:",loss)
            #print("MU:",mu,1-mu)
            '''
            loss += (1-mu)*torch.sum( 1/batch_size/batch_size*((source_weight*source_weight) * SS +\
             (target_weight*target_weight)*TT -\
             2 * (source_weight*target_weight) * ST ))
            '''
             
            #print("LOSS2:",loss)
            loss +=(1-mu)*torch.sum(1/batch_size/batch_size*(SS+TT-2*ST))
            #print("LOSS3:",loss)
            
            #print(mu,loss)
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


