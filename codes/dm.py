#!/usr/bin/python
# -*- coding: utf-8 -*-
# Provide Training and Evaluation functions

import numpy as np
import torch
from my_fno_package import UnitGaussianNormalizer


## Sampling

def sample_z_from_model(
    model,
    theta,
    s,
    t_list_sample,
    device,
    temb_model,
    learning_model,
    grf=None,
    d=2
    ):
    batch_size = theta.shape[0]
    Nt = t_list_sample.shape[0] - 1
    if grf is not None:
        assert d == 2 , "colored noise is only developed for 2d"
        model_noise = \
            grf.spectral2grid_vectorized(np.random.standard_normal(batch_size,
                grf.N_KL))
        z = np.array(model_noise).reshape(batch_size, 1, s,
                s).astype(np.float32)
    else:
        z = np.random.standard_normal((batch_size, 1,)+(s,)*d).astype(np.float32)
    for it in range(Nt):
        dt = t_list_sample[-it - 1] - t_list_sample[-it - 2]
        alpha = np.exp(-dt)
        alpha_bar = np.exp(-t_list_sample[-it - 1])
        t_idx = torch.Tensor([Nt - it
                             + 0.0]).repeat(batch_size).to(device)
        t = torch.Tensor([t_list_sample[-it
                         - 1]]).repeat(batch_size).to(device)
        input_t = get_input_t(t[:, None], t_idx, temb_model, device)
        out = model(torch.from_numpy(z).to(device), theta.to(device),
                    input_t).cpu().detach().numpy()

        if grf is not None:
            model_noise = \
                grf.spectral2grid_vectorized(np.random.randn(batch_size,
                    grf.N_KL))
            model_noise = np.array(model_noise)[:, None, :, :
                    ].astype(np.float32)
        else:
            model_noise = np.random.standard_normal((batch_size, 1,)+(s,)*d).astype(np.float32)

        if learning_model == 'DDPM':
            z = z - (1 - alpha) * out / np.sqrt(1 - alpha_bar)
            z = z / np.sqrt(alpha) + np.sqrt(1 - alpha) * model_noise
        else:

            # S = (z- np.sqrt(alpha_bar)*out)/(1-alpha_bar)

            z = z - (1 - alpha) * (z - np.sqrt(alpha_bar) * out) / (1
                    - alpha_bar)
            z = z / np.sqrt(alpha) + np.sqrt(1 - alpha) * model_noise

    return torch.Tensor(z).to(device)



def set_schedule(schedule, T=10, Nt=100):
    if schedule == 'linear':
        t1 = (T + 0.0) / (Nt + 0.0)
        t_list = t1 * np.linspace(1, Nt, Nt)
        lambda_list = torch.Tensor([1.0]).repeat((Nt, ))
    elif schedule == 'alpha': # Schedule introduced in DDPM

        beta_k = np.linspace(1e-4, 2e-2, Nt)
        alpha_k = 1.0 - beta_k
        dt_k = -np.log(alpha_k)
        t_list = np.cumsum(dt_k)
        lambda_list = torch.Tensor([1.0]).repeat((Nt, ))
    elif schedule == 'cosine':

        s_ = 0.008
        t_idx = np.linspace(0, Nt, Nt + 1)
        ft = np.cos(np.pi / 2.0 * (t_idx / Nt + s_) / (1 + s_)) ** 2
        dt_k = -np.log(ft[1:] / ft[:-1])
        dt_k = np.minimum(dt_k, -np.log(1e-3))
        t_list = np.cumsum(dt_k)
        lambda_list = torch.Tensor([1.0]).repeat((Nt, ))
    elif schedule == 'cosine1':

        s_ = 0.008
        t_idx = np.linspace(0, Nt, Nt + 1)
        ft = np.cos(np.pi / 2.0 * (t_idx / Nt + s_) / (1 + s_)) ** 2
        dt_k = -np.log(ft[1:] / ft[:-1])
        dt_k = np.minimum(dt_k, -np.log(1e-3))
        t_list = np.cumsum(dt_k)
        lambda_list = torch.Tensor(1.0 / (np.exp(t_list) - 1))
    elif schedule == 'cosine2':

        s_ = 0.008
        t_idx = np.linspace(0, Nt, Nt + 1)
        ft = np.cos(np.pi / 2.0 * (t_idx / Nt + s_) / (1 + s_)) ** 2
        dt_k = -np.log(ft[1:] / ft[:-1])
        dt_k = np.minimum(dt_k, -np.log(1e-3))
        t_list = np.cumsum(dt_k)
        lambda_list = torch.Tensor(dt_k / (np.exp(t_list) - 1))
    elif schedule == 'exp':

        t1 = 1e-4
        alpha_t = np.log(T / t1) * (1 / (Nt - 1))
        t_list = t1 * np.exp(alpha_t * np.arange(0, Nt, 1))
        lambda_list = torch.Tensor(t_list / (np.exp(t_list) - 1))  # sampling itself add another weight
    else:

        raise Exception('Schedule Name Unknown')

    return (t_list, lambda_list)


def get_input_t(
    t,
    t_idx,
    temb_model,
    device,
    ):
    if temb_model == 'transformer':
        input_t = t_idx.to(device)
    elif temb_model == 'MLP':
        input_t = torch.log(t.to(device))
    else:
        raise Exception('Time Embedding Name Unknown')

    return input_t



## Universal dataloader

class dataloader:
    def __init__(self,dataset,ntrain,ntest,batch_size,snapshot = 41):
        method = 'sperate' # default method

        # Data sets generated

        if dataset == 'darcy':
            PATH = './data/darcy2d'
            TRAIN_PATH = PATH +  '/data_set.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma = 1.0
            r = 1 # skip when reading data
            s = 80 # original size
            s1 = 49 # observation
            dimension = 2
            self.reference_exist = True


        elif dataset =='identity':
            PATH = './data/identity/'
            TRAIN_PATH = PATH +  '/data_set.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma = 1e-1
            r = 1 # skip when reading data
            s = 80 # original size
            s1 = 49 # observation
            dimension = 2
            self.reference_exist = True

        elif dataset == 'identity3d':
            PATH = './data/identity3d/'
            TRAIN_PATH = PATH +  '/data_set.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma =2e-1
            r = 1
            s = 21
            s1 = 125
            dimension = 3
            self.reference_exist = True

        # Dataset Transferred to nscc
            
        elif dataset == 'darcy-nscc':
            PATH = '/scratch/users/ntu/zhongjia/darcy2d'
            TRAIN_PATH = PATH +  '/data_set_100000.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma = 1.0
            r = 1 # skip when reading data
            s = 80 # original size
            s1 = 49 # observation
            dimension = 2
            self.reference_exist = True
        
        elif dataset =='identity-nscc':
            PATH = '/scratch/users/ntu/zhongjia/identity2d'
            TRAIN_PATH = PATH +  '/data_set.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma = 1e-1
            r = 1 # skip when reading data
            s = 80 # original size
            s1 = 49 # observation
            dimension = 2
            self.reference_exist = True

        elif dataset == 'identity3d-nscc':
            PATH = '/scratch/users/ntu/zhongjia/identity3d'
            TRAIN_PATH = PATH +  '/data_set.npy'
            TEST_PATH = PATH + '/data_set_100.npy'
            TEST1_PATH = PATH + '/test.npy'
            noise_sigma =2e-1
            r = 1
            s = 21
            s1 = 125
            dimension = 3
            self.reference_exist = True
        

        elif dataset == 'cfd3d-nscc':
            PATH = '/scratch/users/ntu/zhongjia/cfd3d/'
            noise_sigma = 1e-2
            r = 2
            s = 64 # original size
            s1 = 64 * 5
            dimension = 3
            self.reference_exist = False
            method = 'dir'

        else:
            raise Exception('unknow type of dataset')
        
        
        ##
        
        s = int(((s - 1)/r) + 1)
        hx = 1.0/(s-1.0) # Assume Uniform Mesh

        # 
        self.dimension = dimension
        self.noise_sigma = noise_sigma
        self.s = s
        self.s1 = s1
        self.r = r
        #

        if method == 'sperate'  :
            with open(TRAIN_PATH, 'rb') as f:
                yshape = (ntrain,1)+(s,)*dimension
                y_train = np.load(f)[:ntrain]
                y_train=torch.from_numpy(y_train.astype(np.float32)).reshape(yshape)
                theta_train = np.load(f)[:ntrain]
                theta_train=torch.from_numpy(theta_train.astype(np.float32)).reshape(ntrain,s1)
            with open(TEST_PATH, 'rb') as f:
                yshape = (ntest,1)+(s,)*dimension
                y_test = np.load(f)[:ntest]
                y_test=torch.from_numpy(y_test.astype(np.float32)).reshape(yshape)
                theta_test = np.load(f)[:ntest,:]
                theta_test=torch.from_numpy(theta_test.astype(np.float32)).reshape(ntest,s1)
            with open(TEST1_PATH, 'rb') as f:
                yshape = (1,1)+(s,)*dimension
                theta_test1=np.load(f)
                y_test1=np.load(f)
                mean2d_test1=np.load(f)
                std2d_test1=np.sqrt(np.load(f))

                theta_test1 = theta_test1.reshape(1,s1)
                theta_test1 = theta_test1.repeat(batch_size,axis=0)
                theta_test1 = torch.from_numpy(theta_test1.astype(np.float32))
                y_test1 = y_test1.reshape(yshape)
                y_test1 = y_test1.repeat(batch_size,axis=0)
                y_test1 = torch.from_numpy(y_test1.astype(np.float32))
            

        elif method =='dir':
            y_train,theta_train,y_test,theta_test,y_test1,theta_test1 = self.read_dir_3d(PATH,ntrain,ntest,snapshot=snapshot)
            y_test1=y_test1.repeat(batch_size,1,1,1,1)
            theta_test1=theta_test1.repeat(batch_size,1)
            
      

        else:
            raise Exception('reading method unknown')

        assert (torch.max(y_test1-y_train[:batch_size]))>1e-5, 'meaningless inverse problem'
        # Encoding
        theta_normalizer = UnitGaussianNormalizer(theta_train)
        theta_train = theta_normalizer.encode(theta_train)
        theta_test = theta_normalizer.encode(theta_test)
        theta_test1 = theta_normalizer.encode(theta_test1)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        # Data Loader
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(theta_train, y_train), \
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(theta_test, y_test), batch_size=batch_size, shuffle=False)

        # Save in class for later use
        self.theta_normalizer = theta_normalizer
        self.y_normalizer = y_normalizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.y_test1 = y_test1
        self.theta_test1 = theta_test1
        
        if self.reference_exist:
            
            self.mean2d_test1 = mean2d_test1
            self.std2d_test1 = std2d_test1
        

    def read_dir_3d(self,PATH,ntrain,ntest,snapshot = 41,start_key = 2001,safemode = False):
        key = start_key
        y = []
        theta = []
        r = self.r
        for i in range(ntrain+ntest+1):
            if safemode:
                accepted = False
                pastkeys = 0
                while ~accepted and pastkeys<50:
                    try:
                        nowpath = PATH + str(key) +'/'
                        y.append(np.load(nowpath+'Data_0000.npy')[1:2,::2,::2,::2].reshape(1,self.s,self.s,self.s)) # 3dcfd data, (rho,vx,vy,vz,p), rho/p constant initial
                        theta.append(np.load(nowpath+'Data_'+str(snapshot).zfill(4)+'.npy').reshape(self.s1,)) #
                        accepted = True
                    finally:
                        key +=1
                        pastkeys +=1
            else:
                nowpath = PATH + str(key) +'/'
                y.append(np.load(nowpath+'Data_0000.npy')[1:2,::r,::r,::r].reshape(1,self.s,self.s,self.s)) # 3dcfd data, (rho,vx,vy,vz,p), rho/p constant initial
                theta.append(np.load(nowpath+'Data_'+str(snapshot).zfill(4)+'.npy').reshape(self.s1,)) #
                key += 1


        y = torch.from_numpy(np.array(y).astype(np.float32))
        print(ntrain+ntest+1)
        theta = torch.from_numpy(np.array(theta).astype(np.float32))
        print(torch.sqrt(torch.mean(theta**2)))
        return y[:ntrain],theta[:ntrain],y[ntrain:ntrain+ntest],theta[ntrain:ntrain+ntest],y[ntrain+ntest:ntrain+ntest+1],theta[ntrain+ntest:ntrain+ntest+1]
        
