from timeit import default_timer
from datetime import datetime
import numpy as np
import torch
import sys
sys.path.append('./codes/')

from my_fno_package import *
from my_fno_package_3d import *
from unet import UNet
import dm
from utils import Gaussian_random_field_3d,log_writer

###

def run_setup(config_name,schedule,dataset,operator_model,learning_model, \
              ntrain = 10000,temb_continuous = False,colored_prior=False, \
                print_on_screen=False,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''A wrapper to run any data set with configs in one line'''
    torch.manual_seed(0)
    np.random.seed(0)
    
    epochs = 500
    ntest = 100 # Test set used to compute l2 err
    test_size = 1000 # samples used to compute mean/std to compare with mcmc result
    batch_size = 20
    learning_rate = 1e-4 # 1e-3 is dangerous
    epsteps = epochs//10 # Lines of  output
    
    # # safe test
    # epochs =10
    # ntrain =100
    # ntest=20
    # test_size = 20
    # epsteps = epochs

    # Common Configs
    temb_model = 'MLP'
    if ntrain != 1e4:
        config_name = config_name +'_'+str(ntrain)+'data'
    if learning_rate != 1e-4:
        config_name = config_name +'_lr-'+str(learning_rate)
    if temb_continuous:
        config_name = config_name +'_CTEmb'
        assert temb_model != 'transformer' # transformer only used idx as input
    if colored_prior:
        N_KL=128
        grf = Gaussian_random_field(np.linspace(0,1,80),N_KL)
        config_name = config_name + '_colored128'
    else:
        grf = None

    # Saving and output
    state_savepath='./results/'+config_name+'_params_'+str(epochs)+'epoch.npy'
    log_savepath = './results/'+config_name+'_log_'+str(epochs)+'epoch.log'
    logf=log_writer(log_savepath,print_on_screen=print_on_screen)
    logf.print('------Model:',config_name,'------')
    logf.print('Started at ', datetime.now().strftime("%D,%H:%M:%S"))
    t1 = default_timer()

    # Schedule in forward process
    t_list,lambda_list = dm.set_schedule(schedule)
    Nt = len(t_list)
    t_list_tensor = torch.Tensor(t_list)

    # Load Data
    dset = dm.dataloader(dataset,ntrain,ntest,batch_size)
    dset.y_normalizer.to(device)
    dset.theta_normalizer.to(device)
    s = dset.s
    s1 = dset.s1
    theta_test1,y_test1 =dset.theta_test1.to(device),dset.y_test1.to(device)

    # Calculation of field shape
    vec_shape=(batch_size,1,)+(1,)*dset.dimension
    field_shape = (batch_size,1,)+ (s,)*dset.dimension
    field_pos = (1,)
    pos_d = 2
    for _ in range(dset.dimension):
        field_pos = field_pos +(pos_d,)
        pos_d = pos_d +1 

    # Network Initialization
    if operator_model == 'Unet':
        model = UNet(image_channels=1,is_attn=(False, False, False, False),temb_model=temb_model,theta_channels=s1).to(device)
        assert dset.dimension == 2, "3d Unet not configured"
    else:
        if dset.dimension == 2:
            modes = 12
            width = 96 
            model = FNO2d_emdT(modes, modes, width,temb_model=temb_model,theta_channels=s1).to(device)
        else:
            modes = 8
            width = 96 
            model = FNO3d_emdT(modes, modes, modes, width,temb_model=temb_model,theta_channels=s1).to(device)

    # Output
    logf.print('Model Initiated with ',str(default_timer()-t1),'s', \
               'No. parameters: ', \
               str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    ################################################################
    # Training Starts
    ################################################################
    logf.print('Epochs','left time', 'train_loss' ,'err_mean' , 'err_std')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.5)
    # Training starts
    t1 = default_timer()
    for ep in range(epochs):
        model.train() # turning to train mode
        train_l2 = 0
        for theta, y in dset.train_loader:
            theta, y = theta.to(device), y.to(device)
            # Add noise to theta, artifical observation error
            theta = dset.theta_normalizer.decode(theta)
            theta = theta + (torch.randn(batch_size,s1) * dset.noise_sigma).to(device)
            theta = dset.theta_normalizer.encode(theta)
            # Time embedding
            t_idx = torch.randint(0, Nt, (batch_size,))
            if not temb_continuous :
                lambda_t = lambda_list[t_idx].to(device).reshape(vec_shape)
                t = t_list_tensor[t_idx].to(device)
            else:
                logt = torch.rand((batch_size,))*(np.log(t_list[-1])-np.log(t_list[0]))+np.log(t_list[0])
                t = torch.exp(logt).to(device)
                lambda_t = t/(torch.exp(t)-1)
            sqrt_alphabar=torch.exp(-t/2).reshape(vec_shape)
            alpha1=torch.sqrt(1-sqrt_alphabar**2)
            input_t = dm.get_input_t(t,t_idx,temb_model,device).reshape(batch_size,1)
            # Forward process
            optimizer.zero_grad()
            standard_normal = torch.randn(field_shape).to(device)
            z = sqrt_alphabar*y + alpha1 * standard_normal
            # Penalties
            out = model(z,theta,input_t)
            out=out.reshape(field_shape)
            if learning_model == 'DDPM':
                loss = torch.mean((standard_normal-out )**2) # L_simple
            else: # default: CEM Model
                loss = torch.mean(lambda_t*torch.mean((y-out )**2,axis=field_pos))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        scheduler.step()
        train_l2/= ntrain

        # Evaluation
        if (ep+1) % int(epochs/epsteps) == 0:
            model.eval() 
            t_list_sample=np.insert(t_list,0,0)
            # compare with MCMC/reference solution
            solutions1 = []
            test_l21=0
            for _ in range(test_size//batch_size):
                z = dm.sample_z_from_model(model,theta_test1,s,t_list_sample,device,temb_model,learning_model,d = dset.dimension)
                z=dset.y_normalizer.decode(z)
                solutions1.append(z)
                loss = torch.sum(torch.norm(z.view(batch_size,-1)- y_test1.view(batch_size,-1),dim=1)/torch.norm(y_test1.view(batch_size,-1),dim=1))
                test_l21 +=loss.item()
            test_l21 /= test_size
            sols1=torch.cat(solutions1).squeeze().cpu().detach().numpy()
            sols1_mean = np.mean(sols1,0)
            sols1_std = np.std(sols1,0)
            # default step with calculating mean and variance
            if dset.reference_exist:
                err_mean = np.sqrt(np.mean((sols1_mean-dset.mean2d_test1)**2)/np.mean((dset.mean2d_test1)**2))
                err_std = np.sqrt(np.mean((sols1_std-dset.std2d_test1)**2)/np.mean((dset.std2d_test1)**2))
            else:
                mean_ref = y_test1[0].cpu().detach().numpy()
                err_mean = np.sqrt(np.mean((sols1_mean-mean_ref)**2)/np.mean((mean_ref)**2))
                err_std = np.max(sols1_std)
            # Output of evaluations
            t2 = default_timer()
            logf.print(ep+1,(t2-t1)/(ep+1)*(epochs-ep-1),train_l2,err_mean,err_std)
    # Save state
    torch.save(model.state_dict(),state_savepath)
    logf.print('Ended at ', datetime.now().strftime("%D,%H:%M:%S"))