'''A data driven approach in solving Bayesian inverse problem'''
# Author: Zhongjian Wang
from run_wrapper import run_setup
import sys
from timeit import default_timer
sys.path.append('./codes/')
from utils import log_writer
########## MAIN ###########
###
schedules = ['cosine1'] # options: exp, cosine1, linear
learning_models = [ 'CEM'] # options: CEM, DDPM
operator_models = ['FNO'] # options: FNO, UNet
datasets = ['identity'] #  options: darcy, identity, identity3d
###
logf=log_writer('./results/R.log',print_on_screen=True)
for schedule in schedules:
	for dataset in datasets:
		for learning_model in learning_models:
			for operator_model in operator_models:
				t1=default_timer()
				config_name = 'R_'+schedule+'_'+dataset+'_'+operator_model+'_'+learning_model
				err_mean,err_std=run_setup(config_name,schedule,dataset,operator_model,learning_model,print_on_screen=True)
				logf.print(config_name,':',err_mean,err_std,'in',default_timer()-t1,'s')