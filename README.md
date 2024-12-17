Conditioned FNO is proposed based on the Fourier neural operator. It accepts both field input (as generally discuss in related paper) and vector type input. 

The operator then models an operator which is continous in the functional space with respect to vector type parameters. The architecture is inspired by the time embedding in UNet.

A featuring application of CFNO is the data driven approach in solving Bayesian inverse problems. The goal of such problems is to generate posterior distribution of a field $u$ given (partial) observation of $\mathcal{G}(u)$, where $\mathcal{G}$ is a nonlinear operator.

Our approach in the example models the posterior distribution by a Diffusion model (more precisely CEM). The backward process starts with white noise and solves a backward SDE where the conditional score functional is approximated by CFNO.

1. Training Data Preparation: There are three 'data_gen' code to generate the corresponding data. The test data are also uploaded to reduce computational time.
2. Run training_model.py to train the model, during training, the loss and relative error of mean and std are output.

Generated mean compared (10000s training, 70s generation for 1000 samples)  with MCMC (Days)
![mean](https://github.com/wang-zhongjian/CFNO/assets/54036885/d6c4bfba-0338-468d-bbf7-950b3f6919c8)
Generated std compared with MCMC
![std](https://github.com/wang-zhongjian/CFNO/assets/54036885/9a1600f8-b377-49b3-b752-cd16636da75d)
Generated realizations
![real1](https://github.com/wang-zhongjian/CFNO/assets/54036885/958a3b6e-eb4e-4d4f-aac3-1c29d70623d3)
![real2](https://github.com/wang-zhongjian/CFNO/assets/54036885/536233b5-1cdd-4f3d-8e7d-cf69ee7d4044)


related preprint: Wasserstein Bounds for generative diffusion models with Gaussian tail
  targets, http://arxiv.org/abs/2412.11251
