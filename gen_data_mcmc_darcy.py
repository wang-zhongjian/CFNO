import numpy as np
import scipy.sparse as sp
import emcee
from tqdm import tqdm
import multiprocessing as mp


np.random.seed(88)

"""
Forward solver
"""


"""
12 13 14 15
8  9  10 11
4  5  6  7    
0  1  2  3
=> 
X  X  X  X
X  2  3  X
X  0  1  X     
X  X  X  X
"""


def ind(N, ix, iy):
    return ix-1 + (iy-1)*(N - 2)

"""
    solve Darcy equation with finite difference method:
    -∇(k∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
    k_2d[i, j] = k(xi, yj)
"""


def solve_Darcy_2D(k_2d, f_2d, L, N): 

    dx = L/(N-1)
    indx, indy, vals = [], [], []

    c = dx**2
    for iy in range(1,N-1):
        for ix in range(1,N-1):
            ixy = ind(N, ix, iy) 
            #top
            if iy == N-2:
                #ft = -(k_2d[ix, iy] + k_2d[ix, iy+1])/2.0 * (0 - h_2d[ix,iy])
                indx.append(ixy)
                indy.append(ixy)
                vals.append((k_2d[ix, iy] + k_2d[ix, iy+1])/2.0/c)
            else:
                #ft = -(k_2d[ix, iy] + k_2d[ix, iy+1])/2.0 * (h_2d[ix,iy+1] - h_2d[ix,iy])
                indx.extend([ixy, ixy])
                indy.extend([ixy, ind(N, ix, iy+1)])
                vals.extend([(k_2d[ix, iy] + k_2d[ix, iy+1])/2.0/c, -(k_2d[ix, iy] + k_2d[ix, iy+1])/2.0/c])
     
            
            #bottom
            if iy == 1:
                #fb = (k_2d[ix, iy] + k_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - 0)
                indx.append(ixy)
                indy.append(ixy)
                vals.append((k_2d[ix, iy] + k_2d[ix, iy-1])/2.0/c)
            else:
                #fb = (k_2d[ix, iy] + k_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - h_2d[ix,iy-1])
                indx.extend([ixy, ixy])
                indy.extend([ixy, ind(N, ix, iy-1)])
                vals.extend([(k_2d[ix, iy] + k_2d[ix, iy-1])/2.0/c, -(k_2d[ix, iy] + k_2d[ix, iy-1])/2.0/c])
  
            
            #right
            if ix == N-2:
                #fr = -(k_2d[ix, iy] + k_2d[ix+1, iy])/2.0 * (0 - h_2d[ix,iy])
                indx.append(ixy)
                indy.append(ixy)
                vals.append((k_2d[ix, iy] + k_2d[ix+1, iy])/2.0/c)
            else:
                #fr = -(k_2d[ix, iy] + k_2d[ix+1, iy])/2.0 * (h_2d[ix+1,iy] - h_2d[ix,iy])
                indx.extend([ixy, ixy])
                indy.extend([ixy, ind(N, ix+1, iy)])
                vals.extend([(k_2d[ix, iy] + k_2d[ix+1, iy])/2.0/c, -(k_2d[ix, iy] + k_2d[ix+1, iy])/2.0/c])
       
            
            #left
            if ix == 1:
                #fl = (k_2d[ix, iy] + k_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - 0)
                indx.append(ixy)
                indy.append(ixy)
                vals.append((k_2d[ix, iy] + k_2d[ix-1, iy])/2.0/c)
            else:
                #fl = (k_2d[ix, iy] + k_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - h_2d[ix-1,iy])
                indx.extend([ixy, ixy])
                indy.extend([ixy, ind(N, ix-1, iy)])
                vals.extend([(k_2d[ix, iy] + k_2d[ix-1, iy])/2.0/c, -(k_2d[ix, iy] + k_2d[ix-1, iy])/2.0/c])
            

    df = sp.csc_matrix((vals, (indx, indy)), shape=((N-2)**2, (N-2)**2))
    
    # Multithread does not support sparse matrix solver
    h = sp.linalg.spsolve(df, f_2d[1:N-1,1:N-1].flatten("F"))
    
    h_2d = np.zeros((N, N))
    h_2d[1:N-1,1:N-1] = np.reshape(h, (N-2, N-2), order="F") 
    
    return h_2d



"""
Compute sorted pair (i, j), sorted by i**2 + j**2
with i>=0 and j>=0 and i+j>0
These pairs are used for Karhunen–Loève expansion
"""
def compute_seq_pairs(N_KL):
    seq_pairs = np.zeros((N_KL, 2), dtype=np.int64)
    trunc_Nx = np.int64(np.sqrt(2*N_KL)) + 1
    
    seq_pairs = np.zeros(((trunc_Nx+1)**2 - 1, 2), dtype=np.int64)
    seq_pairs_mag = np.zeros((trunc_Nx+1)**2 - 1, dtype=np.int64)
    
    seq_pairs_i = 0
    for i in range(trunc_Nx+1):
        for j in range(trunc_Nx+1):
            if (i == 0 and j ==0):
                continue
            seq_pairs[seq_pairs_i, :] = i, j
            seq_pairs_mag[seq_pairs_i] = i**2 + j**2
            seq_pairs_i += 1

    seq_pairs = seq_pairs[np.argsort(seq_pairs_mag)]
    return seq_pairs[0:N_KL, :]


class Gaussian_random_field:
    def __init__(self, xx, N_KL, d=2.0, tau=3.0):
        N = len(xx)
        Y, X = np.meshgrid(xx, xx)
        
        seq_pairs = compute_seq_pairs(N_KL)
        
        phi = np.zeros((N_KL, N, N))
        lam = np.zeros(N_KL)
        
        for i in range(N_KL):
            if (seq_pairs[i, 0] == 0 and seq_pairs[i, 1] == 0):
                phi[i, :, :] = 1.0
            elif (seq_pairs[i, 0] == 0):
                phi[i, :, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*Y))
            elif (seq_pairs[i, 1] == 0):
                phi[i, :, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*X))
            else:
                phi[i, :, :] = 2*np.cos(np.pi * (seq_pairs[i, 0]*X)) *  np.cos(np.pi * (seq_pairs[i, 1]*Y))
            lam[i] = (np.pi**2*(seq_pairs[i, 0]**2 + seq_pairs[i, 1]**2) + tau**2)**(-d)

        self.N = N
        self.X = X
        self.Y = Y
        self.N_KL = N_KL
        self.seq_pairs = seq_pairs
        self.lam = lam
        self.phi = phi 
        
    def spectral2grid(self, theta):
        N, N_KL = self.N, self.N_KL
        lam, phi = self.lam, self.phi
        N_theta = len(theta)
        
        assert(N_theta <= N_KL) 
        logk_2d = np.zeros((N, N))
        for i in range(N_theta):
            logk_2d += theta[i] * np.sqrt(lam[i]) * phi[i, :, :]
        return logk_2d

    def spectral2grid_vectorized(self, theta):
        N, N_KL = self.N, self.N_KL
        lam, phi = self.lam, self.phi
        assert(len(theta.shape)==2)
        N_theta = theta.shape[1]
        assert(N_theta <= N_KL) 
        logk_2d = np.zeros((theta.shape[0],N, N))
        for i in tqdm(range(N_theta)):
            logk_2d += theta[:,i:i+1,None] * np.sqrt(lam[i]) * phi[i:i+1, :, :]
        return logk_2d

def compute_obs(h_2d):
    obs_ΔN = 10
    return h_2d[obs_ΔN-1:-1:obs_ΔN, obs_ΔN-1:-1:obs_ΔN].flatten(order='F')

def compute_f_2d(yy):
    N = len(yy)
    f_2d = np.zeros((N, N))
    for i in range(N):
        if (yy[i] <= 4/6):
            f_2d[:,i] = 1000.0
        elif (yy[i] >= 4/6 and yy[i] <= 5/6):
            f_2d[:,i] = 2000.0
        elif (yy[i] >= 5/6):
            f_2d[:,i] = 3000.0
    return f_2d

def log_prob(theta, f_2d, L, N, y, Sigma_eta, mean_prior, Sigma_prior, grf):
    logk_2d = grf.spectral2grid(theta)
    k_2d = np.exp(logk_2d)
    h_2d = solve_Darcy_2D(k_2d, f_2d, L, N)
    y_pred = compute_obs(h_2d)
    
    log_prob = -0.5*np.dot(y - y_pred, np.linalg.solve(Sigma_eta, y - y_pred))
    log_prob -= 0.5*np.dot(theta - mean_prior, np.linalg.solve(Sigma_prior, theta - mean_prior))
    return log_prob
  
def generate_single(theta,L,N,N_KL,grf):

    xx, yy = np.linspace(0, L, N), np.linspace(0, L, N)
    Y, X = np.meshgrid(xx, yy)
    f_2d = compute_f_2d(yy)

    logk_2d = grf.spectral2grid(theta)

    k_2d = np.exp(logk_2d)
    h_2d = solve_Darcy_2D(k_2d, f_2d, L, N)
    obs_noiseless = compute_obs(h_2d)
    return obs_noiseless,logk_2d


##################
# MAIN PROGRAMME #
##################

L = 1.0 
N = 80
N_KL = 64
sigma_eta = 1.0
xx, yy = np.linspace(0, L, N), np.linspace(0, L, N)
Y, X = np.meshgrid(xx, yy)
f_2d = compute_f_2d(yy)
grf = Gaussian_random_field(xx, N_KL, d=2.0, tau=3.0)

theta = np.random.normal(0, 1, N_KL) 
obs_noiseless,logk_2d = generate_single(theta,L,N,N_KL,grf)
obs = obs_noiseless + sigma_eta*np.random.normal(0,1,len(obs_noiseless))
ref = logk_2d

#### MCMC
ndim, nwalkers = N_KL, 200
p0 = np.random.randn(nwalkers, ndim)
# Obs noise matrix
Sigma_eta = np.diag(sigma_eta**2*np.ones(len(obs_noiseless))) 
# Prior measure, mean and variance
mean_prior, Sigma_prior = np.zeros(N_KL), np.diag(np.ones(N_KL))
# log_prob: fucntion handle to compute mcmc
num_workers = mp.cpu_count()  
print('CPU N.:', num_workers)

with mp.Pool(num_workers) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, \
                                    args=[f_2d, L, N, obs, Sigma_eta, mean_prior, Sigma_prior, grf], \
                                    pool=pool)
    thin_by = 100
    for i in tqdm(range(100)):
        sampler.run_mcmc(p0, thin_by)
        theta_samples=sampler.get_last_sample()
        #
        p0 = theta_samples
        
chain = sampler.get_chain()
samples=chain.reshape(-1,N_KL)
F=grf.spectral2grid_vectorized(samples)
mean_sol2d = np.mean(F,axis =0)
var_sol2d = np.var(F,axis=0)
with open('data/darcy/test.npy', 'wb') as f:
    np.save(f, obs)
    np.save(f, ref)
    np.save(f, mean_sol2d)
    np.save(f, var_sol2d)



print('Generation Started')


N_training =10000
N_test = 100

obs_noiseless_all = []
logk_2d_all = []

for i in tqdm(range(N_training)):
    theta = np.random.normal(0, 1, N_KL)
    obs_noiseless,logk_2d = generate_single(theta,L,N,N_KL,grf)
    obs_noiseless_all.append(obs_noiseless)
    logk_2d_all.append(logk_2d)
logk_2d_all=np.array(logk_2d_all)
obs_noiseless_all=np.array(obs_noiseless_all)
with open('data_set_10000.npy', 'wb') as f:
    np.save(f, logk_2d_all)
    np.save(f, obs_noiseless_all)

obs_noiseless_all = []
logk_2d_all = []
for i in tqdm(range(N_test)):
    theta = np.random.normal(0, 1, N_KL)
    obs_noiseless,logk_2d = generate_single(theta,L,N,N_KL,grf)
    obs_noiseless_all.append(obs_noiseless)
    logk_2d_all.append(logk_2d)
logk_2d_all=np.array(logk_2d_all)
obs_noiseless_all=np.array(obs_noiseless_all)
with open('data_set_100.npy', 'wb') as f:
    np.save(f, logk_2d_all)
    np.save(f, obs_noiseless_all)