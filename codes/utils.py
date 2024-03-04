# Author Zhongjian
# Generate 2d identity data.  
import numpy as np
import scipy.sparse as sp


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

    seq_pairs = seq_pairs[np.argsort(seq_pairs_mag, kind="stable")]
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
        
    def generate(self, theta):
        N, N_KL = self.N, self.N_KL
        lam, phi = self.lam, self.phi
        N_theta = len(theta)
        
        assert(N_theta <= N_KL) 
        logk_2d = np.zeros((N, N))
        for i in range(N_theta):
            logk_2d += theta[i] * np.sqrt(lam[i]) * phi[i, :, :]
        return logk_2d
        
    def spectral2grid(self, theta):
        return self.generate(theta)

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
    
    def compute_covmat(self,x,y):
        '''Compute the covariance matrix'''
        # check shape
        assert(x.shape[1]==2)
        assert(y.shape[1]==2)
        # 
        phiX = np.zeros((self.N_KL, x.shape[0]))
        phiY = np.zeros((self.N_KL, y.shape[0]))
        #
        seq_pairs = self.seq_pairs
        for i in range(self.N_KL):
            if (seq_pairs[i, 0] == 0 and seq_pairs[i, 1] == 0):
                phiX[i, :] = 1.0
                phiY[i, :] = 1.0
            elif (seq_pairs[i, 0] == 0):
                phiX[i, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*x[:,1]))
                phiY[i, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*y[:,1]))
            elif (seq_pairs[i, 1] == 0):
                phiX[i, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*x[:,0]))
                phiY[i, :] = np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*y[:,0]))
            else:
                phiX[i, :] = 2*np.cos(np.pi * (seq_pairs[i, 0]*x[:,0])) *  np.cos(np.pi * (seq_pairs[i, 1]*x[:,1]))
                phiY[i, :] = 2*np.cos(np.pi * (seq_pairs[i, 0]*y[:,0])) *  np.cos(np.pi * (seq_pairs[i, 1]*y[:,1]))
        phiX = phiX * self.lam[:,None]
        CXY=np.einsum('ij,ik->jk',phiX,phiY)
        return CXY


'''3d'''


"""
Compute sorted pair (i, j,k), sorted by i**2 + j**2 + k**2
with i>=0 and j>=0 and i+j>0 .. 
These pairs are used for Karhunen–Loève expansion
"""
def compute_seq_pairs_3d(N_KL):
    trunc_Nx = np.int64((N_KL*6)**(1/3.0)) + 1
    
    seq_pairs = np.zeros(((trunc_Nx+1)**3-1, 3), dtype=np.int64)
    seq_pairs_mag = np.zeros(((trunc_Nx+1)**3-1, ), dtype=np.int64)
    seq_pairs_i = 0

    for i in range(trunc_Nx+1):
        for j in range(trunc_Nx+1):
            for k in range(trunc_Nx+1):
                if (i == 0 and j == 0 and k == 0):
                    continue
                seq_pairs[seq_pairs_i, :] = i, j , k
                seq_pairs_mag[seq_pairs_i] = i**2 + j**2 +k **2
                seq_pairs_i += 1
    
    seq_pairs = seq_pairs[np.argsort(seq_pairs_mag, kind="stable")]
    return seq_pairs[0:N_KL, :]


class Gaussian_random_field_3d:
    def __init__(self, xx, N_KL, d=1.0, tau=3.0, sigma =1.0):
        N = len(xx)
        X, Y, Z = np.meshgrid(xx, xx, xx)
        
        seq_pairs = compute_seq_pairs_3d(N_KL)
        
        
        phi = np.ones((N_KL, N, N,N))
        lam = np.zeros(N_KL)
        
        for i in range(N_KL):
            if seq_pairs[i,0] != 0:
                phi[i, :, :, :] = phi[i, :, :, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*X))
            
            if seq_pairs[i,1] != 0:
                phi[i, :, :, :] = phi[i, :, :, :] * np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*Y))

            if seq_pairs[i,2] != 0:
                phi[i, :, :, :] = phi[i, :, :, :] * np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 2]*Z))
            
            lam[i] = sigma**2*(np.pi**2*(seq_pairs[i, 0]**2 + seq_pairs[i, 1]**2+ seq_pairs[i, 2]**2) + tau**2)**(-d)

        self.N = N
        self.N_KL = N_KL
        self.seq_pairs = seq_pairs
        self.lam = lam
        self.phi = phi 
        
    def generate(self, theta):
        N, N_KL = self.N, self.N_KL
        lam, phi = self.lam, self.phi
        N_theta = len(theta)
        
        assert(N_theta <= N_KL) 
        logk_2d = np.zeros((N, N, N))
        for i in range(N_theta):
            logk_2d += theta[i] * np.sqrt(lam[i]) * phi[i, :, :, :]
        return logk_2d
    
    def compute_covmat(self,x,y):
        '''Compute the covariance matrix'''
        # check shape
        assert(x.shape[1]==3)
        assert(y.shape[1]==3)
        # 
        phiX = np.ones((self.N_KL, x.shape[0]))
        phiY = np.ones((self.N_KL, y.shape[0]))
        #
        seq_pairs = self.seq_pairs
        for i in range(self.N_KL):
            if seq_pairs[i,0] != 0:
                phiX[i, :] = phiX[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*x[:,0]))
                phiY[i, :] = phiY[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 0]*y[:,0]))
            if seq_pairs[i,1] != 0:
                phiX[i, :] = phiX[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*x[:,1]))
                phiY[i, :] = phiY[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 1]*y[:,1]))
            if seq_pairs[i,2] != 0:
                phiX[i, :] = phiX[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 2]*x[:,2]))
                phiY[i, :] = phiY[i, :]* np.sqrt(2)*np.cos(np.pi * (seq_pairs[i, 2]*y[:,2]))
        phiX = phiX * self.lam[:,None]
        CXY=np.einsum('ij,ik->jk',phiX,phiY)
        return CXY



class log_writer(object):
    def __init__(self,file_path,print_on_screen=False):
        super(log_writer,self).__init__()
        self.file_path=file_path
        self.print_on_screen=print_on_screen
        wrlog = open(self.file_path,'a')
        wrlog.close()
    def print(self,*strings,newline=True):
        if self.print_on_screen:
            print(strings)
        wrlog = open(self.file_path,'a')
        for string in strings:
            wrlog.write(str(string))
            wrlog.write(' ')
        if newline:
            wrlog.write('\n')
        wrlog.close()
