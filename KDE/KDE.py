import numpy as np

def guassian_kernel(x, h = 1):
    """
    Guassian Kernel based on wikipedia 
    h is bandwidth
    """
    return 1/np.sqrt(2 * np.pi * h ** 2) * np.exp(-(x**2) / (2*(h**2)))

class KDE(object):
    """
    Multi-Dim KDE
    data: Data
    kernel: kernel function should be like the ones above
    
    Uses silverman's rule of thumb to add choose the bandwidth
    """
    
    def __init__(self,data,kernel):
        self.dataset = data
        self.kernel = kernel
        self.silverman()
        self.kde = self.get_kde()
        
    def change_kernel(self, kernel):
        self.silverman()
        self.kde = self.get_kde()
        
    def k_H(self,y,x,d):
        H_det = np.linalg.det(self.H) ** (-1/2)
        return H_det * np.prod([self.kernel(y[j] - x[j]) / self.H[j,j] for j in range(d)])

    def k(self,y,x):
        return self.kernel(np.linalg.norm(y - x, ord = 2))
    
    def get_kde(self):
        n,d = self.dataset.shape
        return lambda y : 1 / n * sum([self.k_H(y,self.dataset[i],d) for i in range(n)])
    
    def eval(self, data):
        n,d = data.shape
        density = np.zeros(n)
        for i in range(n):
            density[i] = self.kde(data[i])
        return density
    
    def silverman(self):
        n,d = self.dataset.shape
        H = np.zeros((d,d))
        const = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
        for i in range(d):
            H[i,i] = const * np.std(self.dataset[:,i])
        self.H = H

