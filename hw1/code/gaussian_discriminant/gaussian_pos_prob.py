import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    # M行N列，N个样本数据，每个样本有M个特征
    N = X.shape[1]
    # (xi,yi)，当x=xi时候，对应不同的y1, y2, ..., yk
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    l = np.zeros((N, K))
    for i in range(0, N):
        p_sum = 0
        for j in range(0, K):
            # likelihood
            l[i][j] = (1/(2*np.pi*np.sqrt(np.linalg.det(Sigma[:,:,j])))*(np.exp(-0.5*np.dot(np.dot((X[:,i]-Mu[:,j]).T,Sigma[:,:,j].T),(X[:,i]-Mu[:,j])))))
            p_sum += l[i][j]*Phi[j]
        for j in range(0, K):
            # p(y=k|x) = p(x|y=k)*p(y=k) / Sigma(i:1 to K)(p(x|y=ki)*p(y=ki))
            p[i][j] = l[i][j]*Phi[j]/p_sum  
    # end answer
    
    return p
    