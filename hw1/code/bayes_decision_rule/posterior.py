import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer
    prior = np.zeros((C, 1))
    for i in range(0, C):
        prior[i] = x.sum(axis=1)[i] / total

    px = np.zeros((1, N))
    for j in range(0, N):
        for i in range(0, C):
                px[0][j] += prior[i] * l[i][j]

    for j in range(0, N):
        for i in range(0, C):
                p[i][j] = l[i][j] * prior[i] / px[0][j]

    # end answer
    
    return p
