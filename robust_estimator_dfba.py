# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
    Robust estimator for Federated Learning.
'''
import numpy as np
# import cupy as np
import torch
import math
MAX_ITER = 100
ITV = 1000



def power_iteration(mat, iterations, device):
    dim = mat.shape[0]
    u = torch.randn((dim, 1)).to(device)
    for _ in range(iterations):
        u = mat @ u / torch.linalg.norm(mat @ u) 
    eigenvalue = u.T @ mat @ u
    return eigenvalue, u
def randomized_agg_forced(data, eps_poison=0.2, eps_jl=0.1, eps_pow = 0.1, device = 'cuda:0', seed=12):
    # print(data.shape)
    n = int(data.shape[0])
    feature_shape = data[0].shape
    n_dim = int(np.prod(np.array(feature_shape)))
    res =  _randomized_agg(data, eps_poison, eps_jl, eps_pow, 1, 10**-5, device, forced=True, seed=seed) # set threshold for convergence as 1*10**-5 (i.e. float point error)
    return res



def _randomized_agg(data, eps_poison=0.2, eps_jl=0.1, eps_pow = 0.1, threshold = 20, clean_eigen = 10**-5, device = 'cuda:0', forced=False, seed=None):
    if seed:
        torch.manual_seed(seed)
    
    n = int(data.shape[0])
    data = data.to(device)
    
    d = int(math.prod(data[0].shape))
    data_flatten = data.reshape(n, d)
    data_mean = torch.mean(data_flatten, dim=0)
    data_sd = torch.std(data_flatten, dim=0)
    data_norm = (data_flatten - data_mean)/data_sd
    
    k = min(int(math.log(d)//eps_jl**2), d)
    
    A = torch.randn((d, k)).to(device)
    # print(A.shape)
    A = A/(k**0.5)

    Y = data_flatten @ A # n times k
    Y = Y.to(device)
    power_iter_rounds = int(- math.log(4*k)/(2*math.log(1-eps_pow)))
    clean_eigen = clean_eigen * d/k
    old_eigenvalue = None
    for _ in range(max(int(eps_poison*n), 10)):
        Y_mean = torch.mean(Y, dim=0)
        Y = (Y - Y_mean)
        Y_cov = torch.cov(Y.T)
        Y_sq = Y_cov
            
        eigenvalue, eigenvector = power_iteration(Y_sq, power_iter_rounds, device)

        proj_Y = torch.abs(Y @ eigenvector )
        proj_Y = torch.flatten(proj_Y)
        if forced and old_eigenvalue and abs(old_eigenvalue - eigenvalue) < 10**-5:
            print('converge')
            break

        if sum([i > 0.5 for i in proj_Y/torch.max(proj_Y)]) > len(proj_Y)*(1-2*eps_poison):
            print('new_criteria')
            break 
        old_eigenvalue = eigenvalue
        
        uniform_rand = torch.rand(proj_Y.shape).to(device)
        kept_idx = uniform_rand > (proj_Y/torch.max(proj_Y))
        Y = Y[kept_idx]
        data = data[kept_idx]
    out = torch.mean(data, dim=0)
    del data
    return out
