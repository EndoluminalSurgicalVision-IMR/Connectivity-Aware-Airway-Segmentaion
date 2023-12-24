Local Sensitive Distance Transform
https://arxiv.org/pdf/2209.08355.pdf

3 steps:
1. differentiable binarization (or pseudo binarization):
set binarization: (y >= 0.5) + y - y.detach() (y after sigmoid(x))

2. sigmoid with large temperature (sigmoid(x*10))
soft_dilate(y_bin) - soft_erode(y_bin) => contours in the paper they multiply by y

3. convolutional distance transform layer
euclidean weights in a convolutional kernel w_euclidean
use of log-sum-exponential to create a differentiable minimum
L_beta(d1,...,dn) = 1/beta * log(exp(beta * d1) + ... + exp(beta * dn))
they state that min(d1,...dn) = lim beta->-inf L_beta(d1,...,dn)
distance transform is then simply:
lim beta->inf 1/beta log( exp(beta * x).conv(w_euclidean) )

```
import torch
import torch.nn.functional as F
import math 


def get_distance_transform_edt_kernel(k, device):
    ids = [torch.arange(k)-k//2 for item in [k,k,k]]
    ids = [v.to(device) for v in ids]
    grid = torch.stack(torch.meshgrid(*ids, indexing='ij'), -1).float()
    w = torch.linalg.norm(grid.reshape(-1, 3), axis=-1).reshape(1,1,k,k,k) 
    return w


def morpho_grad_soft(xth):
    z_fg = F.max_pool3d(xth, 3, 1, 1) + F.max_pool3d(-xth, 3, 1, 1)
    return z_fg


def log_conv_exp(x, kernel, beta):
    # gray level dilation via log-conv-exp with beta >> 0
    c = kernel.max() # max trick for numerical stability
    gamma = 1./beta
    k_beta = torch.exp(kernel / gamma - c)
    y_beta = F.conv3d(x, k_beta, stride=1, padding='same')
    y_beta[y_beta==0] = 1 # watch out for true zeros
    d_max = gamma * (c + torch.log(y_beta))
    return d_max 


def distance_transform_via_log_conv_exp(x, k=33, beta=-6):
    kernel = get_distance_transform_edt_kernel(k, x.device)
    z_fg = morpho_grad_soft(x)
    out_fltr = log_conv_exp(z_fg, kernel=kernel, beta=beta) + 1
    distances = (out_fltr * x).clip(0)
    return distances
```