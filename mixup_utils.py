#
# Main mixup functions in one file.
#

import numpy as np
import torch

#
# Mixup two examples following the original implementation.
# 
#
def mixup_data(x, y, a, s):
    #l = np.random.beta(a, a, [x.size()[0], 1])
    tmpl = np.random.beta(a, a)
    l = np.ones((x.size()[0], 1)) * tmpl
    t = torch.from_numpy(np.tile(l[..., None, None], [1, 3, s, s])).float()#.cuda()
    l = torch.from_numpy(l[:,0])
    b = x.size()[0]
    i = torch.randperm(b)
    nx = t * x + (1 - t) * x[i,:]
    ny1, ny2 = y, y[i]
    return nx, [ny1, ny2], [l, 1-l]

#
# Alternative unused Mixup.
#
def mixup_std(x, y, a):
    idxs = torch.randperm(x.size(0))
    x2 = x[idxs]
    y2 = y[idxs]
    lam = torch.FloatTensor([np.random.beta(a, a)])
    x = x * lam + x2 * (1-lam)
    return x, [y, y2], [lam, 1-lam]

#
# Mixup with Remix for imbalanced learning.
#
def mixup_data_remix(x, y, a, cd, kappa=1.2, tau=0.5):
    l = np.random.beta(a, a, [x.size()[0], 1])
    t = torch.from_numpy(np.tile(l[..., None, None], [1, 3, 32, 32])).float().cuda()
    l = l[:,0]
    b = x.size()[0]
    i = torch.randperm(b)
    nx = t * x + (1 - t) * x[i,:]
    ny1, ny2 = y, y[i]
    l1 = torch.from_numpy(l)
    l2 = torch.full_like(torch.from_numpy(l), 0)
    for j in range(len(l2)):
        ratio = float(cd[ny1[j]]) / float(cd[ny2[j]])
        if ratio >= kappa and l[j] < tau:
            l2[j] = 0
        elif ratio <= 1./kappa and 1-l[j] < tau:
            l2[j] = 1
        else:
            l2[j] = l[j]
    return nx, [ny1, ny2], [l1, l2]
