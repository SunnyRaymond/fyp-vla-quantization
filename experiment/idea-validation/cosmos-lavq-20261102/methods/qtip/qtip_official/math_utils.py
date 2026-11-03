# Vendored verbatim from Cornell-RelaxML/qtip @ e90c6688c8dfae326a3a81b5eb032db7c6680ec0,
# lib/utils/math_utils.py (GPL-3.0, see ../LICENSE): block_LDL, regularize_H. flat_to_sym (Hessian file
# format) is not needed and not vendored.
import math

import torch


def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b

    try:
        L = torch.linalg.cholesky(H)
    except:
        return None

    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n).contiguous()

    L.view(m, b, m,
           b).permute(0, 2, 1,
                      3)[torch.arange(m), torch.arange(m)] = torch.stack(
                          [torch.eye(b, device=L.device, dtype=H.dtype)] * m)

    return (L, D.to(DL.device))


def regularize_H(H, sigma_reg):
    diagmean = torch.diag(H).mean()
    H /= diagmean
    idx = torch.arange(len(H))
    H[idx, idx] += sigma_reg
    return H * diagmean
