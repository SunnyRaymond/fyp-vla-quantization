# Vendored from https://github.com/Cornell-RelaxML/quip-sharp
# commit 1d8f873e9a2a8b86b12bb1064c312c5689b77d98
# Copyright (c) the QuIP# authors (Tseng, Chee, Sun, Kuleshov, De Sa). Licensed under GPL-3.0; see LICENSE in
# this directory. Modified for the Fast-WAM A16 baseline port (2026-10-28); every change is marked "PORT:".
#
# Contents, in upstream order:
#   lib/utils/misc.py::clean
#   lib/utils/math_utils.py::block_LDL, regularize_H
#   lib/codebook/latticee8_padded12.py: E8P grid construction + E8P12_codebook (quantization side)
#   lib/codebook/latticee8_padded12_rvq3bit.py: E81B grid + E8P12RVQ3B_codebook (quantization side)
#   lib/algo/quip.py: RHT_H, RHT_W, incoherence_preprocess, incoherence_process, low_rank_preprocess,
#                     low_rank_process, LDLQ, LDLQ_buffered, LDLQ_buffered_lowmem, quantize
#   lib/algo/quip.py::quantize_linear minus its file I/O (quantize_linear_inmem)
# Stripped: model loading, data, fine-tuning, glog, quiptools_cuda inference kernels (by_idxs, Quantized*Linear),
# the 4-bit codebook. The grid construction shared verbatim by the 2- and 3-bit codebook files is vendored once.
import copy
import gc
import logging

import torch
from torch import nn

import quipsharp_hadamard

# PORT: glog -> stdlib logging (same call sites, same messages).
glog = logging.getLogger("quipsharp")


# ---------------------------------------------------------------------------------------------------------------
# lib/utils/misc.py
# ---------------------------------------------------------------------------------------------------------------
def clean():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------
# lib/utils/math_utils.py
# ---------------------------------------------------------------------------------------------------------------
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

    L = L.reshape(n, n)
    return (L, D.to(DL.device))


def regularize_H(H, n, sigma_reg):
    H.div_(torch.diag(H).mean())
    idx = torch.arange(n)
    H[idx, idx] += sigma_reg
    return H


# ---------------------------------------------------------------------------------------------------------------
# lib/codebook/latticee8_padded12.py  (identical copies of these grid functions live in latticee8_padded12_rvq3bit.py)
# ---------------------------------------------------------------------------------------------------------------
_E8P_CODESZ = 8


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c, i] *= -1
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25
    return synth_codebook, torch.arange(1 << 16), parity_idx


_E8P_PACKED_ABS_CACHED = get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS_CACHED)


class E8P12_codebook(nn.Module):

    def __init__(self, inference=False):
        super(E8P12_codebook, self).__init__()
        self.opt_scale = 1.03
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int64
        self.packsz = 4
        self.pack_out = False
        self.version = 1

        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED)

        if not inference:
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1)**2)
            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            grid_part = grid_part[
                torch.where(
                    ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                    (grid_part[:, :7].min(dim=-1).values >= -0.5)
                )[0]]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1)**2)
            abs_grid = get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)
            self.register_buffer(
                'part_abs_map',
                self.round(grid_part.abs(), abs_grid,
                           abs_grid.norm(dim=-1)**2)[1])
            self.register_buffer('bit_map', 2**torch.arange(8))
            '''
            self.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(8), torch.eye(8)).rsample([2000000]).cuda()
            for s in torch.arange(0.8, 1.2, 0.01):
                print(s, ((self.quantize(samples*s, False)/s - samples).norm(dim=-1)**2).mean())
            exit()
            '''

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part,
                                     self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:,
                                                   [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    def quantize(self, X, return_idx=True, **kwargs):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(
            X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4,
                                 minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)

        if return_idx:
            return final_vals, final_idx

        return final_vals

    def maybe_pack_idxs(self, idxs):
        m, n = idxs.shape
        idxs = idxs.view(m // 2, 2, (n * 8) // 16,
                         2).transpose(1, 2).contiguous()

        abs32 = (idxs[:, :, 0, 0] >> 8) + \
            ((idxs[:, :, 1, 0] >> 8) << 8) + \
            ((idxs[:, :, 0, 1] >> 8) << 16) + \
            ((idxs[:, :, 1, 1] >> 8) << 24)

        sign32 = torch.zeros(abs32.shape,
                             dtype=abs32.dtype,
                             device=abs32.device)
        for i in range(4):
            wt = idxs[:, :, i % 2, i // 2]
            for j in range(8):
                sign32 += ((wt >> j) & 1) << (4 * j + i)

        output = (sign32 << 32) + abs32
        output = output.reshape(m // 16, 8, n // 8,
                                4).transpose(1, 2).contiguous()
        return output.view(m, n // 4)


# ---------------------------------------------------------------------------------------------------------------
# lib/codebook/latticee8_padded12_rvq3bit.py
# ---------------------------------------------------------------------------------------------------------------
def get_e81bgrid():
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2

    gintr = torch.cartesian_prod(*[intr] * 8)
    ghintr = torch.cartesian_prod(*[hintr] * 8)

    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1)**2 <= 2

    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]

    norm4 = torch.tensor([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [-2, 0, 0, 0, 0, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, 0, 0, 0, 0, 0],
        [0, 0, 0, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, -2, 0],
        #[0, 0, 0, 0, 0, 0, 0, -2],
    ])

    e8 = torch.concat([e8, norm4], dim=0)

    return e8


_E81B_CACHED = get_e81bgrid()
_E81B_NORM_CACHED = _E81B_CACHED.norm(dim=-1)**2


class E8P12RVQ3B_codebook(nn.Module):

    def __init__(self, inference=False):
        super(E8P12RVQ3B_codebook, self).__init__()
        self.opt_scale = 0.98
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int64
        self.packsz = 8 / 3  # fudged, the second half of Qidxs is the residual
        self.pack_out = False
        self.version = 0
        self.opt_resid_scale = 2.04

        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED)
        self.register_buffer('e81b_grid', _E81B_CACHED)

        if not inference:
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1)**2)
            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            grid_part = grid_part[
                torch.where(
                    ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                    (grid_part[:, :7].min(dim=-1).values >= -0.5)
                )[0]]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1)**2)
            abs_grid = get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)
            self.register_buffer(
                'part_abs_map',
                self.round(grid_part.abs(), abs_grid,
                           abs_grid.norm(dim=-1)**2)[1])
            self.register_buffer('bit_map', 2**torch.arange(8))
            self.register_buffer('e81b_grid_norm', _E81B_NORM_CACHED)
            '''
            self.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(8), torch.eye(8)).rsample([1000000]).cuda()

            from scipy.optimize import minimize_scalar
            def opt_err_cvx(fn):
                res = minimize_scalar(fn, bounds=(0.1, 100))
                scale = res.x.item()
                err = res.fun
                return err, scale
            
            for s in torch.arange(0.8, 1.1, 0.01):
                def test_sr(sr):
                    hatwr = self.quantize(samples*s, False, sr)/s
                    return (hatwr - samples).norm().cpu()**2
                err, scale = opt_err_cvx(test_sr)
                print(s, scale, err)
            exit()
            '''

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part,
                                     self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:,
                                                   [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    def quantize_e8p(self, X):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(
            X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4,
                                 minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)
        return final_vals, final_idx

    def quantize(self, X, return_idx=True, resid_scale_override=-1, **kwargs):
        init_vals, init_idxs = self.quantize_e8p(X)
        resid_scale = resid_scale_override if resid_scale_override > 0 else self.opt_resid_scale

        resid = (X - init_vals) * resid_scale
        resid_vals, resid_idxs = self.round(resid, self.e81b_grid,
                                            self.e81b_grid_norm)
        final_vals = init_vals + resid_vals / resid_scale
        final_idxs = (init_idxs << 16) + resid_idxs
        if return_idx:
            return final_vals, final_idxs
        return final_vals

    def maybe_pack_idxs(self, idxs):
        init_idxs = idxs >> 16
        resid_idxs = idxs & ((1 << 16) - 1)

        def pack_e8p(idxs):
            m, n = idxs.shape
            idxs = idxs.view(m // 2, 2, (n * 8) // 16,
                             2).transpose(1, 2).contiguous()

            abs32 = (idxs[:, :, 0, 0] >> 8) + \
                ((idxs[:, :, 1, 0] >> 8) << 8) + \
                ((idxs[:, :, 0, 1] >> 8) << 16) + \
                ((idxs[:, :, 1, 1] >> 8) << 24)

            sign32 = torch.zeros(abs32.shape,
                                 dtype=abs32.dtype,
                                 device=abs32.device)
            for i in range(4):
                wt = idxs[:, :, i % 2, i // 2]
                for j in range(8):
                    sign32 += ((wt >> j) & 1) << (4 * j + i)

            output = (sign32 << 32) + abs32
            output = output.reshape(m // 16, 8, n // 8,
                                    4).transpose(1, 2).contiguous()
            return output.view(m, n // 4)

        def pack_e81b(idxs):
            accum = idxs[:, 0::8]
            for i in range(1, 8):
                accum += idxs[:, i::8] << (8 * i)
            return accum

        return torch.concat(
            [pack_e8p(init_idxs), pack_e81b(resid_idxs)], dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# lib/algo/quip.py
# ---------------------------------------------------------------------------------------------------------------
def RHT_H(H, SU):
    return utils.matmul_hadUt(utils.matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    return utils.matmul_hadUt(utils.matmul_hadUt(W.T * SV).T * SU)


def incoherence_preprocess(H, W, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    device = H.device
    (m, n) = W.shape

    def _dump(Hr, Lhr, msg=''):
        torch.save(Hr, f"{args.save_pfx}/Hr_debug_fft.pt")
        torch.save(Lhr, f"{args.save_pfx}/Lhr_debug_fft.pt")
        raise Exception(msg)

    # diagonally rescale W,H to minimize proxy loss
    scaleWH = None
    Wr = W
    Hr = H
    if args.rescale_WH:
        Hr = H / H.abs().max()
        diagH = torch.diag(Hr)
        diagW2 = torch.diag(W.T @ W)
        diagH = torch.clamp(diagH, min=1e-8)
        diagW2 = torch.clamp(diagW2, min=1e-8)
        scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
        scaleWH = scaleWH.clamp(min=1e-8)
        Wr = Wr * scaleWH[None, :]
        Hr = Hr / scaleWH[None, :]
        Hr = Hr / scaleWH[:, None]
        scaleWH = scaleWH.cpu()

    # randomized hadamard transformation on H, W
    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        Wr = RHT_W(Wr, SU, SV)
    # randomized kronecker product on H, W
    elif args.incoh_mode == "kron":
        SU = utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
        SV = utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
        Hr = SU @ Hr @ SU.T
        Wr = SV @ Wr @ SU.T
    else:
        raise NotImplementedError
    SV = SV.cpu()
    SU = SU.cpu()

    # PORT: torch>=1.13 raises on a non-PD input instead of returning NaNs; map the exception onto the official
    # "non-finite factor -> return None -> retry in fp64" path.
    try:
        Lhr = torch.linalg.cholesky(Hr)
    except torch.linalg.LinAlgError:
        return None
    if not torch.all(torch.isfinite(Lhr)):
        return None

    Wr = Wr.to(device)

    return Lhr, Hr, Wr, SU, SV, scaleWH


def incoherence_process(hatWr, SU, SV, scaleWH, args):
    device = hatWr.device
    # reverse hadamard transformation
    if args.incoh_mode == 'had':
        hatWr = (utils.matmul_hadU(
            (utils.matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
    # reverse kronecker product
    elif args.incoh_mode == 'kron':
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    # reverse rescale W,H
    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr


def low_rank_preprocess(Wr, Hr, Lhr, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    if args.full_svd:
        svdZ = torch.linalg.svd(Wr.to(torch.float64) @ Lhr.to(torch.float64),
                                full_matrices=False)
        Hr -= (Lhr.to(torch.float64) @ svdZ.Vh.T[:, :args.lora_rank] @ \
                   svdZ.Vh[:args.lora_rank] @ Lhr.to(torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (svdZ.U[:, :args.lora_rank] @ svdZ.U.T[:args.lora_rank] @ Wr.to(
            torch.float64)).to(dtype_)
    else:
        U_lrz, S_lrz, V_lrz = torch.svd_lowrank(
            Wr.to(torch.float64) @ Lhr.to(torch.float64),
            q=2 * args.lora_rank,
            niter=10)
        U_lrz = U_lrz[:, :args.lora_rank]
        V_lrz = V_lrz[:, :args.lora_rank]
        Hr -= (Lhr.to(torch.float64) @ V_lrz @ V_lrz.T @ Lhr.to(
            torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (U_lrz @ U_lrz.T @ Wr.to(torch.float64)).to(dtype_)
    return Wr, Hr


def low_rank_process(Wo, hatWr, Lhr, args):
    # invLhr = torch.linalg.inv(Lhr)
    # assert torch.isfinite(invLhr).all()

    svdRZ = torch.linalg.svd((Wo - hatWr) @ Lhr, full_matrices=False)
    A = svdRZ.U[:, :args.lora_rank]
    # B = torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank] @ invLhr
    B = torch.linalg.solve_triangular(
        Lhr,
        torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank],
        upper=False,
        left=False)
    assert torch.isfinite(A).all() and torch.isfinite(B).all()

    svdB = torch.linalg.svd(B, full_matrices=False)
    A = (A @ svdB.U @ torch.diag(svdB.S.sqrt())).half()
    B = (torch.diag(svdB.S.sqrt()) @ svdB.Vh).half()

    hatWr = hatWr.to(A.device) + \
        (A @ B).to(torch.float64 if args.use_fp64 else torch.float32)
    return hatWr, A, B


def LDLQ(Wr, Hr, L, D, cb, args):
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    for k in reversed(range(n // cb.codesz)):
        WXWX = Wr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + \
            (Wr[:, (cb.codesz * (k + 1)):n] - hatWr[:, (cb.codesz * (k + 1)):n]) @ \
            L[(cb.codesz * (k + 1)):n, (cb.codesz * k):(cb.codesz * (k + 1))]
        hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))], Qidxs[:, k] = \
            cb.quantize(WXWX, resid_scale_override=args.resid_scale_override)
    for ie in range(args.quip_tune_iters):
        for k in reversed(range(n // cb.codesz)):
            WXWX = hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + (Wr - hatWr) @ \
                Hr[:, (cb.codesz * k):(cb.codesz * (k + 1))] @ \
                torch.linalg.inv(Hr[(cb.codesz * k):(cb.codesz * (k + 1)),
                                    (cb.codesz * k):(cb.codesz * (k + 1))])
            hatWr[:, (cb.codesz *
                      k):(cb.codesz * (k + 1))], Qidxs[:, k] = cb.quantize(
                          WXWX, resid_scale_override=args.resid_scale_override)

    return hatWr, Qidxs


def LDLQ_buffered(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of codesz (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    assert buf_cols % cb.codesz == 0
    assert n % buf_cols == 0
    buf_size = buf_cols // cb.codesz

    hatWr_T = torch.zeros(n, m, dtype=Hr.dtype, device=Hr.device)
    Qidxs_T = torch.zeros(n // cb.codesz,
                          m,
                          dtype=cb.idx_dtype,
                          device=Hr.device)

    device = Wr.device
    Wr = Wr.cpu()
    Hr = Hr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)
    Hr_T = Hr.T.contiguous().to(device)

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr_T = Wr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col].contiguous()
        b_prod = prod_cache[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)].T @ \
                (b_Wr_T[cb.codesz * (i + 1):] - b_hatWr_T[cb.codesz * (i + 1):]) + \
                b_prod[cb.codesz * i : cb.codesz * (i + 1)]
            q_out = cb.quantize(WXWX.T,
                                resid_scale_override=args.resid_scale_override)
            b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
            b_Qidxs_T[i] = q_out[1]

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta_T = Wr_T - hatWr_T
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Hr_T = Hr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta_T = delta_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        torch.linalg.inv(b_Hr_T[cb.codesz * i : cb.codesz * (i + 1), Hr_offset + cb.codesz * i : Hr_offset + cb.codesz * (i + 1)].T).T @ b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                else:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        (1/b_Hr_T[i, Hr_offset + i]) * b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] += b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                        WXWX.T,
                        return_idx=False,
                        resid_scale_override=args.resid_scale_override).T
                else:
                    q_out = cb.quantize(
                        WXWX.T, resid_scale_override=args.resid_scale_override)
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
                    b_Qidxs_T[i] = q_out[1]

                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] -= b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]
            hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                    cur_col] = b_hatWr_T
            Qidxs_T[cur_col - buf_size:cur_col] = b_Qidxs_T

        del delta_T, b_hatWr_T, b_Hr_T, b_delta_T, b_Qidxs_T, Hr_offset
        utils.clean()

    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()


def LDLQ_buffered_lowmem(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of code_col (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    assert n % buf_cols == 0 and buf_cols % cb.codesz == 0
    buf_size = buf_cols // cb.codesz

    # quip
    prod_cache = torch.zeros(m, n, dtype=Wr.dtype, device=Wr.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr = Wr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr = hatWr[:,
                        cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_prod = prod_cache[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                (b_Wr[:, cb.codesz * (i + 1):] - b_hatWr[:, cb.codesz * (i + 1):]) @ \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)] + \
                b_prod[:, cb.codesz * i : cb.codesz * (i + 1)]
            b_hatWr[:, cb.codesz * i:cb.codesz *
                    (i + 1)], b_Qidxs[:, i] = cb.quantize(
                        WXWX, resid_scale_override=args.resid_scale_override)
        prod_cache += (b_Wr - b_hatWr) @ b_L

    del b_Wr, b_hatWr, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta = Wr - hatWr
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr = hatWr[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Hr = Hr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta = delta[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    inv = torch.linalg.inv(
                        b_Hr[Hr_offset + cb.codesz * i:Hr_offset + cb.codesz *
                             (i + 1), cb.codesz * i:cb.codesz * (i + 1)])
                else:
                    inv = 1 / b_Hr[Hr_offset + i:Hr_offset + i + 1, i:i + 1]

                WXWX = b_hatWr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                    delta @ b_Hr[:, cb.codesz * i : cb.codesz * (i + 1)] @ inv

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] += b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr[:,
                            cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                                WXWX,
                                return_idx=False,
                                resid_scale_override=args.resid_scale_override)
                else:
                    b_hatWr[:, cb.codesz * i:cb.codesz *
                            (i + 1)], b_Qidxs[:, i] = cb.quantize(
                                WXWX,
                                resid_scale_override=args.resid_scale_override)

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] -= b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]
        del delta, b_hatWr, b_Hr, b_delta, b_Qidxs, Hr_offset
        utils.clean()

    return hatWr, Qidxs


def quantize(H_orig, W_orig, rank, codebook_orig, args, device='cpu'):
    orig_device = H_orig.device
    W_orig_dtype = W_orig.dtype
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    (m, n) = W_orig.shape

    H = H_orig.clone().to(dtype_).to(device)
    W = W_orig.clone().to(dtype_).to(device)
    codebook = copy.deepcopy(codebook_orig).to(dtype_)

    assert (m % 2 == 0)
    assert (n % 4 == 0)
    assert (torch.all(torch.isfinite(H.cpu())))
    assert (torch.all(torch.isfinite(W.cpu())))

    # incoherence preprocessing
    incoh_out = incoherence_preprocess(H, W, args)
    if incoh_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('incoherence_preprocess failed, recomputing in fp64')
        del H, W, codebook
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    Lhr, Hr, Wr, SU, SV, scaleWH = incoh_out
    del incoh_out
    utils.clean()

    glog.info(f'mean square of W: {W.square().mean()}')
    glog.info(f'mean square of Wr: {Wr.square().mean()}')
    glog.info(f'difference between Hr and Hr.T: {((Hr - Hr.T).abs().max())}')
    glog.info(f'max abs of Hr: {((Hr.abs().max()))}')
    glog.info(f'min diag of Lhr: {Lhr.diag().min().item()}')

    Wo = Wr.clone()

    # remove low rank components before LDLQ
    if args.lora_rank > 0:
        Wr, Hr = low_rank_preprocess(Wr, Hr, Lhr, args)

    # block LDL
    block_LDL_out = utils.block_LDL(Hr, codebook.codesz)
    if block_LDL_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('block_LDL failed, recomputing in fp64')
        del H, W, codebook, Lhr, Hr, Wr, SU, SV, scaleWH, Wo
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    L, D = block_LDL_out
    del block_LDL_out
    del H_orig, W_orig, codebook_orig
    utils.clean()

    # LDLQ
    Wscale = Wr.square().mean().sqrt()
    if args.scale_override > 0:
        Wscale /= args.scale_override
    else:
        Wscale /= codebook.opt_scale
    Wr = Wr / Wscale
    codebook = codebook.to(device)
    # PORT: buf_cols was hard-coded to 128, which requires in_features % 128 == 0. Read it from args (default 128,
    # i.e. unchanged for every Fast-WAM width that is a multiple of 128); the wrapper passes gcd(n, 128) otherwise.
    # The buffer size only regroups the same LDLQ arithmetic.
    buf_cols = getattr(args, 'buf_cols', 128)
    if args.no_use_buffered:
        hatWr, Qidxs = LDLQ(Wr, Hr, L, D, codebook, args)
    elif args.lowmem_ldlq or args.use_fp64:
        hatWr, Qidxs = LDLQ_buffered_lowmem(Wr,
                                            Hr,
                                            L,
                                            D,
                                            codebook,
                                            args,
                                            buf_cols=buf_cols)
    else:
        hatWr, Qidxs = LDLQ_buffered(Wr,
                                     Hr,
                                     L,
                                     D,
                                     codebook,
                                     args,
                                     buf_cols=buf_cols)

    Wr = Wr.cpu()
    Hr = Hr.cpu()
    L = L.cpu()
    D = D.cpu()
    del Wr, Hr, L, D
    utils.clean()

    hatWr = hatWr * Wscale

    # low rank correction
    if args.lora_rank > 0:
        hatWr, A, B = low_rank_process(Wo, hatWr, Lhr, args)
        A = A.half().cpu()
        B = B.half().cpu()
    else:
        A, B = None, None

    # reverse incoherence process
    hatW = incoherence_process(hatWr, SU, SV, scaleWH, args)

    Qidxs = codebook.maybe_pack_idxs(Qidxs)

    attr = {
        'Qidxs': Qidxs.to(orig_device),
        'A': A,
        'B': B,
        'SU': SU.to(torch.float16).to(orig_device),
        'SV': (SV * Wscale.to(SV.device)).to(
            torch.float16).to(orig_device),  # fuse Wscale into SV
        'scaleWH': scaleWH,
    }

    utils.clean()

    return hatW.to(W_orig_dtype).to(orig_device), attr


def quantize_linear_inmem(weights, H, cb, args, device='cpu'):
    """PORT: body of quip.py::quantize_linear with the Hessian file load, the save_path short-circuit, torch.save and
    show_metrics logging removed. `H` is the full second moment E[x x^T] (the upstream file stores the covariance
    and adds mu mu^T back right here, giving the same matrix). Returns (hatW, attr) instead of saving attr."""
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    shapes = [_.shape for _ in weights]
    scales = [_.to(dtype_).square().mean().sqrt() for _ in weights]

    n = H.shape[0]
    W = torch.vstack([
        weights[i].to(dtype_) / scales[i] for i in range(len(weights))
    ]).to(dtype_)
    H = regularize_H(H, n, args.sigma_reg)
    hatW, attr = quantize(H, W, args.lora_rank, cb, args, device)
    if len(scales) == 1:
        # fuse single scale into SV too
        attr['SV'] *= scales[0]
        scales = [1.0]
    attr.update({
        'fused': len(shapes) > 1,
        'shapes': shapes,
        'scales': scales,
    })
    return hatW, attr


# PORT: `from lib import utils` -> namespace with the helpers the vendored functions call through `utils.`.
class utils:
    matmul_hadU = staticmethod(quipsharp_hadamard.matmul_hadU)
    matmul_hadUt = staticmethod(quipsharp_hadamard.matmul_hadUt)
    block_LDL = staticmethod(block_LDL)
    regularize_H = staticmethod(regularize_H)
    # PORT: utils.clean() is memory hygiene only. Its gc.collect() costs ~40+ ms per call and the official path
    # calls it ~24 times per layer (it dominated small layers); there are no large-tensor reference cycles to
    # collect, so only torch.cuda.empty_cache() is kept. No numerical effect.
    clean = staticmethod(quipsharp_hadamard.utils.clean)
