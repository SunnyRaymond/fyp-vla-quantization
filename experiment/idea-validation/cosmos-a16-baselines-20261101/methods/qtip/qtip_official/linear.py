# Vendored from Cornell-RelaxML/qtip @ e90c6688c8dfae326a3a81b5eb032db7c6680ec0 (GPL-3.0, see ../LICENSE).
#
# quantize_linear(): the per-Linear body of lib/algo/finetune.py::quantize_finetune_decoder_layer, non-TP
# branch (split_for_tp=False), from `cb = cb.to(device).to(orig_dtype)` through the saved dict, with the
# statements kept verbatim. Removed: Hessian file loading (flat_to_sym + mu mu^T; the caller passes
# HR = E[x x^T] directly), QuantizedLinear construction and the per-layer fine-tuning that follows
# (finetune_decoder_layer), torch.save. SU/SV are passed in instead of drawn from the CUDA RNG.
#
# unpack_quip(): lib/utils/data_utils.py::unpack_quip (rcp == 0 branch), returning the QuantizedLinear buffers
# (trellis int16, SU, SV * Wscale in fp32, tlut fp16) instead of copying them into a module.
import torch

from . import ldlq
from .math_utils import block_LDL, regularize_H
from .matmul_had import clean, matmul_hadUt


def quantize_linear(W, HR, SU, SV, cb, args, device, orig_dtype, has_kernel, for_kernel,
                    buf_cols=128):
    """W: [m, n] float32, HR: [n, n] float32 (= E[x x^T]), SU: [n] +-1, SV: [m] +-1.

    args needs: sigma_reg, scale_override, td_x, td_y, K, V. HR is modified in place (as official).
    has_kernel: apply the CUDA-kernel bit swizzle to the packed trellis (official: has_kernel(...)).
    for_kernel: LDLQ's _PERMUTE trellis-order flag (official passes for_kernel=has_kernel).
    """
    dtype_ = W.dtype
    clean()
    cb = cb.to(device).to(orig_dtype)
    (m, n) = W.shape

    HR = regularize_H(HR, args.sigma_reg)

    Wr = matmul_hadUt(matmul_hadUt(W.T.to(device) * SV).T * SU)
    HRr = matmul_hadUt(matmul_hadUt(HR.to(device) * SU).T * SU)

    Wscale = Wr.square().mean().sqrt() / (
        cb.lut.to(torch.float64).square().mean().sqrt().float() *
        args.scale_override)
    Wr /= Wscale

    LRr, _ = block_LDL(HRr, args.td_y)
    diag = torch.arange(n, device=LRr.device)
    LRr[diag, diag] = 0

    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, buf_cols=buf_cols, for_kernel=for_kernel)

    Qidxs = Qidxs.cpu()
    packed = cb.pack_trellis(
        Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                      args.td_y // args.V).transpose(1, 2).reshape(
                          -1, args.td_x * args.td_y // args.V))

    if has_kernel:
        packed = packed.view(torch.uint8).view(-1, 2).flip(
            (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                            args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                (-1, )).contiguous().flatten().view(
                                    torch.int16).reshape(packed.shape)
    else:
        packed = packed.view(torch.int16)

    Wr *= Wscale
    hatWr *= Wscale

    err = torch.trace(
        (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)

    saved = {
        'trellis': packed.cpu(),
        'SU': SU.to(orig_dtype).cpu(),
        'SV': SV.to(orig_dtype).cpu(),
        'Wscale': Wscale,
        'proxy_err': err.item(),
        'tlut': cb.tlut.data.to(orig_dtype).cpu() if hasattr(cb, 'tlut') else None,
        'rcp': 0,
        'tp_rank': 8,
    }
    return saved, dict(hatWr=hatWr, Wr=Wr, Qidxs=Qidxs, LRr=LRr, HRr=HRr, cb=cb)


def unpack_quip(saved_layer):
    """QuantizedLinear buffers after lib/utils/data_utils.py::unpack_quip (rcp == 0)."""
    trellis = saved_layer['trellis']
    tlut = saved_layer['tlut'].float().to(torch.float16)
    SU = saved_layer['SU'].float()
    SV = saved_layer['SV'].float()
    Wscale = saved_layer['Wscale'].float()
    # QuantizedLinear buffers: SU in the model dtype, SV in float32 (Wscale folded in).
    return dict(trellis=trellis, tlut=tlut, SU=SU.to(saved_layer['SU'].dtype),
                SV=(SV * Wscale.to(SV.device)).to(torch.float32))
