import os
import torch
import torchao.quantization.quant_primitives as qp
import torchao.prototype.smoothquant.core as sq_core
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


def insert_smooth_quant_observer_filtered(model, filter_fn, alpha=0.5):
    quant_min, quant_max = -127, 127
    eps = torch.finfo(torch.float32).eps

    def replace_with_observer(layer):
        observer = sq_core.SmoothQuantObserver(
            layer.weight, alpha, "dynamic",
            quant_min=quant_min, quant_max=quant_max, eps=eps,
        )
        return sq_core.SmoothQuantObservedLinear.from_float(layer, observer)

    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, filter_fn)


def apply_observer_reshape_patch():
    def _forward(self, input):
        self.act_ic_obs(input.reshape(-1, input.shape[-1]).to("cpu"))
        return input
    sq_core.SmoothQuantObserver.forward = _forward


_SQ_CHUNK_MIN_BYTES = int(os.environ.get("SQ_CHUNK_MIN_BYTES", 256 << 20))
_SQ_CHUNK_BYTES = 64 << 20  # ~64 MiB fp32 working set per chunk


def apply_chunked_quantize_patch():
    _orig = qp._quantize_affine_no_dtype_cast

    def _patched(input, block_size, scale, zero_point, quant_min, quant_max,
                 quant_dtype, zero_point_domain=qp.ZeroPointDomain.INT.name):
        per_token = (list(block_size[:-1]) == [1] * (input.dim() - 1)
                     and block_size[-1] == input.shape[-1])
        if (not input.is_cuda or not input.is_contiguous() or not per_token
                or zero_point_domain != qp.ZeroPointDomain.INT.name
                or quant_dtype != torch.int8
                or input.numel() * input.element_size() < _SQ_CHUNK_MIN_BYTES):
            return _orig(input, block_size, scale, zero_point,
                         quant_min, quant_max, quant_dtype, zero_point_domain)

        x = input.reshape(-1, input.shape[-1])                    # [T, C]
        s = scale.reshape(-1, 1)
        zp = (zero_point.reshape(-1, 1)
              if zero_point is not None and zero_point.numel() > 0 else None)
        out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
        rows = max(1, _SQ_CHUNK_BYTES // (x.shape[1] * x.element_size()))
        for i in range(0, x.shape[0], rows):
            sl = slice(i, i + rows)
            c = x[sl] * (1.0 / s[sl])       # the only full-precision temporary
            c.round_()
            if zp is not None:
                c.add_(zp[sl])
            c.clamp_(quant_min, quant_max)
            out[sl] = c                     # int8 cast on copy
        return out.view(input.shape)

    qp._quantize_affine_no_dtype_cast = _patched
