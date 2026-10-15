"""Device check of the GDN MTP cores: reference vs prepared (vs fused), numerics and device time.
python3 device_gdn_test.py <label> [--fused]"""
import json
import sys
from unittest import mock

import torch
import habana_frameworks.torch  # noqa: F401

from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_packed_reference, _gated_delta_rule_mtp_prepared

label = sys.argv[1]
with_fused = "--fused" in sys.argv
if with_fused:
    from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_fused


def make_inputs(batch, seed):
    rng = torch.Generator().manual_seed(seed)
    packed = torch.randn(batch, 8, 10240, dtype=torch.bfloat16, generator=rng) * 0.3
    log_decay = -torch.rand(batch, 8, 48, generator=rng) * 0.1
    beta = torch.sigmoid(torch.randn(batch, 8, 48, generator=rng)).to(torch.bfloat16)
    slots = batch * 8 + 2
    pool = torch.randn(slots, 48, 128, 128, generator=rng) * 0.05
    indices = torch.arange(1, batch * 8 + 1, dtype=torch.int32).reshape(batch, 8)
    accepted = torch.randint(1, 9, (batch, ), generator=rng, dtype=torch.int32)
    lengths = torch.full((batch, ), 8, dtype=torch.int32)
    return [x.to("hpu") for x in (packed, log_decay, beta, pool, indices, accepted, lengths)]


def reference(packed, log_decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_packed_reference(packed, log_decay, beta, pool, indices, accepted, lengths,
                                                  assume_full_query=True)[0]


def prepared(packed, log_decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_prepared(packed, log_decay, beta, pool, indices, accepted,
                                          assume_distinct_checkpoints=True)[0]


def fused(packed, log_decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_fused(packed, log_decay, beta, pool, indices, accepted,
                                       assume_distinct_checkpoints=True)[0]


def fused_direct(packed, log_decay, beta, pool, indices, accepted, lengths):
    return _gated_delta_rule_mtp_fused(packed, log_decay, beta, pool, indices, accepted,
                                       assume_distinct_checkpoints=True, checkpoint_start=1)[0]


def measure(function, inputs, iterations=200):
    begin, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function(*inputs)
    end.record()
    torch.hpu.synchronize()
    return begin.elapsed_time(end) / iterations


from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config
from flashinfer_gaudi._native import public_mtp_prepared_op

# Load the native extension before tracing (its loader takes a lock Dynamo cannot trace).
assert public_mtp_prepared_op() is not None, "prepared op missing"
if with_fused:
    from flashinfer_gaudi._native import public_mtp_fused_op
    assert public_mtp_fused_op() is not None, "fused op missing"

report = {"label": label}
names = [("reference", reference), ("prepared", prepared)]
if with_fused:
    names += [("fused", fused), ("fused_direct", fused_direct)]
with mock.patch.object(hpu_config, "use_eager_fallback", False):
    for batch in (1, 2):
        torch._dynamo.reset()
        functions = {n: torch.compile(f, backend="hpu_backend", fullgraph=True, dynamic=False) for n, f in names}
        result = {"checks": []}
        for seed in range(4):
            base = make_inputs(batch, 100 * batch + seed)
            outs, pools = {}, {}
            for n, f in functions.items():
                args = list(base)
                args[3] = base[3].clone()
                outs[n] = f(*args).float().cpu()
                pools[n] = args[3].cpu()
            torch.hpu.synchronize()
            for n in functions:
                if n == "reference":
                    continue
                result["checks"].append({
                    "seed": seed, "core": n, "accepted": base[5].cpu().tolist(),
                    "out_max_abs": (outs[n] - outs["reference"]).abs().max().item(),
                    "pool_max_abs": (pools[n] - pools["reference"]).abs().max().item(),
                    "out_ref_max": outs["reference"].abs().max().item(),
                })
        inputs = make_inputs(batch, 7)
        times = {}
        for n, f in functions.items():
            for _ in range(20):
                f(*inputs)
            torch.hpu.synchronize()
            times[n] = sorted(measure(f, inputs) for _ in range(5))[2]
        result["device_ms"] = times
        report[str(batch)] = result
print(json.dumps(report, indent=1))
