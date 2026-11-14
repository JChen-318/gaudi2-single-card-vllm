# Qwen3.8-27B DSpine qualification card

- Draft: `tencent/Qwen3.8-27B-DSpine-b16`, revision
  `5f54174d301dbca65d9aa4448dc47e56732d9a12`.
- Target: Qwen3.8-27B, the repository's existing FP8 checkpoint layout.
- Hardware: one Gaudi2; draft BF16; FP8 target/KV; TP=PP=DP=1.
- Verification: 8 or 16 tokens per request; 7 or 15 speculative tokens.
- Initial configuration: `config/dspine.env.example`; text only, no prefix
  caching, context 16384, two concurrent sequences.
- Run `tools/validate_dspine_api.py` for streaming, repeated greedy requests,
  sampled Chinese output and concurrent request isolation.
- Run `tools/validate_dspine_greedy.py` to compare generated tokens against
  independent single-token target-prefill requests. Those requests finish
  before speculative verification, providing a target-only reference.
- Run `tools/check_dspine_gdn.py` with the launcher's native-kernel environment
  for real-device checkpoint comparison, including accepted positions beyond 8.

No GSM8K or other aggregate accuracy threshold has been established by this
card. Record numerical comparisons, outputs and serving metrics separately;
the generic language-evaluation YAML harness does not accept draft settings.

Device checks cover BF16 bridge/refinement, vocabulary-sized boundary ties,
and eager/compiled T16 GDN rollback through both halves of the checkpoint pool.
End-to-end greedy repeat variation remains unresolved. The original host
lookahead/predraft reuse caused HPU runtime concat failures with this draft
and is disabled for DSpine. Use the API check's explicit
`--allow-repeat-variation` option for functional serving checks; it records
variation without certifying target-only equivalence or output accuracy.

Use `tools/benchmark_dspine_decode.py` for warm single-request throughput.
An acceptance-length improvement or a faster candidate-selection kernel does
not establish an end-to-end speedup over the existing DFlash2 configuration.
