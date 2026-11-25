# G12 week 4: status (Shubham, Wed 25 Nov 2026)

## Merged on main
- #508, #669: one-command flow (`scripts/run_flow.py`), dry run 1 job request, 100 fixed prompts, `eval_own_20`.
- #762, #777: router_bias export (A1, A3); a missing `bias_gamma` counts as 0.
- #780: KAS V2 export (codec and head folded), A3 fixes. Tested on Team 5's synthetic three-step bundle and real-text probe.

## This branch (not on main)
- `llama_cpp_bias/granite-bias.patch` (llama.cpp b11669): optional `output.bias` and per-layer `exp_probs_b`, so the unigram output bias and the router selection bias are kept.
- `export_hf.py --keep-biases` plus `modeling_granitemoeshared_bias.py` (needs `trust_remote_code`). Default export unchanged: it still refuses.
- `--keep-biases` in `quantize_eval.py` and `check_gguf_logits.py`.
- Tests: `test_export_hf` 33, `test_quantize_eval` 5, `test_flow_fixed_prompts` 8, all pass on this branch rebased on main 821c82ad.

## Results (Team 5 synthetic bundle, 33 real-text tokens, parity only, not quality)
| Variant | top-1 vs native | mean NLL diff |
|---|---|---|
| HF custom class, fp32 | 100% | ~1e-6 |
| llama.cpp bf16, biases kept | 33/33 | +0.002 |
| llama.cpp Q8_0, biases kept | 33/33 | -0.003 |
| llama.cpp Q4_K_M, biases kept | 30/33 | +0.020 |
| llama.cpp bf16, both biases dropped (stock) | 0/33 | +4.78 |

A1 with biases kept passes (2.4e-5). A3 passes in bf16 (0.0101 vs 0.05); Q8_0 and Q4_K_M give 0.10 and 0.14, which is quantization noise against a bf16 reference.

## Not covered
- Stock llama.cpp, Ollama and vLLM cannot load a GGUF with these biases. The model is not yet usable the way other downloaded models are.
- No trained KAS checkpoint seen: dry run 2 had not started on 25 Nov. Dry run 1 flow job (#669): no result from TSAI yet.
- No GPU/Metal run, no Ollama or vLLM test, no hub upload, no KLD run.

## Open
- Release requirement: stock runtimes (needs an upstream llama.cpp PR, or a bias that folds into the weights; a mean-hidden-state fold is untested).
- Team 12 line for the Gate 4 record (#705): drafted, not posted.
