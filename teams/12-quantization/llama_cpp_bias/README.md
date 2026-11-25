# llama.cpp with the two G5 biases (G12, Nov 2026)

`granite-bias.patch` makes llama.cpp load and use the two biases that G5's KAS V2 model has and the stock `granitemoe`
architecture cannot carry. It is written for **llama.cpp b11669, commit 620d2225cc436ab0b001e3200e590e3b286ef9ab**
(tag `v0.7.0` of that tree; `llama-server --version` prints build 11669). The checkout lives outside this repo.

| Change | File | What it does |
|---|---|---|
| optional `output.bias` {n_vocab} | `src/models/granite-moe.cpp` (loader), `src/models/granite.cpp` (graph) | `if (model.output_b) cur = ggml_add(cur, output_b)` after the logit scale. `llama_model.output_b` already exists in b11669; granite never read it. |
| per-layer `blk.N.exp_probs_b.bias` {n_expert} | `src/models/granite-moe.cpp`, `src/models/granite.cpp` | passed as `exp_probs_b` to `build_moe_ffn`, which already does what G5 does: top-k of `softmax + bias`, gates from the unbiased `softmax`, renormalised (`norm_w`). Stock granite passed `nullptr`. Both tensors are optional, so every stock Granite GGUF loads and runs as before. |
| converter | `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`, `conversion/granite.py`, `conversion/__init__.py` | `GraniteMoeSharedBiasForCausalLM` (the architecture `export_hf.py --keep-biases` writes) converts like `GraniteMoeSharedForCausalLM`; `lm_head.bias` becomes `output.bias`, `...router.selection.bias` becomes `blk.N.exp_probs_b.bias` (both stay F32). |

Apply and build (no GPU, no web UI, so nothing is downloaded):

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab
    git apply <repo>/teams/12-quantization/llama_cpp_bias/granite-bias.patch
    cmake -S . -B build -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target llama-server llama-quantize llama-perplexity -j 8

Use with `quantize_eval.py --keep-biases --work <dir holding llama.cpp-b11669 = the patched checkout> --llama-bin <build>/bin`.

What this does **not** do: stock llama.cpp, Ollama and vLLM cannot load these GGUFs. Tested: stock b11669 (homebrew
`llama-server`) stops at load with `done_getting_tensors: wrong number of tensors; expected 116, got 107` (it fails loudly,
it does not run without the biases). Ollama embeds its own llama.cpp and vLLM has its own loader; neither was tried, and
neither has the patch. Only CPU was tested (`-ngl 0`, Metal off); the GPU kernels for `ggml_add` and top-k are the stock ones.
