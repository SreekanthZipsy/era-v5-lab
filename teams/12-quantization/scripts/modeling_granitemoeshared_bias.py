"""GraniteMoeShared plus the two biases that G5's KAS V2 model has and the stock class cannot carry (G12, Nov 2026).

 - lm_head.bias [vocab]: the KAS head's log prior (head.unigram), added to the logits after the logit scale.
 - model.layers.N.block_sparse_moe.router.selection.bias [experts]: G5's router_bias. It is added to the router
   probabilities only to choose the experts; the gates stay the unbiased probabilities of the chosen experts,
   renormalised (G5 week2/model.py, MoE.forward). The router is the stock one otherwise.

Everything else is transformers' GraniteMoeShared. Load with `trust_remote_code=True`; export_hf.py --keep-biases
writes this file next to the weights and sets config.json's model_type and auto_map to it, so that a stock
transformers (which does not know model_type "granitemoeshared_bias") stops with an error instead of loading the
model without its biases. Only the two flags in the config say which biases the folder holds.
"""
import torch
from torch import nn
from transformers import GraniteMoeSharedConfig, GraniteMoeSharedForCausalLM


class GraniteMoeSharedBiasConfig(GraniteMoeSharedConfig):
    model_type = 'granitemoeshared_bias'

    def __init__(self, has_lm_head_bias=False, has_router_selection_bias=False, **kw):
        super().__init__(**kw)
        self.has_lm_head_bias = has_lm_head_bias
        self.has_router_selection_bias = has_router_selection_bias


class _SelectionBias(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.register_buffer('bias', torch.zeros(n_experts))   # persistent: it is in the state_dict as `.bias`


class _BiasedGating(nn.Module):
    """Same inputs and outputs as transformers' GraniteMoeSharedTopKGating.forward, with G5's choice of experts."""
    def __init__(self, stock):
        super().__init__()
        self.num_experts, self.input_size, self.top_k, self.layer = stock.num_experts, stock.input_size, stock.top_k, stock.layer
        self.selection = _SelectionBias(stock.num_experts)

    def forward(self, hidden_states):
        logits = self.layer(hidden_states).float()
        probs = logits.softmax(-1)
        top_k_indices = (probs + self.selection.bias.float()).topk(self.top_k, dim=-1).indices   # choice
        gates = probs.gather(-1, top_k_indices)
        gates = (gates / gates.sum(-1, keepdim=True)).type_as(hidden_states)                     # the unbiased probs, renormalised
        size = torch.zeros(gates.size(0), self.num_experts, dtype=torch.long, device=gates.device).scatter(1, top_k_indices, 1).sum(0).tolist()
        _, order = top_k_indices.flatten().sort(0)
        return order, order.div(self.top_k, rounding_mode='trunc'), gates.flatten()[order], size, logits


class GraniteMoeSharedBiasForCausalLM(GraniteMoeSharedForCausalLM):
    config_class = GraniteMoeSharedBiasConfig

    def __init__(self, config):
        super().__init__(config)
        if config.logits_scaling != 1.0:   # the bias is added after the scale in llama.cpp, inside lm_head here: equal only at 1
            raise ValueError('granitemoeshared_bias needs logits_scaling 1.0')
        if config.has_lm_head_bias:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        if config.has_router_selection_bias:
            for layer in self.model.layers:
                layer.block_sparse_moe.router = _BiasedGating(layer.block_sparse_moe.router)
