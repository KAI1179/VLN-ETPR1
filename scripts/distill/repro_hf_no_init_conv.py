"""Does transformers 4.43.4 leave nn.Conv2d / nn.MultiheadAttention uninitialised when a
model is built via from_pretrained(None, config=..., state_dict=...) and those modules are
absent from the state_dict?  Mirrors pretrain_src/train_r2r.py's construction path."""
import torch, torch.nn as nn, transformers
from transformers import BertConfig, PreTrainedModel

class M(PreTrainedModel):
    config_class = BertConfig
    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Conv2d(512, 768, kernel_size=10, stride=10)   # like spatial_tokenizer
        self.lin = nn.Linear(14, 768)                                 # like metadata_encoder[0]
        self.mha = nn.MultiheadAttention(768, 12, batch_first=True)   # like fusion / token_transformer
        self.post_init()
    def _init_weights(self, module):                                  # BERT's rule set
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_(); module.weight.data.fill_(1.0)

def stats(name, t):
    t = t.detach().double()
    print(f"  {name:28s} norm {t.norm():12.4e}  max|.| {t.abs().max():12.4e}  zero-frac {(t == 0).double().mean():.3f}")

cfg = BertConfig()
print("transformers", transformers.__version__, "| torch", torch.__version__)
print("A) M(config) -- plain construction (kaiming / xavier expected):")
a = M(cfg)
stats("conv.weight", a.conv.weight); stats("conv.bias", a.conv.bias); stats("lin.weight", a.lin.weight); stats("mha.in_proj_weight", a.mha.in_proj_weight)
print("B) M.from_pretrained(None, config, state_dict={}) -- train_r2r.py path, modules missing from ckpt:")
b = M.from_pretrained(None, config=cfg, state_dict={})
stats("conv.weight", b.conv.weight); stats("conv.bias", b.conv.bias); stats("lin.weight", b.lin.weight); stats("mha.in_proj_weight", b.mha.in_proj_weight)
print("C) from_pretrained with the mha weights PRESENT in state_dict (conv/lin missing):")
sd = {k: v for k, v in a.state_dict().items() if k.startswith("mha.")}
c = M.from_pretrained(None, config=cfg, state_dict=sd)
stats("conv.weight", c.conv.weight); stats("conv.bias", c.conv.bias); stats("lin.weight", c.lin.weight); stats("mha.in_proj_weight", c.mha.in_proj_weight)
import inspect
src = inspect.getsource(transformers.modeling_utils.no_init_weights)
print("no_init_weights patches torch.nn.init:", "TORCH_INIT_FUNCTIONS" in src or "_skip_init" in src)
