import torch

from pretrain_src.pretrain_src.optim.adamw import AdamW as CustomAdamW


def main() -> None:
    torch.manual_seed(0)
    w1 = torch.randn(768, 512, 10, 10)
    b1 = torch.randn(768)
    w2, b2 = w1.clone(), b1.clone()
    p1 = [torch.nn.Parameter(w1), torch.nn.Parameter(b1)]
    p2 = [torch.nn.Parameter(w2), torch.nn.Parameter(b2)]
    o1 = CustomAdamW(p1, lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    o2 = torch.optim.AdamW(p2, lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    for _ in range(1000):
        for a, b in zip(p1, p2):
            g = torch.randn_like(a)
            a.grad = g.clone()
            b.grad = g.clone()
        o1.step()
        o2.step()
        o1.zero_grad()
        o2.zero_grad()
    print("max_abs_diff_weight", (p1[0] - p2[0]).abs().max().item())
    print("max_abs_diff_bias", (p1[1] - p2[1]).abs().max().item())


if __name__ == "__main__":
    main()
