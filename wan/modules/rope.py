import torch


def rope_params_riflex(max_seq_len, dim, theta=10000, k=0, L_test=None):
    assert dim % 2 == 0
    omega = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    if k is not None:
        print("Doing riflex w/ ltest", L_test)
        omega[k - 1] = 0.9 * 2 * torch.pi / L_test
    freqs = torch.outer(torch.arange(max_seq_len), omega)

    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs
