import torch
torch.set_default_device("cuda")

B = 2048
D = 4096

a = torch.linspace(-1000, 1000, B*D).reshape(B, D)
b = torch.linspace(-1000, 1000, D*D).reshape(D, D)

out1 = torch.mm(a[:1], b)

out2 = torch.mm(a, b)[:1]

print(out1, out2)