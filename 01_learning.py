import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.zeros(4, 4)
c = torch.ones(4, 4)
d = torch.rand(4, 4)
e = torch.randn(4, 4)


print(a.shape)
print(a.dtype)
print(a.device)
print(a)
print(b)
print(c)
print(d)
print(e)

if torch.cuda.is_available():
    a = a.cuda()