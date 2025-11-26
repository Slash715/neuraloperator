# sampler.py
import torch, math

def sample_grf(batch, Nx, length_scale, device):
    x = torch.linspace(0,1,Nx,device=device)
    X1 = x.view(Nx,1)
    X2 = x.view(1,Nx)
    K = torch.exp(-2.0 * torch.sin(math.pi*(X1-X2))**2 / length_scale**2)
    K += 1e-5 * torch.eye(Nx,device=device)
    L = torch.linalg.cholesky(K)
    z = torch.randn(Nx, batch, device=device)
    v = (L @ z).T                          # (B,Nx)
    v = (v - v.mean(1,keepdim=True)) / (v.std(1,keepdim=True)+1e-8)
    return v

def build_fno_input(v, Nt, Nx, T, device):
    B = v.shape[0]
    x = torch.linspace(0,1,Nx,device=device)
    t = torch.linspace(0,T,Nt,device=device)

    v_tile = v[:,None,:].repeat(1,Nt,1)
    t_tile = t[None,:,None].repeat(B,1,Nx)
    x_tile = x[None,None,:].repeat(B,Nt,1)

    return torch.stack([v_tile, t_tile, x_tile], dim=1)   # (B,3,Nt,Nx)
