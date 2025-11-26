# train_fno_inviscid.py
import torch


def burgers_residual_fd(u, T, Lx=1.0):
    """
    u: (B, 1, Nt, Nx) predicted solution
    T: final time
    Lx: spatial domain length (default 1.0 for [0,1])

    Returns:
      residual tensor (B, Nt-2, Nx-2) for interior points
    """
    B, C, Nt, Nx = u.shape
    assert C == 1, "Expect scalar field u"

    dt = T / (Nt - 1)
    dx = Lx / (Nx - 1)

    # strip channel
    u = u[:, 0, :, :]   # (B, Nt, Nx)

    # time derivative (central difference in t)
    u_t = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dt)   # (B, Nt-2, Nx-2)

    # space derivative (central difference in x)
    u_x = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)   # (B, Nt-2, Nx-2)

    # align u with interior
    u_mid = u[:, 1:-1, 1:-1]                                # (B, Nt-2, Nx-2)

    residual = u_t + u_mid * u_x
    return residual


import torch.nn.functional as F
#from neuralop import BurgersEqnLoss, ICLoss
from neuralop import ICLoss
from sampler import sample_grf, build_fno_input
from fno_model import make_fno
import yaml
import argparse

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

from neuralop.losses.equation_losses import ICLoss

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    Nx = config["Nx"]
    Nt = config["Nt"]
    T  = config["T"]
    batch = config["batch_size"]
    iters = config["iterations"]

    # model
    model = make_fno(
        hidden_width=config["width"],
        n_layers=config["layers"],
        n_modes_t=config["n_modes_t"],
        n_modes_x=config["n_modes_x"]
    ).to(device)

    # losses: PDE + IC
    def burgers_pde_loss(u_pred, T, Lx=1.0):
        r = burgers_residual_fd(u_pred, T, Lx)
        return (r ** 2).mean()

    '''equation_loss = BurgersEqnLoss(
        visc=0.0,
        method="fdm",
        loss=F.mse_loss
    )'''
    

    ic_loss = ICLoss()

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for step in range(1, iters + 1):
        # 1) sample ICs v(x)
        v = sample_grf(batch, Nx, config["ic_length_scale"], device)
        print(f"[step {step}] sampled v", flush=True)
    
        # 2) build FNO input
        inp = build_fno_input(v, Nt, Nx, T, device)
        print(f"[step {step}] built inp, shape={inp.shape}", flush=True)
    
        # 3) forward
        u_pred = model(inp)   # (B,1,Nt,Nx)
        print(f"[step {step}] forward done, u_pred.shape={u_pred.shape}", flush=True)
    
        # 4) IC target: enforce u(t=0,x) ≈ v(x)
        u_ic = torch.zeros_like(u_pred)
        u_ic[:, 0, 0, :] = v
        print(f"[step {step}] IC target built", flush=True)
    
        # NEW: PDE residual (our own)
        L_pde = burgers_pde_loss(u_pred, T)
        
        # 5) IC-only loss
        L_ic = ic_loss(u_pred, u_ic)
        loss  = L_pde + config["w_ic"] * L_ic
        print(f"[step {step}] L_ic={L_ic.item():.4e}", flush=True)
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[step {step}] optimizer step done", flush=True)
    
        if step % 100 == 0:
            print(f"{step}/{iters}: loss={loss.item():.4e}  L_pde={L_pde.item():.4e}  L_ic={L_ic.item():.4e}")    
        
        # stop early for debug if you want
        # if step == 20:
        #     break


    torch.save(model.state_dict(), "fno_inviscid_burgers.pt")
    print("Model saved → fno_inviscid_burgers.pt")


    # quick eval on a fixed IC
    model.eval()
    with torch.no_grad():
        v_eval = sample_grf(1, Nx, config["ic_length_scale"], device)
        inp_eval = build_fno_input(v_eval, Nt, Nx, T, device)
        u_eval = model(inp_eval)  # (1,1,Nt,Nx)
    
        torch.save(
            {
                "v": v_eval.cpu(),
                "u": u_eval.cpu(),
            },
            "sample_solution.pt",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)