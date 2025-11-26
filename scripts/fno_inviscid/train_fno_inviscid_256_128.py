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
from neuralop import BurgersEqnLoss, ICLoss
#from neuralop import ICLoss
from sampler import sample_grf, build_fno_input
from fno_model import make_fno
import yaml
import argparse

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

from neuralop.losses.equation_losses import ICLoss
from neuralop.models import FNO

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    Nx = config["Nx"]
    Nt = config["Nt"]
    T  = float(config["T"])
    
    width      = config["width"]
    layers     = config["layers"]
    n_modes_x  = config["n_modes_x"]
    n_modes_t  = config["n_modes_t"]
    in_ch      = config.get("in_channels", 3)
    out_ch     = config.get("out_channels", 1)
    
    batch_size = config["batch_size"]
    iterations = config["iterations"]
    lr         = config["lr"]
    
    w_ic  = config["w_ic"]
    w_pde = config.get("w_pde", 1.0)
    
    weight_decay    = config.get("weight_decay", 0.0)
    scheduler_step  = config.get("scheduler_step", None)
    scheduler_gamma = config.get("scheduler_gamma", 0.5)
    print_every = config.get("print_every", 100)
    val_every   = config.get("val_every", 1000)

    model = FNO(
        n_modes=(n_modes_t, n_modes_x),
        hidden_channels=width,
        in_channels=in_ch,
        out_channels=out_ch,
        n_layers=layers,
    ).to(device)


    # model
    '''model = make_fno(
        hidden_width=config["width"],
        n_layers=config["layers"],
        n_modes_t=config["n_modes_t"],
        n_modes_x=config["n_modes_x"]
    ).to(device)'''

    # losses: PDE + IC
    '''def burgers_pde_loss(u_pred, T, Lx=1.0):
        r = burgers_residual_fd(u_pred, T, Lx)
        return (r ** 2).mean()'''

    equation_loss = BurgersEqnLoss(
        visc=0.0,
        method="fdm",
        loss=F.mse_loss,
        domain_length=(T, 1.0), 
    )
    

    ic_loss = ICLoss()

    # optimizer
    #opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    if scheduler_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=scheduler_step, gamma=scheduler_gamma
        )
    else:
        scheduler = None

    # Prepare fixed validation batch
    with torch.no_grad():
        v_val = sample_grf(batch_size, Nx, config["ic_length_scale"], device)
        inp_val = build_fno_input(v_val, Nt, Nx, T, device)
        # build IC target for validation
        u_ic_val = torch.zeros((v_val.shape[0], 1, Nt, Nx), device=device)
        u_ic_val[:, 0, 0, :] = v_val
    
    train_losses = []
    test_losses  = []

    for step in range(1, iterations + 1):
        # 1) sample ICs v(x)
        v = sample_grf(batch_size, Nx, config["ic_length_scale"], device)
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
        #L_pde = burgers_pde_loss(u_pred, T)
        
        L_pde = equation_loss(u_pred)          # library PDE residual
        # 5) IC-only loss
        L_ic = ic_loss(u_pred, u_ic)
        #loss = L_pde + config["w_ic"] * L_ic
        loss  = w_pde * L_pde + w_ic * L_ic
        print(f"[step {step}] L_pde={L_pde.item():.4e}  L_ic={L_ic.item():.4e}  loss={loss.item():.4e}", flush=True,)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[step {step}] optimizer step done", flush=True)

        if scheduler:
            scheduler.step()

        train_losses.append(loss.item())
        

        if step % print_every == 0:
            print(f"{step}/{iterations}: loss={loss.item():.4e}  L_pde={L_pde.item():.4e}  L_ic={L_ic.item():.4e}", flush=True)
    
        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                u_val = model(inp_val)
                L_pde_val = equation_loss(u_val)
                L_ic_val  = ic_loss(u_val, u_ic_val)
                loss_val  = w_pde * L_pde_val + w_ic * L_ic_val
            test_losses.append(loss_val.item())
            print(f"[VAL {step}] loss_val={loss_val.item():.4e}  L_pde_val={L_pde_val.item():.4e}  L_ic_val={L_ic_val.item():.4e}", flush=True)
            model.train()


        
        # stop early for debug if you want
        # if step == 20:
        #     break


    torch.save(model.state_dict(), "PDE-burgers-256-128-FNO.pt")
    print("Model saved → PDE-burgers-256-128-FNO.pt")




    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8,6))
    plt.semilogy(train_losses, label="Train loss")
    

    # Align test curve to same x-axis
    pad = [None]*(len(train_losses)-len(test_losses))
    plt.semilogy(pad + test_losses, label="Test loss")
    plt.xlabel("# Steps")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss-PDE-burgers-256-128-FNO.png", dpi=300)
    plt.close()
    print("Loss curves saved → loss-PDE-burgers-256-128-FNO.png")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_256_128.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)