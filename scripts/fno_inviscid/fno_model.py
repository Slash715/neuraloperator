# fno_model.py
from neuralop.models import FNO

def make_fno(hidden_width=256, n_layers=4, n_modes_t=8, n_modes_x=32):
    return FNO(
        n_modes=(n_modes_t, n_modes_x),
        hidden_channels=hidden_width,
        in_channels=3,      # [v(x), t, x]
        out_channels=1
    )
