import json, torch
import numpy as np
from pathlib import Path
from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK

def load_cmssl(out_root: str, ckpt_path: str, device="cuda"):
    out_root = Path(out_root)
    meta = json.loads((out_root / "meta.json").read_text())
    feat_dim = int(meta["feature_dim_total"])  # includes AUX_DIM already

    args = ModelArgs(DMODEL, MAMBA_LAYERS, feat_dim, LOOKBACK)
    model = SAMBA(args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta

@torch.no_grad()
def cmssl_predict(model, x_core, x_aux, device="cuda"):
    # x_core: [B, L, F_core]  x_aux: [B, L, AUX_DIM]
    x_core = torch.as_tensor(x_core, device=device)
    x_aux  = torch.as_tensor(x_aux, device=device)
    out = model(x_core, x_aux)
    # adapt this unpacking to your SAMBA forward output order
    return out
