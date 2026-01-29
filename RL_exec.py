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


def iter_chunk_batches(out_root: str):
    out_root = Path(out_root)
    meta = json.loads((out_root / "meta.json").read_text())
    for week in meta.get("weeks", []):
        week_dir = out_root / week
        week_meta_path = week_dir / "meta_week.json"
        if not week_meta_path.exists():
            continue
        week_meta = json.loads(week_meta_path.read_text())
        for entry in week_meta.get("chunks", []):
            files = entry.get("files", {})
            x_core = np.load(week_dir / files["core"])
            x_aux = np.load(week_dir / files["aux"])
            y = np.load(week_dir / files["y"])
            ts = np.load(week_dir / files["ts"])
            yield week, int(entry.get("chunk", 0)), ts, x_core, x_aux, y
