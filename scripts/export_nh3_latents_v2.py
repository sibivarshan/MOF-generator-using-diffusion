"""
Export NH3 latent representations from pretrained MOFDiff model.
Uses MOFDataset to properly preprocess data.
"""
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

from mofdiff.common.eval_utils import load_mofdiff_model
from mofdiff.data.dataset import MOFDataset

PROP_KEY = "NH3_uptake_298K_1bar [mmol/g]"
LMDB_PATH = "raw_nh3_core/lmdbs/data_filtered.lmdb"  # Use filtered LMDB
MODEL_PATH = "/home/sibivarshan_m7/MOFDiff/MOFDiff/pretrained/mofdiff_ckpt"
OUT_PATH = "raw_nh3_core/nh3_latent_dataset.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading model...")
    model_path = Path(MODEL_PATH)
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    model = model.to(DEVICE).eval()

    # Get data config from pretrained model
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    
    # Check if NH3 property is in prop_list
    prop_list = data_cfg.get("prop_list", [])
    print(f"Model's prop_list: {prop_list}")
    
    if PROP_KEY not in prop_list:
        print(f"Warning: {PROP_KEY} not in model's prop_list.")
        print("This is expected if using pretrained model on new data.")
        # Add NH3 property to allow dataset to load
        prop_list = [PROP_KEY]
        data_cfg["prop_list"] = prop_list
    
    # Update config for NH3 data
    data_cfg.update({
        "name": "nh3_core",
        "path": LMDB_PATH,
        "bb_encoder": bb_encoder,
        "keep_bbs": False,  # Don't need BBs after encoding
    })
    
    print("Loading dataset...")
    try:
        dataset = MOFDataset(**data_cfg)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No valid samples in dataset!")
        return
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    z_chunks = []
    y_list = []
    id_list = []
    
    print("Encoding latents...")
    for batch in tqdm(loader, desc="Encoding"):
        batch = batch.to(DEVICE)
        
        with torch.no_grad():
            try:
                _, _, z = model.encode(batch)
                z_chunks.append(z.cpu())
                y_list.append(batch.y.cpu())
                id_list.extend(getattr(batch, 'm_id', [None] * batch.num_graphs) if hasattr(batch, 'm_id') else [None] * batch.num_graphs)
            except Exception as e:
                print(f"Error encoding batch: {e}")
                print(f"  batch.num_atoms: {batch.num_atoms}")
                print(f"  batch.atom_types shape: {batch.atom_types.shape}")
                raise
    
    if not z_chunks:
        print("No latents encoded!")
        return
    
    Z = torch.cat(z_chunks, dim=0)
    Y = torch.cat(y_list, dim=0)
    
    # Get the NH3 property column
    if Y.dim() > 1 and Y.shape[1] > 1:
        # Find the index of NH3 property
        prop_idx = prop_list.index(PROP_KEY) if PROP_KEY in prop_list else 0
        Y = Y[:, prop_idx:prop_idx+1]
    
    torch.save({"z": Z, "y": Y, "m_id": id_list}, OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"Z shape: {Z.shape}, Y shape: {Y.shape}")


if __name__ == "__main__":
    main()
