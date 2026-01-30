"""
sample coarse-grained MOFs using a trained CG diffusion model.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import numpy as np
import gc
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree

from mofdiff.common.atomic_utils import arrange_decoded_mofs
from mofdiff.common.eval_utils import load_mofdiff_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="PATH/mofdiff_ckpt",
    )
    parser.add_argument(
        "--bb_cache_path",
        type=str,
        default="PATH/bb_emb_space.pt",
    )

    parser.add_argument("--n_samples", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str, help="Device: cuda or cpu")
    parser.add_argument("--low_memory", action="store_true", help="Use low memory mode (slower but uses less RAM)")

    # get datamodule prop_list
    args = parser.parse_args()
    seed_everything(args.seed)
    model_path = Path(args.model_path)
    
    # Resolve to absolute path for Hydra
    model_path = model_path.resolve()
    
    print(f"Loading model from {model_path}...")
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    print(f"Loading building block cache from {args.bb_cache_path}...")
    if args.low_memory:
        # Memory-mapped loading for large files
        all_data, all_z = torch.load(args.bb_cache_path, map_location='cpu')
        all_z = all_z.numpy()  # Convert to numpy for KDTree
    else:
        all_data, all_z = torch.load(args.bb_cache_path)
    
    print("Building KDTree...")
    kdtree = KDTree(all_z)
    
    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    output = defaultdict(list)
    n_batch = int(np.ceil(args.n_samples / args.batch_size))
    print(f"Generating {args.n_samples} samples in {n_batch} batches...")
    
    all_latent_z = []  # Store the latent z vectors used for sampling
    
    for idx in range(n_batch):
        print(f"Processing batch {idx + 1}/{n_batch}...")
        current_batch_size = min(args.batch_size, args.n_samples - idx * args.batch_size)
        z = torch.randn(current_batch_size, model.hparams.latent_dim).to(device)
        samples = model.sample(z.shape[0], z, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        output["mofs"].extend(mofs)
        all_latent_z.append(z.cpu())
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    output["z"] = torch.cat(all_latent_z, dim=0)
    savedir = f"samples_{args.n_samples}_seed_{args.seed}"
    (model_path / savedir).mkdir(exist_ok=True)
    save_path = model_path / savedir / "samples.pt"
    print(f"Saving samples to {save_path}...")
    torch.save(output, save_path)
    print("Done!")
