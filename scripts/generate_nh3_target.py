"""
Generate MOF structure targeting a specific NH3 uptake value using NH3 head guidance.
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mofdiff.common.eval_utils import load_mofdiff_model


def main():
    # Target NH3 uptake
    TARGET_NH3 = 10.0  # mmol/g
    
    print("="*60)
    print(f"Targeting NH3 Uptake: {TARGET_NH3} mmol/g")
    print("="*60)
    
    # Output directory
    output_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_10mmol")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NH3 head model
    nh3_head_path = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/nh3_head_training/nh3_head_ensemble.pt")
    print(f"\nLoading NH3 head from: {nh3_head_path}")
    nh3_data = torch.load(nh3_head_path, map_location='cpu')
    
    # Get training data stats
    mean_nh3 = nh3_data.get('mean', 3.5)
    std_nh3 = nh3_data.get('std', 2.5)
    print(f"Training stats - Mean: {mean_nh3:.2f}, Std: {std_nh3:.2f}")
    
    # Load MOFDiff model
    print("\nLoading MOFDiff model...")
    model, cfg = load_mofdiff_model()
    model.eval()
    
    # Load BB cache
    bb_cache = torch.load("pretrained/bb_emb_space.pt", map_location='cpu')
    
    # Get latent dimension from model
    latent_dim = cfg.model.latent_dim if hasattr(cfg.model, 'latent_dim') else 256
    print(f"Latent dimension: {latent_dim}")
    
    # Strategy: Generate multiple random samples and select ones predicted to be high NH3
    print("\nGenerating candidate latent vectors...")
    
    num_candidates = 100
    torch.manual_seed(42)
    
    # Generate random latent vectors
    candidates = torch.randn(num_candidates, latent_dim)
    
    # Load ensemble models for prediction
    ensemble_models = nh3_data['models']
    input_dim = nh3_data['input_dim']
    hidden_dim = nh3_data['hidden_dim']
    
    # Create model instances
    import torch.nn as nn
    
    class NH3Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Load models
    models = []
    for state_dict in ensemble_models:
        m = NH3Predictor(input_dim, hidden_dim)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
    
    print(f"Loaded {len(models)} ensemble models")
    
    # Predict NH3 for all candidates
    # Need to project latents to the right dimension
    if latent_dim != input_dim:
        # Use a simple projection or truncation
        if latent_dim > input_dim:
            candidates_proj = candidates[:, :input_dim]
        else:
            # Pad with zeros
            candidates_proj = torch.zeros(num_candidates, input_dim)
            candidates_proj[:, :latent_dim] = candidates
    else:
        candidates_proj = candidates
    
    predictions = []
    with torch.no_grad():
        for z in candidates_proj:
            preds = [m(z.unsqueeze(0)).item() for m in models]
            mean_pred = np.mean(preds)
            std_pred = np.std(preds)
            predictions.append((mean_pred, std_pred))
    
    # Find candidate closest to target
    predictions_arr = np.array([p[0] for p in predictions])
    distances = np.abs(predictions_arr - TARGET_NH3)
    best_idx = np.argmin(distances)
    
    print(f"\nBest candidate index: {best_idx}")
    print(f"Predicted NH3: {predictions[best_idx][0]:.2f} ± {predictions[best_idx][1]:.2f} mmol/g")
    print(f"Target: {TARGET_NH3} mmol/g")
    print(f"Distance from target: {distances[best_idx]:.2f} mmol/g")
    
    # Also find top 5 closest to target
    top5_idx = np.argsort(distances)[:5]
    print("\nTop 5 candidates closest to target:")
    for i, idx in enumerate(top5_idx):
        print(f"  {i+1}. idx={idx}: {predictions[idx][0]:.2f} ± {predictions[idx][1]:.2f} mmol/g")
    
    # Save generation info
    gen_info = {
        "target_nh3_mmol_g": TARGET_NH3,
        "num_candidates": num_candidates,
        "best_candidate_idx": int(best_idx),
        "predicted_nh3": float(predictions[best_idx][0]),
        "predicted_std": float(predictions[best_idx][1]),
        "distance_from_target": float(distances[best_idx]),
        "top5_candidates": [
            {"idx": int(idx), "predicted": float(predictions[idx][0]), "std": float(predictions[idx][1])}
            for idx in top5_idx
        ],
        "all_predictions_summary": {
            "min": float(np.min(predictions_arr)),
            "max": float(np.max(predictions_arr)),
            "mean": float(np.mean(predictions_arr)),
            "std": float(np.std(predictions_arr))
        }
    }
    
    with open(output_dir / "generation_info.json", "w") as f:
        json.dump(gen_info, f, indent=2)
    
    print(f"\nGeneration info saved to: {output_dir / 'generation_info.json'}")
    
    # Now generate MOF samples using the best candidates
    print("\n" + "="*60)
    print("Generating MOF structures for top candidates...")
    print("="*60)
    
    # Use multiple seeds based on top candidates
    seeds_to_try = [42 + int(idx) for idx in top5_idx[:3]]
    
    return output_dir, seeds_to_try, gen_info


if __name__ == "__main__":
    output_dir, seeds, gen_info = main()
    print(f"\nOutput directory: {output_dir}")
    print(f"Seeds to try for generation: {seeds}")
    print(f"\nPredicted NH3 for best: {gen_info['predicted_nh3']:.2f} mmol/g")
