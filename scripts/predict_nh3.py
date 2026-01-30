"""
Inference script to use the best trained NH3 head model.
"""
import torch
import numpy as np

def load_model(path="raw_nh3_core/nh3_head_best_final.pt"):
    """Load the trained model"""
    ckpt = torch.load(path, weights_only=False)
    return ckpt

def predict_nh3_uptake(latent_vectors, ckpt=None, model_path="raw_nh3_core/nh3_head_best_final.pt"):
    """
    Predict NH3 uptake from latent vectors.
    
    Args:
        latent_vectors: numpy array or torch tensor of shape (N, 256) or (256,)
        ckpt: preloaded checkpoint (optional)
        model_path: path to model checkpoint
    
    Returns:
        predictions: numpy array of NH3 uptake in mmol/g
    """
    if ckpt is None:
        ckpt = torch.load(model_path, weights_only=False)
    
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.cpu().numpy()
    
    # Handle single sample
    if latent_vectors.ndim == 1:
        latent_vectors = latent_vectors.reshape(1, -1)
    
    # Standardize features
    scaler = ckpt['scaler_X']
    X_scaled = scaler.transform(latent_vectors)
    
    # Feature selection if available
    selector = ckpt.get('selector', None)
    if selector is not None:
        X_scaled = selector.transform(X_scaled)
    
    # Predict
    model_type = ckpt.get('model_type', 'single')
    
    if model_type == 'voting':
        # Ensemble voting
        models = ckpt['models']
        preds = np.stack([m.predict(X_scaled) for m in models])
        predictions = preds.mean(axis=0)
    else:
        # Single model
        model = ckpt['model']
        predictions = model.predict(X_scaled)
    
    return predictions

if __name__ == "__main__":
    # Test with the training data
    data = torch.load("raw_nh3_core/nh3_latent_dataset.pt")
    Z = data["z"].float().numpy()
    Y = data["y"].float().numpy().ravel()
    m_ids = data["m_id"]
    
    # Load model
    ckpt = load_model()
    print(f"Model type: {ckpt.get('model_type', 'unknown')}")
    print(f"CV MAE: {ckpt.get('cv_mae', 'N/A'):.4f}")
    
    # Predict on all data
    preds = predict_nh3_uptake(Z, ckpt)
    
    # Calculate metrics
    mae = np.mean(np.abs(preds - Y))
    r2 = 1 - np.sum((preds - Y)**2) / np.sum((Y - Y.mean())**2)
    
    print(f"\nFull dataset metrics:")
    print(f"  MAE: {mae:.4f} mmol/g")
    print(f"  R2: {r2:.4f}")
    
    # Show some examples
    print(f"\nSample predictions:")
    print(f"{'MOF ID':<15} {'True':>10} {'Pred':>10} {'Error':>10}")
    print("-" * 50)
    for i in range(min(10, len(m_ids))):
        err = preds[i] - Y[i]
        print(f"{m_ids[i]:<15} {Y[i]:>10.3f} {preds[i]:>10.3f} {err:>10.3f}")
    
    # Distribution of predictions
    print(f"\nPrediction distribution:")
    print(f"  Mean: {preds.mean():.4f}")
    print(f"  Std: {preds.std():.4f}")
    print(f"  Min: {preds.min():.4f}")
    print(f"  Max: {preds.max():.4f}")
    
    # Distribution of errors
    errors = preds - Y
    print(f"\nError distribution:")
    print(f"  Mean: {errors.mean():.4f}")
    print(f"  Std: {errors.std():.4f}")
    print(f"  Min: {errors.min():.4f}")
    print(f"  Max: {errors.max():.4f}")
