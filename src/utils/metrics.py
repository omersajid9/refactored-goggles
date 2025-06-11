import numpy as np

def analyze_model_complexity(model):
    """Analyze model complexity and parameter statistics"""
    total_params = 0
    trainable_params = 0
    
    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        # Parameter statistics
        param_data = param.data.cpu().numpy()
        param_mean = np.mean(param_data)
        param_std = np.std(param_data)
        param_min = np.min(param_data)
        param_max = np.max(param_data)
        
        print(f"{name:25s} | Shape: {str(param.shape):20s} | "
              f"Params: {param_count:8d} | Mean: {param_mean:8.4f} | "
              f"Std: {param_std:7.4f} | Range: [{param_min:7.4f}, {param_max:7.4f}]")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
    print("=" * 60)


def monitor_gradients(model):
    """Monitor gradient statistics during training"""
    total_norm = 0
    param_count = 0
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Detailed gradient statistics
            grad_data = param.grad.data.cpu().numpy()
            gradient_stats[name] = {
                'norm': param_norm.item(),
                'mean': np.mean(grad_data),
                'std': np.std(grad_data),
                'min': np.min(grad_data),
                'max': np.max(grad_data),
                'zero_fraction': np.mean(grad_data == 0)
            }
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, gradient_stats

def log_gradient_stats(gradient_stats, step, log_every=1000):
    """Log detailed gradient statistics"""
    if step % log_every == 0:
        print(f"\n--- Gradient Analysis (Step {step}) ---")
        for name, stats in gradient_stats.items():
            print(f"{name:25s} | Norm: {stats['norm']:8.4f} | "
                  f"Mean: {stats['mean']:8.4f} | Std: {stats['std']:7.4f} | "
                  f"Zero%: {stats['zero_fraction']*100:5.1f}")
        print("-" * 60)


def detect_gradient_anomalies(gradient_stats, total_norm, step):
    """Detect potential gradient problems"""
    warnings = []
    
    # Check for gradient explosion
    if total_norm > 10.0:
        warnings.append(f"⚠️  Large gradient norm: {total_norm:.4f}")
    
    # Check for vanishing gradients
    if total_norm < 1e-6:
        warnings.append(f"⚠️  Very small gradient norm: {total_norm:.6f}")
    
    # Check individual layer gradients
    for name, stats in gradient_stats.items():
        if stats['norm'] > 5.0:
            warnings.append(f"⚠️  Large gradient in {name}: {stats['norm']:.4f}")
        if stats['zero_fraction'] > 0.9:
            warnings.append(f"⚠️  Many zero gradients in {name}: {stats['zero_fraction']*100:.1f}%")
    
    if warnings:
        print(f"\n🚨 GRADIENT WARNINGS (Step {step}):")
        for warning in warnings:
            print(f"   {warning}")
        print()