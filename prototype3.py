import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# MODELS
# ============================================================================

class PlasticLayer(nn.Module):
    """Hebbian plasticity - local learning."""
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        self.W_encode = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
    
    def forward(self, x):
        hidden = torch.matmul(x, self.W_encode)
        hidden = torch.relu(hidden)
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        return hidden, reconstruction
    
    def hebbian_update(self, x, hidden, target):
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        error = target - reconstruction
        
        dW_encode = torch.matmul(x.T, hidden) / x.size(0)
        self.W_encode.data += self.learning_rate * dW_encode
        
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        self.W_decode.data += self.learning_rate * dW_decode
        
        return torch.mean(error ** 2)


class StandardNetwork(nn.Module):
    """Standard network with backprop."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        hidden = self.encoder(x)
        reconstruction = self.decoder(hidden)
        return hidden, reconstruction


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def generate_bouncing_ball_video(num_frames=100, size=32, velocity_scale=1.0, seed=42):
    """Generate bouncing ball video with configurable velocity."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    frames = []
    x, y = size // 2, size // 2
    vx, vy = 1.5 * velocity_scale, 1.0 * velocity_scale
    radius = 2
    
    for _ in range(num_frames):
        frame = np.zeros((size, size))
        x += vx
        y += vy
        
        if x < radius or x > size - radius:
            vx *= -1
            x = np.clip(x, radius, size - radius)
        if y < radius or y > size - radius:
            vy *= -1
            y = np.clip(y, radius, size - radius)
        
        xx = np.arange(size)
        yy = np.arange(size)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - int(x))**2 + (YY - int(y))**2 <= radius**2
        frame[mask] = 1.0
        
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


# ============================================================================
# TRAINING
# ============================================================================

def train_plasticity_on_video(model, video, num_epochs=50, batch_size=8):
    """Train plasticity model on video."""
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            current = video[start_idx:end_idx].reshape(-1, 1024)
            target = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            hidden, _ = model.forward(current)
            loss = model.hebbian_update(current, hidden, target)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return losses


def train_backprop_on_video(model, video, num_epochs=50, batch_size=8, learning_rate=0.01):
    """Train backprop model on video."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            current = video[start_idx:end_idx].reshape(-1, 1024)
            target = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            optimizer.zero_grad()
            hidden, reconstruction = model.forward(current)
            loss = criterion(reconstruction, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return losses


def evaluate_on_video(model, video):
    """Measure average error on entire video."""
    with torch.no_grad():
        total_error = 0
        for i in range(len(video) - 1):
            current = video[i].reshape(1, 1024)
            target = video[i + 1]
            hidden, predicted = model.forward(current)
            error = torch.mean((predicted - target.reshape(1, 1024)) ** 2)
            total_error += error.item()
    
    return total_error / (len(video) - 1)


# ============================================================================
# CONTINUAL LEARNING TEST
# ============================================================================

def continual_learning_test():
    """
    Test catastrophic forgetting:
    1. Train on video 1
    2. Train on video 2
    3. Test on both - does plasticity remember video 1?
    """
    
    print("=" * 80)
    print("CONTINUAL LEARNING TEST: CATASTROPHIC FORGETTING")
    print("=" * 80)
    
    # Generate two different videos
    print("\n1. Generating videos...")
    video1 = generate_bouncing_ball_video(num_frames=100, size=32, velocity_scale=1.0, seed=42)
    video2 = generate_bouncing_ball_video(num_frames=100, size=32, velocity_scale=2.0, seed=99)
    print(f"   Video 1 (slow): {video1.shape}")
    print(f"   Video 2 (fast): {video2.shape}")
    
    # Create models
    print("\n2. Creating models...")
    plasticity_model = PlasticLayer(input_size=1024, hidden_size=128, learning_rate=0.01)
    backprop_model = StandardNetwork(input_size=1024, hidden_size=128)
    print("   ✓ Plasticity model")
    print("   ✓ Backprop model")
    
    # PHASE 1: Train on video 1
    print("\n3. PHASE 1: Training on Video 1 (SLOW motion)...")
    print("-" * 80)
    
    plasticity_losses_v1_phase1 = train_plasticity_on_video(plasticity_model, video1, num_epochs=50, batch_size=8)
    backprop_losses_v1_phase1 = train_backprop_on_video(backprop_model, video1, num_epochs=50, batch_size=8)
    
    print(f"   Plasticity - Initial: {plasticity_losses_v1_phase1[0]:.6f}, Final: {plasticity_losses_v1_phase1[-1]:.6f}")
    print(f"   Backprop - Initial: {backprop_losses_v1_phase1[0]:.6f}, Final: {backprop_losses_v1_phase1[-1]:.6f}")
    
    # Evaluate on video 1 after phase 1
    plasticity_v1_after_phase1 = evaluate_on_video(plasticity_model, video1)
    backprop_v1_after_phase1 = evaluate_on_video(backprop_model, video1)
    
    print(f"\n   After Phase 1 - Video 1 Error:")
    print(f"     Plasticity: {plasticity_v1_after_phase1:.6f}")
    print(f"     Backprop:   {backprop_v1_after_phase1:.6f}")
    
    # PHASE 2: Train on video 2 (different dynamics)
    print("\n4. PHASE 2: Training on Video 2 (FAST motion)...")
    print("   (This is where catastrophic forgetting might happen)")
    print("-" * 80)
    
    plasticity_losses_v2_phase2 = train_plasticity_on_video(plasticity_model, video2, num_epochs=50, batch_size=8)
    backprop_losses_v2_phase2 = train_backprop_on_video(backprop_model, video2, num_epochs=50, batch_size=8)
    
    print(f"   Plasticity - Initial: {plasticity_losses_v2_phase2[0]:.6f}, Final: {plasticity_losses_v2_phase2[-1]:.6f}")
    print(f"   Backprop - Initial: {backprop_losses_v2_phase2[0]:.6f}, Final: {backprop_losses_v2_phase2[-1]:.6f}")
    
    # PHASE 3: TEST on both videos (check for forgetting)
    print("\n5. PHASE 3: Testing on BOTH videos (checking for catastrophic forgetting)...")
    print("-" * 80)
    
    # Test on video 1 after training on video 2
    plasticity_v1_after_phase2 = evaluate_on_video(plasticity_model, video1)
    backprop_v1_after_phase2 = evaluate_on_video(backprop_model, video1)
    
    # Test on video 2
    plasticity_v2_final = evaluate_on_video(plasticity_model, video2)
    backprop_v2_final = evaluate_on_video(backprop_model, video2)
    
    print("\n   VIDEO 1 (Slow) - Performance After Learning Video 2:")
    print(f"     Plasticity: {plasticity_v1_after_phase1:.6f} → {plasticity_v1_after_phase2:.6f}")
    forgetting_p = ((plasticity_v1_after_phase2 - plasticity_v1_after_phase1) / plasticity_v1_after_phase1 * 100)
    print(f"     Forgetting: {forgetting_p:+.1f}%")
    
    print(f"\n     Backprop: {backprop_v1_after_phase1:.6f} → {backprop_v1_after_phase2:.6f}")
    forgetting_b = ((backprop_v1_after_phase2 - backprop_v1_after_phase1) / backprop_v1_after_phase1 * 100)
    print(f"     Forgetting: {forgetting_b:+.1f}%")
    
    print("\n   VIDEO 2 (Fast) - Final Performance:")
    print(f"     Plasticity: {plasticity_v2_final:.6f}")
    print(f"     Backprop:   {backprop_v2_final:.6f}")
    
    # ========================================================================
    # RESULTS & ANALYSIS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\n🔍 CATASTROPHIC FORGETTING METRIC:")
    print(f"   Plasticity memory loss: {forgetting_p:+.1f}%")
    print(f"   Backprop memory loss:   {forgetting_b:+.1f}%")
    
    if abs(forgetting_p) < abs(forgetting_b):
        print(f"\n   ✓ PLASTICITY WINS: Better memory retention!")
        print(f"     ({abs(forgetting_b) - abs(forgetting_p):.1f}% less forgetting)")
    elif abs(forgetting_b) < abs(forgetting_p):
        print(f"\n   ✓ BACKPROP WINS: Better memory retention!")
        print(f"     ({abs(forgetting_p) - abs(forgetting_b):.1f}% less forgetting)")
    else:
        print(f"\n   = EQUAL: Both forgot equally")
    
    print("\n📊 TASK PERFORMANCE:")
    print(f"   Plasticity on video 1: {plasticity_v1_after_phase2:.6f}")
    print(f"   Plasticity on video 2: {plasticity_v2_final:.6f}")
    print(f"   Total plasticity error: {plasticity_v1_after_phase2 + plasticity_v2_final:.6f}")
    
    print(f"\n   Backprop on video 1: {backprop_v1_after_phase2:.6f}")
    print(f"   Backprop on video 2: {backprop_v2_final:.6f}")
    print(f"   Total backprop error: {backprop_v1_after_phase2 + backprop_v2_final:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Learning curves phase 1
    axes[0, 0].plot(plasticity_losses_v1_phase1, label='Plasticity', linewidth=2)
    axes[0, 0].plot(backprop_losses_v1_phase1, label='Backprop', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Phase 1: Training on Video 1 (Slow)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning curves phase 2
    axes[0, 1].plot(plasticity_losses_v2_phase2, label='Plasticity', linewidth=2)
    axes[0, 1].plot(backprop_losses_v2_phase2, label='Backprop', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Phase 2: Training on Video 2 (Fast)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory retention comparison
    models = ['Plasticity', 'Backprop']
    forgetting = [forgetting_p, forgetting_b]
    colors = ['green' if f < 0 else 'red' for f in forgetting]
    axes[0, 2].bar(models, forgetting, color=colors, alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 2].set_ylabel('Memory Loss (%)', fontweight='bold')
    axes[0, 2].set_title('Catastrophic Forgetting\n(Lower = Better)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    for i, (model, f) in enumerate(zip(models, forgetting)):
        axes[0, 2].text(i, f + (5 if f > 0 else -5), f'{f:+.1f}%', 
                       ha='center', fontsize=11, fontweight='bold')
    
    # Video 1 performance over time
    video1_plasticity = [plasticity_v1_after_phase1, plasticity_v1_after_phase2]
    video1_backprop = [backprop_v1_after_phase1, backprop_v1_after_phase2]
    
    axes[1, 0].plot(['After Phase 1', 'After Phase 2'], video1_plasticity, 
                   marker='o', linewidth=2, markersize=8, label='Plasticity')
    axes[1, 0].plot(['After Phase 1', 'After Phase 2'], video1_backprop, 
                   marker='s', linewidth=2, markersize=8, label='Backprop')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Video 1 (Slow) Performance\nDuring Continual Learning', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final performance on both videos
    final_errors = {
        'Plasticity': [plasticity_v1_after_phase2, plasticity_v2_final],
        'Backprop': [backprop_v1_after_phase2, backprop_v2_final]
    }
    
    x = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x - width/2, final_errors['Plasticity'], width, label='Plasticity', alpha=0.8)
    axes[1, 1].bar(x + width/2, final_errors['Backprop'], width, label='Backprop', alpha=0.8)
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Final Performance on Both Videos', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Video 1\n(Slow)', 'Video 2\n(Fast)'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Summary text
    axes[1, 2].axis('off')
    summary_text = f"""
CONTINUAL LEARNING TEST SUMMARY
{'='*45}

VIDEO 1 (Slow Motion):
  Plasticity before/after: {plasticity_v1_after_phase1:.6f} → {plasticity_v1_after_phase2:.6f}
  Backprop before/after:   {backprop_v1_after_phase1:.6f} → {backprop_v1_after_phase2:.6f}
  
VIDEO 2 (Fast Motion):
  Plasticity error: {plasticity_v2_final:.6f}
  Backprop error:   {backprop_v2_final:.6f}

FORGETTING (Δ on Video 1):
  Plasticity: {forgetting_p:+.1f}%
  Backprop:   {forgetting_b:+.1f}%
  
WINNER: {"PLASTICITY" if abs(forgetting_p) < abs(forgetting_b) else "BACKPROP" if abs(forgetting_b) < abs(forgetting_p) else "TIE"}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('continual_learning_test.png', dpi=100, bbox_inches='tight')
    print("   Saved: continual_learning_test.png")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    continual_learning_test()