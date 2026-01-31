import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PLASTICITY-BASED MODELS
# ============================================================================

class PlasticLayer(nn.Module):
    """Base class for plasticity-based learning."""
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
        """Standard Hebbian: w_ij += eta * a_i * a_j"""
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        error = target - reconstruction
        
        # Hebbian: input-hidden correlation
        dW_encode = torch.matmul(x.T, hidden) / x.size(0)
        self.W_encode.data += self.learning_rate * dW_encode
        
        # Error-driven decoder update
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        self.W_decode.data += self.learning_rate * dW_decode
        
        return torch.mean(error ** 2)


class OjasLayer(nn.Module):
    """Oja's rule: normalized Hebbian learning."""
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
    
    def update(self, x, hidden, target):
        """Oja's rule: w_ij += eta * a_j * (a_i - a_j * w_ij)"""
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        error = target - reconstruction
        
        # Oja's rule for encoder: normalized Hebbian
        # Hebbian term
        hebbian_term = torch.matmul(x.T, hidden) / x.size(0)
        # Anti-Hebbian self-scaling term
        post_norm = torch.sum(hidden ** 2, dim=0, keepdim=True) / x.size(0)
        anti_hebbian_term = post_norm * self.W_encode
        
        dW_encode = hebbian_term - anti_hebbian_term
        self.W_encode.data += self.learning_rate * dW_encode
        
        # Decoder: error-driven
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        self.W_decode.data += self.learning_rate * dW_decode
        
        return torch.mean(error ** 2)


# ============================================================================
# STANDARD BACKPROP MODEL (for comparison)
# ============================================================================

class StandardNetwork(nn.Module):
    """Standard neural network trained with backprop + SGD."""
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
# DATA GENERATION
# ============================================================================

def generate_bouncing_ball_video(num_frames=100, size=32, seed=42):
    """Generate synthetic video of bouncing ball."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    frames = []
    x, y = size // 2, size // 2
    vx, vy = 1.5, 1.0
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
# TRAINING FUNCTIONS
# ============================================================================

def train_hebbian(model, video, num_epochs=50, batch_size=8):
    """Train plasticity model with Hebbian rule."""
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            
            current_frames = video[start_idx:end_idx].reshape(-1, 1024)
            next_frames = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            hidden, reconstruction = model.forward(current_frames)
            loss = model.hebbian_update(current_frames, hidden, next_frames)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Hebbian - Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def train_oja(model, video, num_epochs=50, batch_size=8):
    """Train plasticity model with Oja's rule."""
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            
            current_frames = video[start_idx:end_idx].reshape(-1, 1024)
            next_frames = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            hidden, reconstruction = model.forward(current_frames)
            loss = model.update(current_frames, hidden, next_frames)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Oja's - Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def train_backprop(model, video, num_epochs=50, batch_size=8, learning_rate=0.01):
    """Train standard network with backprop."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            
            current_frames = video[start_idx:end_idx].reshape(-1, 1024)
            next_frames = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            optimizer.zero_grad()
            hidden, reconstruction = model.forward(current_frames)
            loss = criterion(reconstruction, next_frames)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Backprop - Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def evaluate_model(model, video):
    """Measure average prediction error on entire video."""
    with torch.no_grad():
        total_error = 0
        for i in range(len(video) - 1):
            current = video[i].reshape(1, 1024)
            actual_next = video[i + 1]
            
            hidden, predicted = model.forward(current)
            error = torch.mean((predicted - actual_next.reshape(1, 1024)) ** 2)
            total_error += error.item()
        
        avg_error = total_error / (len(video) - 1)
    
    return avg_error


def visualize_comparison(models_dict, video, num_examples=3):
    """Compare predictions across different models."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(num_examples, n_models + 2, figsize=(4*(n_models+2), 3*num_examples))
    
    with torch.no_grad():
        for ex in range(num_examples):
            idx = ex * 20
            current = video[idx].reshape(1, 1024)
            actual_next = video[idx + 1].reshape(32, 32)
            
            # Show current and actual
            axes[ex, 0].imshow(video[idx].reshape(32, 32), cmap='gray')
            axes[ex, 0].set_title('Current' if ex == 0 else '')
            axes[ex, 0].axis('off')
            
            axes[ex, 1].imshow(actual_next, cmap='gray')
            axes[ex, 1].set_title('Actual Next' if ex == 0 else '')
            axes[ex, 1].axis('off')
            
            # Show predictions from each model
            for model_idx, (name, model) in enumerate(models_dict.items()):
                hidden, predicted = model.forward(current)
                predicted = predicted.reshape(32, 32)
                
                axes[ex, model_idx + 2].imshow(predicted.numpy(), cmap='gray')
                axes[ex, model_idx + 2].set_title(name if ex == 0 else '')
                axes[ex, model_idx + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    print("Saved model_comparison.png")


def plot_learning_curves(results):
    """Plot learning curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Learning curves
    for name, losses in results.items():
        axes[0].plot(losses, label=name, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Learning Curves', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final performance
    names = list(results.keys())
    final_losses = [results[name][-1] for name in names]
    initial_losses = [results[name][0] for name in names]
    improvements = [((initial_losses[i] - final_losses[i]) / initial_losses[i] * 100) 
                   for i in range(len(names))]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[1].bar(names, improvements, color=colors, alpha=0.7)
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].set_title('Learning Improvement\n(Initial Loss → Final Loss)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (name, imp) in enumerate(zip(names, improvements)):
        axes[1].text(i, imp + 1, f'{imp:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('learning_comparison.png', dpi=100, bbox_inches='tight')
    print("Saved learning_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPARING PLASTICITY RULES vs BACKPROPAGATION")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating video...")
    video = generate_bouncing_ball_video(num_frames=100, size=32)
    print(f"   Video shape: {video.shape}")
    
    # Create models
    print("\n2. Creating models...")
    hebbian_model = PlasticLayer(input_size=1024, hidden_size=128, learning_rate=0.01)
    oja_model = OjasLayer(input_size=1024, hidden_size=128, learning_rate=0.01)
    backprop_model = StandardNetwork(input_size=1024, hidden_size=128)
    print("   ✓ Hebbian (plasticity)")
    print("   ✓ Oja's Rule (plasticity)")
    print("   ✓ Backprop (standard)")
    
    # Train models
    print("\n3. Training models (100 epochs)...")
    print("-" * 70)
    
    print("\n  Training Hebbian...")
    hebbian_losses = train_hebbian(hebbian_model, video, num_epochs=100, batch_size=8)
    
    print("\n  Training Oja's...")
    oja_losses = train_oja(oja_model, video, num_epochs=100, batch_size=8)
    
    print("\n  Training Backprop...")
    backprop_losses = train_backprop(backprop_model, video, num_epochs=100, batch_size=8, learning_rate=0.01)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("4. RESULTS")
    print("=" * 70)
    
    results = {
        "Hebbian": hebbian_losses,
        "Oja's Rule": oja_losses,
        "Backprop": backprop_losses
    }
    
    models = {
        "Hebbian": hebbian_model,
        "Oja's Rule": oja_model,
        "Backprop": backprop_model
    }
    
    for name in results.keys():
        initial = results[name][0]
        final = results[name][-1]
        improvement = ((initial - final) / initial * 100)
        eval_error = evaluate_model(models[name], video)
        
        print(f"\n{name}:")
        print(f"  Initial loss:    {initial:.6f}")
        print(f"  Final loss:      {final:.6f}")
        print(f"  Improvement:     {improvement:.1f}%")
        print(f"  Eval error:      {eval_error:.6f}")
    
    # Visualize
    print("\n5. Generating visualizations...")
    plot_learning_curves(results)
    visualize_comparison(models, video, num_examples=4)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("- Different learning rules show different convergence patterns")
    print("- Backprop has global gradient information (advantage)")
    print("- Plasticity rules are LOCAL only (biological advantage)")
    print("- What matters: does one approach work better on some tasks?")
    print("=" * 70)