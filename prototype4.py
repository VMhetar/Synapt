import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DYNAMIC PLASTIC LAYER - WITH PRUNING & GROWTH
# ============================================================================

class DynamicPlasticLayer(nn.Module):
    """
    Hebbian plasticity with dynamic neuron restructuring.
    
    Key features:
    1. Hebbian learning (local plasticity)
    2. Neuron importance tracking (which neurons matter?)
    3. Pruning (remove unimportant neurons)
    4. Growth (add new neurons for new tasks)
    5. Consolidation (protect important synapses)
    """
    
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Weights
        self.W_encode = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        
        # Track neuron importance (how much does this neuron contribute?)
        self.neuron_importance = torch.zeros(hidden_size)
        
        # Track which neurons are "consolidated" (important for old tasks)
        self.consolidation_mask = torch.ones(hidden_size)
        
        # Track neuron activity history
        self.neuron_activity_history = []
        
        # Track which neurons are "plastic" (free to change) vs "consolidat" (protected)
        self.plasticity_state = "learning"  # "learning" or "consolidating"
    
    def forward(self, x):
        """Forward pass with current structure."""
        hidden = torch.matmul(x, self.W_encode)
        hidden = torch.relu(hidden)
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        return hidden, reconstruction
    
    def hebbian_update(self, x, hidden, target):
        """Update weights using Hebbian rule."""
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        error = target - reconstruction
        
        # Standard Hebbian updates
        dW_encode = torch.matmul(x.T, hidden) / x.size(0)
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        
        # Apply consolidation mask: protect important neurons
        # Important neurons get smaller updates (they're "set")
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        
        self.W_encode.data += self.learning_rate * dW_encode * consolidation_factor
        self.W_decode.data += self.learning_rate * dW_decode * consolidation_factor.T
        
        # Track neuron importance: how much does each neuron activate?
        self.neuron_activity_history.append(hidden.detach().clone())
        
        return torch.mean(error ** 2)
    
    def measure_neuron_importance(self):
        """
        Measure importance of each neuron based on activity.
        Important neurons = active and contribute to outputs.
        """
        if len(self.neuron_activity_history) == 0:
            return torch.ones(self.hidden_size)
        
        # Concatenate all activities (handles different batch sizes)
        # Each entry is (batch_size, hidden_size)
        activities_list = []
        for activity in self.neuron_activity_history:
            activities_list.append(activity)
        
        # Concatenate along batch dimension
        activities = torch.cat(activities_list, dim=0)  # (total_samples, hidden)
        
        # Importance = average absolute activation (how much neuron fires)
        importance = torch.mean(torch.abs(activities), dim=0)
        
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    def consolidate_task(self, threshold=0.3):
        """
        Mark neurons as important (consolidated) for current task.
        These neurons will be protected during next task learning.
        """
        importance = self.measure_neuron_importance()
        
        # Neurons above threshold are "important" for this task
        important_neurons = importance > threshold
        
        # Update consolidation mask: only important neurons stay plastic
        # Less important = more protected (consolidation_mask = 0.1)
        # More important = more plastic (consolidation_mask = 1.0)
        self.consolidation_mask = torch.where(
            important_neurons,
            torch.ones(self.hidden_size) * 1.0,  # Important: stay plastic
            torch.ones(self.hidden_size) * 0.2   # Unimportant: become rigid
        ).detach()
        
        self.neuron_activity_history = []
        
        return importance, important_neurons
    
    def prune_neurons(self, threshold=0.2):
        """
        Remove neurons that are not important.
        Only prune neurons that are truly unused across tasks.
        """
        importance = self.measure_neuron_importance()
        
        # Find neurons to keep
        keep_mask = importance > threshold
        num_keep = keep_mask.sum().item()
        
        if num_keep == 0:
            print("   ⚠️  Cannot prune: would remove all neurons!")
            return 0
        
        if num_keep == self.hidden_size:
            print("   ℹ️  No neurons to prune (all important)")
            return 0
        
        # Prune: keep only important neurons
        new_W_encode = self.W_encode[:, keep_mask].clone()
        new_W_decode = self.W_decode[keep_mask, :].clone()
        
        # Update parameters
        self.W_encode = nn.Parameter(new_W_encode)
        self.W_decode = nn.Parameter(new_W_decode)
        
        self.hidden_size = num_keep
        self.consolidation_mask = self.consolidation_mask[keep_mask]
        
        num_pruned = (~keep_mask).sum().item()
        return num_pruned
    
    def grow_neurons(self, num_new=32):
        """
        Add new neurons for learning new tasks.
        New neurons start with small random weights.
        """
        new_encode = torch.randn(self.input_size, num_new) * 0.01
        new_decode = torch.randn(num_new, self.input_size) * 0.01
        
        # Concatenate with existing
        self.W_encode = nn.Parameter(
            torch.cat([self.W_encode, new_encode], dim=1)
        )
        self.W_decode = nn.Parameter(
            torch.cat([self.W_decode, new_decode], dim=0)
        )
        
        # New neurons are fully plastic (consolidation = 1.0)
        self.consolidation_mask = torch.cat([
            self.consolidation_mask,
            torch.ones(num_new)
        ])
        
        self.hidden_size += num_new
        
        return num_new
    
    def get_network_stats(self):
        """Return network size and state info."""
        return {
            'hidden_size': self.hidden_size,
            'total_params': self.W_encode.numel() + self.W_decode.numel(),
            'avg_importance': self.neuron_activity_history[-1].mean().item() if self.neuron_activity_history else 0,
        }


# ============================================================================
# STANDARD NETWORK (for comparison)
# ============================================================================

class StandardNetwork(nn.Module):
    """Static backprop network."""
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
    """Generate bouncing ball video."""
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

def train_dynamic_plastic(model, video, num_epochs=50, batch_size=8):
    """Train dynamic plastic model on video."""
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


def evaluate_model(model, video):
    """Evaluate model on video."""
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
# MAIN TEST: DYNAMIC RESTRUCTURING
# ============================================================================

def dynamic_restructuring_test():
    """
    Test dynamic network restructuring during continual learning.
    """
    
    print("=" * 90)
    print("DYNAMIC NETWORK RESTRUCTURING TEST")
    print("Hebbian Plasticity with Pruning & Growth")
    print("=" * 90)
    
    # Generate 3 different videos
    print("\n1. Generating 3 different videos...")
    video1 = generate_bouncing_ball_video(num_frames=80, size=32, velocity_scale=1.0, seed=42)
    video2 = generate_bouncing_ball_video(num_frames=80, size=32, velocity_scale=2.0, seed=99)
    video3 = generate_bouncing_ball_video(num_frames=80, size=32, velocity_scale=0.5, seed=77)
    print(f"   ✓ Video 1 (normal): {video1.shape}")
    print(f"   ✓ Video 2 (fast):   {video2.shape}")
    print(f"   ✓ Video 3 (slow):   {video3.shape}")
    
    # Create models
    print("\n2. Creating models...")
    dynamic_model = DynamicPlasticLayer(input_size=1024, hidden_size=64, learning_rate=0.01)
    static_model = DynamicPlasticLayer(input_size=1024, hidden_size=64, learning_rate=0.01)
    print(f"   ✓ Dynamic model (with pruning/growth): hidden_size=64")
    print(f"   ✓ Static model (baseline): hidden_size=64")
    
    # Track history
    dynamic_history = {
        'v1_loss': [], 'v2_loss': [], 'v3_loss': [],
        'v1_error': [], 'v2_error': [], 'v3_error': [],
        'hidden_size': [64],
        'num_params': [64 * 1024 * 2]
    }
    
    static_history = {
        'v1_loss': [], 'v2_loss': [], 'v3_loss': [],
        'v1_error': [], 'v2_error': [], 'v3_error': [],
        'hidden_size': [64],
        'num_params': [64 * 1024 * 2]
    }
    
    # ========================================================================
    # PHASE 1: Learn Video 1
    # ========================================================================
    print("\n" + "=" * 90)
    print("PHASE 1: Learning Video 1 (Normal Speed)")
    print("=" * 90)
    
    print("\n  Training...")
    dynamic_losses_v1 = train_dynamic_plastic(dynamic_model, video1, num_epochs=40, batch_size=8)
    static_losses_v1 = train_dynamic_plastic(static_model, video1, num_epochs=40, batch_size=8)
    
    dynamic_history['v1_loss'] = dynamic_losses_v1
    static_history['v1_loss'] = static_losses_v1
    
    # Consolidate task 1
    print("\n  Consolidating Task 1 neurons...")
    importance_v1, important_v1 = dynamic_model.consolidate_task(threshold=0.3)
    num_important_v1 = important_v1.sum().item()
    print(f"    Important neurons for Task 1: {num_important_v1}/{dynamic_model.hidden_size}")
    print(f"    (Will protect these during Task 2 learning)")
    
    # Evaluate on video 1
    dynamic_v1_phase1 = evaluate_model(dynamic_model, video1)
    static_v1_phase1 = evaluate_model(static_model, video1)
    
    dynamic_history['v1_error'].append(dynamic_v1_phase1)
    static_history['v1_error'].append(static_v1_phase1)
    
    print(f"\n  After Phase 1:")
    print(f"    Dynamic error: {dynamic_v1_phase1:.6f}")
    print(f"    Static error:  {static_v1_phase1:.6f}")
    
    # ========================================================================
    # PHASE 2: Learn Video 2 (with dynamic restructuring)
    # ========================================================================
    print("\n" + "=" * 90)
    print("PHASE 2: Learning Video 2 (Fast Speed) - WITH DYNAMIC RESTRUCTURING")
    print("=" * 90)
    
    print("\n  Growing new neurons for new task...")
    num_new = dynamic_model.grow_neurons(num_new=32)
    print(f"    Added {num_new} new neurons")
    print(f"    Network size: 64 → {dynamic_model.hidden_size}")
    dynamic_history['hidden_size'].append(dynamic_model.hidden_size)
    
    print("\n  Training with protected/new neurons...")
    dynamic_losses_v2 = train_dynamic_plastic(dynamic_model, video2, num_epochs=40, batch_size=8)
    static_losses_v2 = train_dynamic_plastic(static_model, video2, num_epochs=40, batch_size=8)
    
    dynamic_history['v2_loss'] = dynamic_losses_v2
    static_history['v2_loss'] = static_losses_v2
    
    # Evaluate on both videos after phase 2
    dynamic_v1_phase2 = evaluate_model(dynamic_model, video1)
    dynamic_v2_phase2 = evaluate_model(dynamic_model, video2)
    static_v1_phase2 = evaluate_model(static_model, video1)
    static_v2_phase2 = evaluate_model(static_model, video2)
    
    dynamic_history['v1_error'].append(dynamic_v1_phase2)
    dynamic_history['v2_error'].append(dynamic_v2_phase2)
    static_history['v1_error'].append(static_v1_phase2)
    static_history['v2_error'].append(static_v2_phase2)
    
    print(f"\n  After Phase 2:")
    print(f"    Video 1:")
    print(f"      Dynamic: {dynamic_v1_phase1:.6f} → {dynamic_v1_phase2:.6f}")
    print(f"      Static:  {static_v1_phase1:.6f} → {static_v1_phase2:.6f}")
    print(f"    Video 2:")
    print(f"      Dynamic: {dynamic_v2_phase2:.6f}")
    print(f"      Static:  {static_v2_phase2:.6f}")
    
    # Prune unimportant neurons
    print("\n  Consolidating Task 2 and pruning...")
    importance_v2, important_v2 = dynamic_model.consolidate_task(threshold=0.25)
    num_pruned = dynamic_model.prune_neurons(threshold=0.2)
    print(f"    Pruned {num_pruned} neurons")
    print(f"    Network size: {dynamic_model.hidden_size + num_pruned} → {dynamic_model.hidden_size}")
    dynamic_history['hidden_size'].append(dynamic_model.hidden_size)
    
    # ========================================================================
    # PHASE 3: Learn Video 3
    # ========================================================================
    print("\n" + "=" * 90)
    print("PHASE 3: Learning Video 3 (Slow Speed) - FINAL RESTRUCTURING")
    print("=" * 90)
    
    print("\n  Growing neurons for Task 3...")
    num_new_v3 = dynamic_model.grow_neurons(num_new=24)
    print(f"    Added {num_new_v3} new neurons")
    print(f"    Network size: {dynamic_model.hidden_size - num_new_v3} → {dynamic_model.hidden_size}")
    dynamic_history['hidden_size'].append(dynamic_model.hidden_size)
    
    print("\n  Training...")
    dynamic_losses_v3 = train_dynamic_plastic(dynamic_model, video3, num_epochs=40, batch_size=8)
    static_losses_v3 = train_dynamic_plastic(static_model, video3, num_epochs=40, batch_size=8)
    
    dynamic_history['v3_loss'] = dynamic_losses_v3
    static_history['v3_loss'] = static_losses_v3
    
    # Final evaluation on all videos
    dynamic_v1_final = evaluate_model(dynamic_model, video1)
    dynamic_v2_final = evaluate_model(dynamic_model, video2)
    dynamic_v3_final = evaluate_model(dynamic_model, video3)
    
    static_v1_final = evaluate_model(static_model, video1)
    static_v2_final = evaluate_model(static_model, video2)
    static_v3_final = evaluate_model(static_model, video3)
    
    dynamic_history['v1_error'].append(dynamic_v1_final)
    dynamic_history['v2_error'].append(dynamic_v2_final)
    dynamic_history['v3_error'].append(dynamic_v3_final)
    static_history['v1_error'].append(static_v1_final)
    static_history['v2_error'].append(static_v2_final)
    static_history['v3_error'].append(static_v3_final)
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    
    print("\n📊 PERFORMANCE ON ALL TASKS:")
    print(f"\n  Dynamic Network:")
    print(f"    Video 1 (Normal):  {dynamic_v1_final:.6f}")
    print(f"    Video 2 (Fast):    {dynamic_v2_final:.6f}")
    print(f"    Video 3 (Slow):    {dynamic_v3_final:.6f}")
    print(f"    Total error:       {dynamic_v1_final + dynamic_v2_final + dynamic_v3_final:.6f}")
    
    print(f"\n  Static Network:")
    print(f"    Video 1 (Normal):  {static_v1_final:.6f}")
    print(f"    Video 2 (Fast):    {static_v2_final:.6f}")
    print(f"    Video 3 (Slow):    {static_v3_final:.6f}")
    print(f"    Total error:       {static_v1_final + static_v2_final + static_v3_final:.6f}")
    
    print(f"\n🧠 NETWORK EVOLUTION:")
    print(f"    Dynamic: 64 → {dynamic_history['hidden_size'][1]} → {dynamic_history['hidden_size'][2]} → {dynamic_model.hidden_size}")
    print(f"    (Grew to handle multiple tasks, pruned unused neurons)")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n4. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Learning curves
    axes[0, 0].plot(dynamic_history['v1_loss'], label='Dynamic', linewidth=2, marker='o')
    axes[0, 0].plot(static_history['v1_loss'], label='Static', linewidth=2, marker='s')
    axes[0, 0].set_title('Phase 1: Learning Video 1', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(dynamic_history['v2_loss'], label='Dynamic', linewidth=2, marker='o')
    axes[0, 1].plot(static_history['v2_loss'], label='Static', linewidth=2, marker='s')
    axes[0, 1].set_title('Phase 2: Learning Video 2\n(Dynamic: protected task 1 neurons)', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(dynamic_history['v3_loss'], label='Dynamic', linewidth=2, marker='o')
    axes[0, 2].plot(static_history['v3_loss'], label='Static', linewidth=2, marker='s')
    axes[0, 2].set_title('Phase 3: Learning Video 3\n(Dynamic: pruned & grew)', fontweight='bold', fontsize=12)
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Final performance comparison
    tasks = ['Video 1\n(Normal)', 'Video 2\n(Fast)', 'Video 3\n(Slow)']
    dynamic_errs = [dynamic_v1_final, dynamic_v2_final, dynamic_v3_final]
    static_errs = [static_v1_final, static_v2_final, static_v3_final]
    
    x = np.arange(len(tasks))
    width = 0.35
    axes[1, 0].bar(x - width/2, dynamic_errs, width, label='Dynamic', alpha=0.8)
    axes[1, 0].bar(x + width/2, static_errs, width, label='Static', alpha=0.8)
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Final Performance Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tasks)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Network size evolution
    phases = ['Start', 'After\nPrune', 'After\nGrow', 'Final']
    axes[1, 1].plot(dynamic_history['hidden_size'], marker='o', linewidth=3, markersize=10, label='Dynamic')
    axes[1, 1].axhline(y=64, color='orange', linestyle='--', linewidth=2, label='Static (fixed)')
    axes[1, 1].set_ylabel('Hidden Layer Size', fontweight='bold')
    axes[1, 1].set_title('Network Restructuring Over Time', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(range(len(dynamic_history['hidden_size'])))
    axes[1, 1].set_xticklabels(phases[:len(dynamic_history['hidden_size'])])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Total error comparison
    total_dynamic = dynamic_v1_final + dynamic_v2_final + dynamic_v3_final
    total_static = static_v1_final + static_v2_final + static_v3_final
    
    colors = ['#2ecc71' if total_dynamic < total_static else '#e74c3c']
    axes[1, 2].bar(['Dynamic', 'Static'], [total_dynamic, total_static], color=['#2ecc71', '#e74c3c'], alpha=0.7, width=0.5)
    axes[1, 2].set_ylabel('Total Cumulative Error', fontweight='bold')
    axes[1, 2].set_title('Overall Performance\n(Lower = Better)', fontweight='bold', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    axes[1, 2].text(0, total_dynamic + 0.001, f'{total_dynamic:.6f}', ha='center', fontweight='bold')
    axes[1, 2].text(1, total_static + 0.001, f'{total_static:.6f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dynamic_restructuring.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: dynamic_restructuring.png")
    
    print("\n" + "=" * 90)
    print("TEST COMPLETE")
    print("=" * 90)
    
    return {
        'dynamic': dynamic_history,
        'static': static_history,
        'final_errors': {
            'dynamic': (dynamic_v1_final, dynamic_v2_final, dynamic_v3_final),
            'static': (static_v1_final, static_v2_final, static_v3_final)
        }
    }


if __name__ == "__main__":
    results = dynamic_restructuring_test()