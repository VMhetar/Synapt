import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# MODELS
# ============================================================================

class DynamicPlasticLayer(nn.Module):
    """Hebbian plasticity with dynamic restructuring."""
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        self.W_encode = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        
        self.neuron_importance = torch.zeros(hidden_size)
        self.consolidation_mask = torch.ones(hidden_size)
        self.neuron_activity_history = []
    
    def forward(self, x):
        hidden = torch.matmul(x, self.W_encode)
        hidden = torch.relu(hidden)
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        return hidden, reconstruction
    
    def hebbian_update(self, x, hidden, target):
        """Hebbian learning with consolidation protection."""
        reconstruction = torch.matmul(hidden, self.W_decode)
        reconstruction = torch.sigmoid(reconstruction)
        error = target - reconstruction
        
        dW_encode = torch.matmul(x.T, hidden) / x.size(0)
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_encode.data += self.learning_rate * dW_encode * consolidation_factor
        self.W_decode.data += self.learning_rate * dW_decode * consolidation_factor.T
        
        self.neuron_activity_history.append(hidden.detach().clone())
        return torch.mean(error ** 2)
    
    def measure_neuron_importance(self):
        if len(self.neuron_activity_history) == 0:
            return torch.ones(self.hidden_size)
        
        activities_list = [activity for activity in self.neuron_activity_history]
        activities = torch.cat(activities_list, dim=0)
        importance = torch.mean(torch.abs(activities), dim=0)
        
        if importance.max() > 0:
            importance = importance / importance.max()
        return importance
    
    def consolidate_task(self, threshold=0.3):
        importance = self.measure_neuron_importance()
        important_neurons = importance > threshold
        
        self.consolidation_mask = torch.where(
            important_neurons,
            torch.ones(self.hidden_size) * 1.0,
            torch.ones(self.hidden_size) * 0.2
        ).detach()
        
        self.neuron_activity_history = []
        return importance, important_neurons
    
    def prune_neurons(self, threshold=0.2):
        importance = self.measure_neuron_importance()
        keep_mask = importance > threshold
        num_keep = keep_mask.sum().item()
        
        if num_keep == 0 or num_keep == self.hidden_size:
            return 0
        
        self.W_encode = nn.Parameter(self.W_encode[:, keep_mask].clone())
        self.W_decode = nn.Parameter(self.W_decode[keep_mask, :].clone())
        
        self.hidden_size = num_keep
        self.consolidation_mask = self.consolidation_mask[keep_mask]
        
        num_pruned = (~keep_mask).sum().item()
        return num_pruned
    
    def grow_neurons(self, num_new=32):
        new_encode = torch.randn(self.input_size, num_new) * 0.01
        new_decode = torch.randn(num_new, self.input_size) * 0.01
        
        self.W_encode = nn.Parameter(torch.cat([self.W_encode, new_encode], dim=1))
        self.W_decode = nn.Parameter(torch.cat([self.W_decode, new_decode], dim=0))
        
        self.consolidation_mask = torch.cat([self.consolidation_mask, torch.ones(num_new)])
        self.hidden_size += num_new
        
        return num_new


class StaticPlasticLayer(nn.Module):
    """Hebbian plasticity WITHOUT dynamic restructuring."""
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        self.W_encode = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.consolidation_mask = torch.ones(hidden_size)
        self.neuron_activity_history = []
    
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
        dW_decode = torch.matmul(hidden.T, error) / x.size(0)
        
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_encode.data += self.learning_rate * dW_encode * consolidation_factor
        self.W_decode.data += self.learning_rate * dW_decode * consolidation_factor.T
        
        self.neuron_activity_history.append(hidden.detach().clone())
        return torch.mean(error ** 2)
    
    def measure_neuron_importance(self):
        if len(self.neuron_activity_history) == 0:
            return torch.ones(self.hidden_size)
        
        activities_list = [activity for activity in self.neuron_activity_history]
        activities = torch.cat(activities_list, dim=0)
        importance = torch.mean(torch.abs(activities), dim=0)
        
        if importance.max() > 0:
            importance = importance / importance.max()
        return importance
    
    def consolidate_task(self, threshold=0.3):
        importance = self.measure_neuron_importance()
        important_neurons = importance > threshold
        
        self.consolidation_mask = torch.where(
            important_neurons,
            torch.ones(self.hidden_size) * 1.0,
            torch.ones(self.hidden_size) * 0.2
        ).detach()
        
        self.neuron_activity_history = []
        return importance, important_neurons


class DynamicBackpropLayer(nn.Module):
    """Backprop with dynamic restructuring."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        self.consolidation_mask = torch.ones(hidden_size)
        self.neuron_activity_history = []
    
    def forward(self, x):
        hidden = self.encoder(x)
        hidden = torch.relu(hidden)
        reconstruction = self.decoder(hidden)
        reconstruction = torch.sigmoid(reconstruction)
        return hidden, reconstruction
    
    def backprop_update(self, x, target):
        self.optimizer.zero_grad()
        hidden, reconstruction = self.forward(x)
        loss = self.criterion(reconstruction, target)
        loss.backward()
        
        # Mask gradients - protect consolidated neurons
        with torch.no_grad():
            # encoder: (hidden_size, input_size) - mask the hidden dimension
            if self.encoder.weight.grad is not None:
                self.encoder.weight.grad *= self.consolidation_mask.unsqueeze(1)
            # decoder: (input_size, hidden_size) - mask the hidden dimension
            if self.decoder.weight.grad is not None:
                self.decoder.weight.grad *= self.consolidation_mask.unsqueeze(0)
        
        self.optimizer.step()
        self.neuron_activity_history.append(hidden.detach().clone())
        return loss.item()
    
    def measure_neuron_importance(self):
        if len(self.neuron_activity_history) == 0:
            return torch.ones(self.hidden_size)
        
        activities = torch.cat(self.neuron_activity_history, dim=0)
        importance = torch.mean(torch.abs(activities), dim=0)
        if importance.max() > 0:
            importance = importance / importance.max()
        return importance
    
    def consolidate_task(self, threshold=0.3):
        importance = self.measure_neuron_importance()
        important_neurons = importance > threshold
        self.consolidation_mask = torch.where(
            important_neurons,
            torch.ones(self.hidden_size) * 1.0,
            torch.ones(self.hidden_size) * 0.2
        ).detach()
        self.neuron_activity_history = []
        return importance, important_neurons
    
    def prune_neurons(self, threshold=0.2):
        importance = self.measure_neuron_importance()
        keep_mask = importance > threshold
        num_keep = keep_mask.sum().item()
        
        if num_keep == 0 or num_keep == self.hidden_size:
            return 0
        
        with torch.no_grad():
            new_encoder = self.encoder.weight[:, keep_mask].clone()
            new_decoder = self.decoder.weight[keep_mask, :].clone()
            self.encoder.weight = nn.Parameter(new_encoder)
            self.decoder.weight = nn.Parameter(new_decoder)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.hidden_size = num_keep
        self.consolidation_mask = self.consolidation_mask[keep_mask]
        return (~keep_mask).sum().item()
    
    def grow_neurons(self, num_new=32):
        with torch.no_grad():
            # encoder: (hidden_size, input_size) -> add new hidden neurons
            new_encoder = torch.randn(num_new, self.input_size) * 0.01
            self.encoder.weight = nn.Parameter(torch.cat([self.encoder.weight, new_encoder], dim=0))
            # Also update encoder bias
            new_encoder_bias = torch.zeros(num_new)
            self.encoder.bias = nn.Parameter(torch.cat([self.encoder.bias, new_encoder_bias], dim=0))
            
            # decoder: (input_size, hidden_size) -> add new hidden neurons as input
            new_decoder = torch.randn(self.input_size, num_new) * 0.01
            self.decoder.weight = nn.Parameter(torch.cat([self.decoder.weight, new_decoder], dim=1))
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.consolidation_mask = torch.cat([self.consolidation_mask, torch.ones(num_new)])
        self.hidden_size += num_new
        return num_new


class StaticBackpropLayer(nn.Module):
    """Standard backprop WITHOUT restructuring."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        hidden = self.encoder(x)
        hidden = torch.relu(hidden)
        reconstruction = self.decoder(hidden)
        reconstruction = torch.sigmoid(reconstruction)
        return hidden, reconstruction
    
    def backprop_update(self, x, target):
        self.optimizer.zero_grad()
        hidden, reconstruction = self.forward(x)
        loss = self.criterion(reconstruction, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# VIDEO GENERATION - DIFFERENT TYPES
# ============================================================================

def generate_video_type_1(num_frames=80, size=32, seed=42):
    """Bouncing ball - normal speed."""
    np.random.seed(seed)
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
        
        xx, yy = np.arange(size), np.arange(size)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - int(x))**2 + (YY - int(y))**2 <= radius**2
        frame[mask] = 1.0
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


def generate_video_type_2(num_frames=80, size=32, seed=42):
    """Two bouncing balls."""
    np.random.seed(seed)
    frames = []
    
    x1, y1, vx1, vy1 = 10, 10, 1.2, 1.5
    x2, y2, vx2, vy2 = 22, 22, -1.0, -1.2
    radius = 2
    
    for _ in range(num_frames):
        frame = np.zeros((size, size))
        
        for (x, y, vx, vy) in [(x1, y1, vx1, vy1), (x2, y2, vx2, vy2)]:
            xx, yy = np.arange(size), np.arange(size)
            XX, YY = np.meshgrid(xx, yy)
            mask = (XX - int(x))**2 + (YY - int(y))**2 <= radius**2
            frame[mask] = 1.0
        
        x1 += vx1
        y1 += vy1
        if x1 < radius or x1 > size - radius:
            vx1 *= -1
        if y1 < radius or y1 > size - radius:
            vy1 *= -1
        
        x2 += vx2
        y2 += vy2
        if x2 < radius or x2 > size - radius:
            vx2 *= -1
        if y2 < radius or y2 > size - radius:
            vy2 *= -1
        
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


def generate_video_type_3(num_frames=80, size=32, seed=42):
    """Rotating pattern."""
    np.random.seed(seed)
    frames = []
    
    for t in range(num_frames):
        frame = np.zeros((size, size))
        angle = (t / num_frames) * 2 * np.pi
        
        cx, cy = size // 2, size // 2
        radius_orbit = 10
        
        x = int(cx + radius_orbit * np.cos(angle))
        y = int(cy + radius_orbit * np.sin(angle))
        
        xx, yy = np.arange(size), np.arange(size)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - x)**2 + (YY - y)**2 <= 3**2
        frame[mask] = 1.0
        
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


# ============================================================================
# TRAINING
# ============================================================================

def train_plastic(model, video, num_epochs=30, batch_size=8):
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
            epoch_loss += loss
            num_batches += 1
        losses.append(epoch_loss / num_batches)
    return losses


def train_backprop(model, video, num_epochs=30, batch_size=8):
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            current = video[start_idx:end_idx].reshape(-1, 1024)
            target = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            loss = model.backprop_update(current, target)
            epoch_loss += loss
            num_batches += 1
        losses.append(epoch_loss / num_batches)
    return losses


def evaluate_model(model, video):
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
# MAIN COMPREHENSIVE TEST
# ============================================================================

def comprehensive_test():
    """Test all 4 model types on 3 different video types."""
    
    print("=" * 100)
    print("COMPREHENSIVE CONTINUAL LEARNING TEST")
    print("4 Learning Approaches × 3 Video Types × Continual Learning")
    print("=" * 100)
    
    # Generate videos
    print("\n1. Generating 3 different video types...")
    video1 = generate_video_type_1(num_frames=80, seed=42)
    video2 = generate_video_type_2(num_frames=80, seed=99)
    video3 = generate_video_type_3(num_frames=80, seed=77)
    print(f"   ✓ Video 1 (Single ball - normal)")
    print(f"   ✓ Video 2 (Two balls)")
    print(f"   ✓ Video 3 (Rotating pattern)")
    
    videos = [video1, video2, video3]
    video_names = ["V1: Single Ball", "V2: Two Balls", "V3: Rotating"]
    
    # Create models
    print("\n2. Creating 4 model types...")
    models = {
        'Hebbian + Dynamic': DynamicPlasticLayer(1024, 64),
        'Hebbian + Static': StaticPlasticLayer(1024, 64),
        'Backprop + Dynamic': DynamicBackpropLayer(1024, 64),
        'Backprop + Static': StaticBackpropLayer(1024, 64),
    }
    
    for name in models.keys():
        print(f"   ✓ {name}")
    
    # Track results
    results = {model_name: {'errors': [], 'forgetting': []} for model_name in models.keys()}
    
    print("\n3. Sequential training on 3 tasks...")
    print("-" * 100)
    
    for task_num, (video, video_name) in enumerate(zip(videos, video_names), 1):
        print(f"\n   TASK {task_num}: {video_name}")
        print(f"   {'=' * 60}")
        
        for model_name, model in models.items():
            print(f"\n     {model_name}:")
            
            # Train
            if 'Hebbian' in model_name:
                train_plastic(model, video, num_epochs=30)
            else:
                train_backprop(model, video, num_epochs=30)
            
            # Consolidate
            if hasattr(model, 'consolidate_task'):
                model.consolidate_task(threshold=0.3)
            
            # Apply restructuring
            if 'Dynamic' in model_name and task_num < 3:
                if hasattr(model, 'grow_neurons'):
                    model.grow_neurons(num_new=20)
            
            # Evaluate on ALL videos seen so far
            task_error = evaluate_model(model, video)
            print(f"       Current task error: {task_error:.6f}")
            
            results[model_name]['errors'].append(task_error)
    
    # Test memory retention
    print("\n\n4. Testing memory retention...")
    print("-" * 100)
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        for vid_num, (video, video_name) in enumerate(zip(videos, video_names), 1):
            error = evaluate_model(model, video)
            print(f"     {video_name}: {error:.6f}")
    
    # ========================================================================
    # VISUALIZATION - CLEAN AND ORGANIZED
    # ========================================================================
    print("\n\n5. Creating comprehensive visualization...")
    
    # Get final performance for each model
    final_performance = {}
    for model_name, model in models.items():
        errors = []
        for video in videos:
            errors.append(evaluate_model(model, video))
        final_performance[model_name] = errors
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ===== PLOT 1: Overall Performance Comparison (Main) =====
    ax1 = fig.add_subplot(gs[0, :])
    
    model_list = list(models.keys())
    x = np.arange(len(model_list))
    width = 0.25
    
    for i, video_name in enumerate(video_names):
        values = [final_performance[model][i] for model in model_list]
        ax1.bar(x + (i - 1) * width, values, width, label=video_name, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Prediction Error', fontweight='bold', fontsize=12)
    ax1.set_title('Performance on Each Video Type Across All Models', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_list, fontsize=11)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== PLOT 2: Total Cumulative Error (Bottom Left) =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    total_errors = [sum(final_performance[model]) for model in model_list]
    winner_idx = np.argmin(total_errors)
    
    colors = ['#2ecc71' if i == winner_idx else '#e74c3c' if i == np.argmax(total_errors) else '#3498db' 
              for i in range(len(model_list))]
    
    bars = ax2.bar(range(len(model_list)), total_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Total Cumulative Error', fontweight='bold', fontsize=11)
    ax2.set_title('Overall Score (All Videos Combined)', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(model_list)))
    ax2.set_xticklabels(model_list, rotation=15, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, total_errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== PLOT 3: Forgetting Analysis (Bottom Right) =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    forgetting_data = []
    model_names_short = []
    
    for model_name, model in models.items():
        # Get V1 error at the end
        forgetting = final_performance[model_name][0]
        forgetting_data.append(forgetting)
        
        # Shorten name
        if 'Hebbian' in model_name and 'Dynamic' in model_name:
            model_names_short.append('Hebb\n+Dyn')
        elif 'Hebbian' in model_name and 'Static' in model_name:
            model_names_short.append('Hebb\n+Stat')
        elif 'Backprop' in model_name and 'Dynamic' in model_name:
            model_names_short.append('BP\n+Dyn')
        else:
            model_names_short.append('BP\n+Stat')
    
    colors_forget = ['#2ecc71' if e == min(forgetting_data) else '#e74c3c' 
                     for e in forgetting_data]
    
    bars_forget = ax3.bar(range(len(forgetting_data)), forgetting_data, 
                          color=colors_forget, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Video 1 Error (After Training All)', fontweight='bold', fontsize=11)
    ax3.set_title('Memory Retention on First Task', fontweight='bold', fontsize=12)
    ax3.set_xticks(range(len(forgetting_data)))
    ax3.set_xticklabels(model_names_short, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_forget, forgetting_data)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('COMPREHENSIVE CONTINUAL LEARNING TEST\nHebbian vs Backprop | Dynamic vs Static | 3 Different Tasks', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('comprehensive_comparison.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: comprehensive_comparison.png")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)
    
    return results, final_performance


if __name__ == "__main__":
    results, final_perf = comprehensive_test()