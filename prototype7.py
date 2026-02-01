import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# ============================================================================
# EFFICIENCY TRACKING
# ============================================================================

class EfficiencyTracker:
    """Track computational metrics during training."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    def end(self, epoch):
        elapsed = time.time() - self.start_time
        self.metrics['time'].append(elapsed)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.metrics['memory_mb'].append(peak_memory)
    
    def count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics['total_params'] = total
        return total
    
    def get_flops_estimate(self, model_type, input_size, batch_size):
        """Estimate FLOPs for different model types."""
        if model_type == 'pixel_based':
            # Pixel: Conv layers + dense layers
            # Rough estimate: 2 * input_size * hidden_size
            return 2 * input_size * batch_size * 256
        else:  # situation_based
            # Situation: Much smaller encoder + MLP predictor
            # Rough estimate: smaller by factor of 10
            return 2 * 64 * batch_size * 128
    
    def summary(self):
        """Return efficiency metrics summary."""
        return {
            'total_time': sum(self.metrics['time']),
            'avg_time_per_epoch': np.mean(self.metrics['time']),
            'total_params': self.metrics['total_params'],
            'peak_memory_mb': max(self.metrics['memory_mb']) if self.metrics['memory_mb'] else 0,
        }


# ============================================================================
# PIXEL-BASED BASELINE (for comparison)
# ============================================================================

class PixelBasedPredictor(nn.Module):
    """Standard pixel-to-pixel prediction (token-like approach)."""
    
    def __init__(self):
        super().__init__()
        
        # Direct pixel processing
        self.encode_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        self.decode_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        encoded = self.encode_conv(frame)
        decoded = self.decode_conv(encoded)
        return decoded
    
    def update(self, frame_current, frame_target):
        self.optimizer.zero_grad()
        prediction = self.forward(frame_current)
        loss = self.criterion(prediction, frame_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# SITUATION-BASED (from before)
# ============================================================================

class SituationEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.spatial_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        self.spatial_compress = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )
    
    def forward(self, frame):
        spatial = self.spatial_features(frame)
        spatial_flat = spatial.view(spatial.size(0), -1)
        situation = self.spatial_compress(spatial_flat)
        return situation


class SituationPredictor(nn.Module):
    def __init__(self, situation_dim=64, hidden_dim=128):
        super().__init__()
        self.situation_dim = situation_dim
        
        self.dynamics = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, situation_dim),
        )
    
    def forward(self, situation_current):
        next_situation = self.dynamics(situation_current)
        return next_situation


class SituationDecoder(nn.Module):
    def __init__(self, situation_dim=64):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(situation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU(),
        )
        
        self.spatial_decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, situation):
        x = self.decode(situation)
        x = x.view(x.size(0), 128, 4, 4)
        frame = self.spatial_decode(x)
        return frame


class SituationBasedPredictor(nn.Module):
    def __init__(self, situation_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=situation_dim)
        self.predictor = SituationPredictor(situation_dim=situation_dim, hidden_dim=hidden_dim)
        self.decoder = SituationDecoder(situation_dim=situation_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        situation_current = self.encoder(frame)
        situation_next = self.predictor(situation_current)
        reconstructed = self.decoder(situation_next)
        return reconstructed, situation_current
    
    def update(self, frame_current, frame_target):
        self.optimizer.zero_grad()
        
        situation_current = self.encoder(frame_current)
        situation_pred = self.predictor(situation_current)
        situation_target = self.encoder(frame_target)
        
        loss = self.criterion(situation_pred, situation_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def generate_video_type_1(num_frames=80, size=32, velocity_scale=1.0, seed=42):
    """Single bouncing ball."""
    np.random.seed(seed)
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


# ============================================================================
# GENERALIZATION TEST
# ============================================================================

def test_generalization():
    """
    Train on video 1, test on:
    1. Same video (memorization)
    2. Faster version (generalization)
    3. Slower version (generalization)
    """
    
    print("=" * 100)
    print("COMPUTATIONAL EFFICIENCY & GENERALIZATION TEST")
    print("Situation-Based vs Pixel-Based Learning")
    print("=" * 100)
    
    # Generate training and test videos
    print("\n1. Generating videos...")
    train_video = generate_video_type_1(num_frames=80, velocity_scale=1.0, seed=42)
    test_same = generate_video_type_1(num_frames=80, velocity_scale=1.0, seed=100)  # Different seed, same speed
    test_faster = generate_video_type_1(num_frames=80, velocity_scale=2.0, seed=42)
    test_slower = generate_video_type_1(num_frames=80, velocity_scale=0.5, seed=42)
    print(f"   ✓ Training video (normal speed)")
    print(f"   ✓ Test: Same speed")
    print(f"   ✓ Test: 2x faster")
    print(f"   ✓ Test: 0.5x slower")
    
    # Create models
    print("\n2. Creating models...")
    models = {
        'Pixel-Based': PixelBasedPredictor(),
        'Situation-Based': SituationBasedPredictor(),
    }
    
    # Track efficiency
    trackers = {name: EfficiencyTracker() for name in models.keys()}
    for name, model in models.items():
        trackers[name].count_parameters(model)
        print(f"   ✓ {name}: {trackers[name].metrics['total_params']:,} parameters")
    
    # Train models
    print("\n3. Training models (30 epochs)...")
    print("-" * 100)
    
    training_results = {}
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        tracker = trackers[model_name]
        
        losses = []
        for epoch in range(30):
            tracker.start()
            
            epoch_loss = 0
            num_batches = 0
            
            for frame_idx in range(len(train_video) - 1):
                current = train_video[frame_idx].unsqueeze(0).unsqueeze(0)
                target = train_video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                loss = model.update(current, target)
                epoch_loss += loss
                num_batches += 1
            
            tracker.end(epoch)
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
        
        training_results[model_name] = losses
    
    # Test generalization
    print("\n\n4. Testing generalization...")
    print("-" * 100)
    
    generalization_results = {name: {} for name in models.keys()}
    
    test_videos = {
        'Same Speed': test_same,
        '2x Faster': test_faster,
        '0.5x Slower': test_slower,
    }
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for test_name, test_video in test_videos.items():
            total_error = 0
            
            with torch.no_grad():
                for frame_idx in range(len(test_video) - 1):
                    current = test_video[frame_idx].unsqueeze(0).unsqueeze(0)
                    target = test_video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                    
                    if model_name == 'Pixel-Based':
                        prediction = model(current)
                    else:
                        prediction, _ = model(current)
                    
                    error = torch.mean((prediction - target) ** 2)
                    total_error += error.item()
            
            avg_error = total_error / (len(test_video) - 1)
            generalization_results[model_name][test_name] = avg_error
            print(f"     {test_name}: {avg_error:.6f}")
    
    # Get efficiency summaries
    efficiency_summaries = {name: trackers[name].summary() for name in models.keys()}
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n\n5. Creating comprehensive analysis...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Plot 1: Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name, losses in training_results.items():
        ax1.plot(losses, label=model_name, linewidth=2, marker='o', markersize=3)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_title('Training Convergence', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model parameters
    ax2 = fig.add_subplot(gs[0, 1])
    model_names = list(models.keys())
    params = [efficiency_summaries[name]['total_params'] for name in model_names]
    colors = ['#3498db', '#2ecc71']
    bars = ax2.bar(model_names, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Parameters', fontweight='bold', fontsize=11)
    ax2.set_title('Model Complexity', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Generalization performance
    ax3 = fig.add_subplot(gs[1, 0])
    test_names = list(test_videos.keys())
    x = np.arange(len(test_names))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        errors = [generalization_results[model_name][test_name] for test_name in test_names]
        ax3.bar(x + (i - 0.5) * width, errors, width, label=model_name, alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax3.set_title('Generalization to Different Speeds', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training efficiency (time)
    ax4 = fig.add_subplot(gs[1, 1])
    times = [efficiency_summaries[name]['total_time'] for name in model_names]
    bars = ax4.bar(model_names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Total Training Time (seconds)', fontweight='bold', fontsize=11)
    ax4.set_title('Training Speed', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 5: Efficiency score (combined metrics)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Normalize metrics for comparison
    norm_params = np.array([efficiency_summaries[name]['total_params'] for name in model_names])
    norm_params = norm_params / norm_params.max()
    
    norm_time = np.array(times) / max(times)
    
    # Average generalization error
    avg_gen_error = np.array([
        np.mean(list(generalization_results[name].values())) for name in model_names
    ])
    avg_gen_error = avg_gen_error / avg_gen_error.max()
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax5.bar(x - width, norm_params, width, label='Normalized Parameters', alpha=0.8, edgecolor='black')
    ax5.bar(x, norm_time, width, label='Normalized Training Time', alpha=0.8, edgecolor='black')
    ax5.bar(x + width, avg_gen_error, width, label='Normalized Gen. Error', alpha=0.8, edgecolor='black')
    
    ax5.set_ylabel('Normalized Score (Lower = Better)', fontweight='bold', fontsize=11)
    ax5.set_title('Efficiency Comparison (All Metrics Normalized)', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_names)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1.1])
    
    plt.suptitle('COMPUTATIONAL EFFICIENCY & GENERALIZATION ANALYSIS\nSituation-Based vs Pixel-Based Learning',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('efficiency_generalization_analysis.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: efficiency_generalization_analysis.png")
    
    # Print detailed results
    print("\n\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    
    print("\n📊 MODEL EFFICIENCY:")
    for model_name in model_names:
        summary = efficiency_summaries[model_name]
        print(f"\n   {model_name}:")
        print(f"     - Parameters: {summary['total_params']:,}")
        print(f"     - Training time: {summary['total_time']:.2f}s")
        print(f"     - Avg time/epoch: {summary['avg_time_per_epoch']:.4f}s")
    
    print("\n🎯 GENERALIZATION RESULTS:")
    for model_name in model_names:
        print(f"\n   {model_name}:")
        for test_name, error in generalization_results[model_name].items():
            print(f"     - {test_name}: {error:.6f}")
    
    print("\n💡 KEY INSIGHTS:")
    
    # Compare
    param_reduction = (1 - efficiency_summaries['Situation-Based']['total_params'] / 
                       efficiency_summaries['Pixel-Based']['total_params']) * 100
    time_reduction = (1 - efficiency_summaries['Situation-Based']['total_time'] / 
                      efficiency_summaries['Pixel-Based']['total_time']) * 100
    
    print(f"   • Situation-based uses {param_reduction:.1f}% fewer parameters")
    print(f"   • Situation-based is {time_reduction:.1f}% faster to train")
    
    pixel_gen_errors = list(generalization_results['Pixel-Based'].values())
    situation_gen_errors = list(generalization_results['Situation-Based'].values())
    
    pixel_avg = np.mean(pixel_gen_errors)
    situation_avg = np.mean(situation_gen_errors)
    
    gen_improvement = (1 - situation_avg / pixel_avg) * 100
    print(f"   • Situation-based generalizes {gen_improvement:.1f}% better on average")
    
    print("\n" + "=" * 100)
    
    return efficiency_summaries, generalization_results, training_results


if __name__ == "__main__":
    efficiency_summaries, generalization_results, training_results = test_generalization()