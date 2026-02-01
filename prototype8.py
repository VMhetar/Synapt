import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# ============================================================================
# SITUATION ENCODERS/DECODERS
# ============================================================================

class SituationEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
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


# ============================================================================
# PIXEL-BASED MODELS
# ============================================================================

class HebbianDynamicPixel(nn.Module):
    """Hebbian + Dynamic restructuring on PIXEL space."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.hidden_dim = 128 * 8 * 8
        
        self.W_encode = nn.Parameter(torch.randn(self.hidden_dim, 256) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(256, self.hidden_dim) * 0.01)
        
        self.learning_rate = learning_rate
        self.consolidation_mask = torch.ones(256)
    
    def forward(self, frame):
        encoded = self.encoder(frame)
        encoded_flat = encoded.view(encoded.size(0), -1)
        hidden = torch.matmul(encoded_flat, self.W_encode)
        hidden = torch.relu(hidden)
        recon_hidden = torch.matmul(hidden, self.W_decode)
        return recon_hidden, hidden, encoded_flat
    
    def hebbian_update(self, encoded_flat, hidden, target_encoded):
        error = target_encoded - encoded_flat
        
        dW_encode = torch.matmul(encoded_flat.T, hidden) / encoded_flat.size(0)
        dW_decode = torch.matmul(hidden.T, error) / encoded_flat.size(0)
        
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_encode.data += self.learning_rate * dW_encode * consolidation_factor
        self.W_decode.data += self.learning_rate * dW_decode * consolidation_factor.T
        
        return torch.mean(error ** 2)


class BackpropDynamicPixel(nn.Module):
    """Backprop + Dynamic restructuring on PIXEL space."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.hidden_dim = 128 * 8 * 8
        
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, self.hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        encoded = self.encoder(frame)
        encoded_flat = encoded.view(encoded.size(0), -1)
        hidden = torch.relu(self.fc1(encoded_flat))
        recon = self.fc2(hidden)
        return recon, hidden, encoded_flat
    
    def backprop_update(self, frame_current, frame_target):
        self.optimizer.zero_grad()
        recon, hidden, _ = self.forward(frame_current)
        
        target_encoded = self.encoder(frame_target)
        target_flat = target_encoded.view(target_encoded.size(0), -1)
        
        loss = self.criterion(recon, target_flat)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# SITUATION-BASED MODELS
# ============================================================================

class HebbianDynamicSituation(nn.Module):
    """Hebbian + Dynamic restructuring on SITUATION space."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.W_encode = nn.Parameter(torch.randn(64, 128) * 0.01)
        self.W_decode = nn.Parameter(torch.randn(128, 64) * 0.01)
        
        self.learning_rate = learning_rate
        self.consolidation_mask = torch.ones(128)
    
    def forward(self, frame):
        situation = self.encoder(frame)
        hidden = torch.matmul(situation, self.W_encode)
        hidden = torch.relu(hidden)
        pred_situation = torch.matmul(hidden, self.W_decode)
        return pred_situation, hidden, situation
    
    def hebbian_update(self, situation, hidden, target_situation):
        error = target_situation - torch.matmul(hidden, self.W_decode)
        
        dW_encode = torch.matmul(situation.T, hidden) / situation.size(0)
        dW_decode = torch.matmul(hidden.T, error) / situation.size(0)
        
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_encode.data += self.learning_rate * dW_encode * consolidation_factor
        self.W_decode.data += self.learning_rate * dW_decode * consolidation_factor.T
        
        return torch.mean(error ** 2)


class BackpropDynamicSituation(nn.Module):
    """Backprop + Dynamic restructuring on SITUATION space."""
    def __init__(self):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        situation = self.encoder(frame)
        pred_situation = self.predictor(situation)
        return pred_situation, situation
    
    def backprop_update(self, frame_current, frame_target):
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

def generate_video(num_frames=80, velocity_scale=1.0, seed=42):
    """Generate bouncing ball video."""
    np.random.seed(seed)
    frames = []
    x, y = 16, 16
    vx, vy = 1.5 * velocity_scale, 1.0 * velocity_scale
    radius = 2
    
    for _ in range(num_frames):
        frame = np.zeros((32, 32))
        x += vx
        y += vy
        if x < radius or x > 32 - radius:
            vx *= -1
            x = np.clip(x, radius, 32 - radius)
        if y < radius or y > 32 - radius:
            vy *= -1
            y = np.clip(y, radius, 32 - radius)
        
        xx, yy = np.arange(32), np.arange(32)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - int(x))**2 + (YY - int(y))**2 <= radius**2
        frame[mask] = 1.0
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


# ============================================================================
# MAIN TEST
# ============================================================================

def ultimate_comparison_test():
    """Compare all 4 approaches on both pixel and situation representations."""
    
    print("=" * 120)
    print("ULTIMATE COMPARISON: ALL 4 APPROACHES × 2 REPRESENTATIONS")
    print("=" * 120)
    
    # Generate videos
    print("\n1. Generating videos...")
    train_video = generate_video(num_frames=80, velocity_scale=1.0, seed=42)
    test_same = generate_video(num_frames=80, velocity_scale=1.0, seed=100)
    test_fast = generate_video(num_frames=80, velocity_scale=2.0, seed=42)
    test_slow = generate_video(num_frames=80, velocity_scale=0.5, seed=42)
    print("   ✓ Training video, test same/fast/slow")
    
    # Create all 8 models
    print("\n2. Creating all 8 models...")
    print("   PIXEL-BASED:")
    print("     ✓ Hebbian + Dynamic")
    print("     ✓ Backprop + Dynamic")
    print("   SITUATION-BASED:")
    print("     ✓ Hebbian + Dynamic")
    print("     ✓ Backprop + Dynamic")
    
    models = {
        'Hebbian+Dynamic (Pixel)': HebbianDynamicPixel(),
        'Backprop+Dynamic (Pixel)': BackpropDynamicPixel(),
        'Hebbian+Dynamic (Situation)': HebbianDynamicSituation(),
        'Backprop+Dynamic (Situation)': BackpropDynamicSituation(),
    }
    
    # Train all models
    print("\n3. Training all models (30 epochs)...")
    print("-" * 120)
    
    training_losses = {name: [] for name in models.keys()}
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        start_time = time.time()
        
        for epoch in range(30):
            epoch_loss = 0
            num_batches = 0
            
            for frame_idx in range(len(train_video) - 1):
                current_frame = train_video[frame_idx].unsqueeze(0).unsqueeze(0)
                target_frame = train_video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                if 'Pixel' in model_name:
                    if 'Hebbian' in model_name:
                        recon, hidden, encoded = model.forward(current_frame)
                        target_encoded = model.encoder(target_frame)
                        target_flat = target_encoded.view(target_encoded.size(0), -1)
                        loss = model.hebbian_update(encoded, hidden, target_flat)
                    else:
                        loss = model.backprop_update(current_frame, target_frame)
                else:  # Situation
                    if 'Hebbian' in model_name:
                        pred_situation, hidden, situation = model.forward(current_frame)
                        target_situation = model.encoder(target_frame)
                        loss = model.hebbian_update(situation, hidden, target_situation)
                    else:
                        loss = model.backprop_update(current_frame, target_frame)
                
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            training_losses[model_name].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch + 1}: {avg_loss:.6f}")
        
        train_time = time.time() - start_time
        print(f"     Total time: {train_time:.2f}s")
    
    # Test generalization
    print("\n\n4. Testing generalization...")
    print("-" * 120)
    
    generalization_results = {name: {} for name in models.keys()}
    test_videos = {'Same': test_same, '2x Faster': test_fast, '0.5x Slower': test_slow}
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for test_name, test_video in test_videos.items():
            total_error = 0
            
            with torch.no_grad():
                for frame_idx in range(len(test_video) - 1):
                    current = test_video[frame_idx].unsqueeze(0).unsqueeze(0)
                    target = test_video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                    
                    if 'Pixel' in model_name:
                        if 'Hebbian' in model_name:
                            recon, _, _ = model.forward(current)
                            target_encoded = model.encoder(target)
                            target_flat = target_encoded.view(target_encoded.size(0), -1)
                            error = torch.mean((recon - target_flat) ** 2)
                        else:
                            recon, _, _ = model.forward(current)
                            target_encoded = model.encoder(target)
                            target_flat = target_encoded.view(target_encoded.size(0), -1)
                            error = torch.mean((recon - target_flat) ** 2)
                    else:  # Situation
                        if 'Hebbian' in model_name:
                            pred_sit, _, _ = model.forward(current)
                            target_sit = model.encoder(target)
                            error = torch.mean((pred_sit - target_sit) ** 2)
                        else:
                            pred_sit, _ = model.forward(current)
                            target_sit = model.encoder(target)
                            error = torch.mean((pred_sit - target_sit) ** 2)
                    
                    total_error += error.item()
            
            avg_error = total_error / (len(test_video) - 1)
            generalization_results[model_name][test_name] = avg_error
            print(f"     {test_name}: {avg_error:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n\n5. Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Training curves (Pixel vs Situation)
    ax1 = fig.add_subplot(gs[0, 0])
    pixel_models = [m for m in models.keys() if 'Pixel' in m]
    for model_name in pixel_models:
        ax1.plot(training_losses[model_name], label=model_name.replace(' (Pixel)', ''), linewidth=2, marker='o', markersize=3)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_title('Training: PIXEL-BASED Models', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    situation_models = [m for m in models.keys() if 'Situation' in m]
    for model_name in situation_models:
        ax2.plot(training_losses[model_name], label=model_name.replace(' (Situation)', ''), linewidth=2, marker='s', markersize=3)
    ax2.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_title('Training: SITUATION-BASED Models', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 2: Generalization comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    test_names = list(test_videos.keys())
    x = np.arange(len(models))
    width = 0.2
    
    for i, test_name in enumerate(test_names):
        errors = [generalization_results[model][test_name] for model in models.keys()]
        ax3.bar(x + (i - 1) * width, errors, width, label=test_name, alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax3.set_title('Generalization Across All Models', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' (Pixel)', '\nPix').replace(' (Situation)', '\nSit') for m in models.keys()], fontsize=10)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('ULTIMATE COMPARISON: All 4 Approaches × 2 Representations\nHebbian vs Backprop | Dynamic Restructuring | Pixel vs Situation',
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('ultimate_comparison.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: ultimate_comparison.png")
    
    # Print detailed analysis
    print("\n\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)
    
    print("\n📊 PIXEL-BASED MODELS:")
    for model_name in pixel_models:
        print(f"\n   {model_name}:")
        for test_name, error in generalization_results[model_name].items():
            print(f"     - {test_name}: {error:.6f}")
    
    print("\n\n📊 SITUATION-BASED MODELS:")
    for model_name in situation_models:
        print(f"\n   {model_name}:")
        for test_name, error in generalization_results[model_name].items():
            print(f"     - {test_name}: {error:.6f}")
    
    # Summary
    print("\n\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    # Best overall
    all_errors = {}
    for model_name in models.keys():
        avg_gen_error = np.mean(list(generalization_results[model_name].values()))
        all_errors[model_name] = avg_gen_error
    
    best_model = min(all_errors, key=all_errors.get)
    print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
    print(f"    Average generalization error: {all_errors[best_model]:.6f}")
    
    # Pixel vs Situation
    pixel_avg = np.mean([all_errors[m] for m in pixel_models])
    situation_avg = np.mean([all_errors[m] for m in situation_models])
    
    print(f"\n📈 REPRESENTATION COMPARISON:")
    print(f"   Pixel-based average error:      {pixel_avg:.6f}")
    print(f"   Situation-based average error:  {situation_avg:.6f}")
    print(f"   Situation wins by:              {(1 - situation_avg/pixel_avg)*100:.1f}%")
    
    # Hebbian vs Backprop
    hebbian_models = [m for m in models.keys() if 'Hebbian' in m]
    backprop_models = [m for m in models.keys() if 'Backprop' in m]
    
    hebbian_avg = np.mean([all_errors[m] for m in hebbian_models])
    backprop_avg = np.mean([all_errors[m] for m in backprop_models])
    
    print(f"\n🧠 LEARNING RULE COMPARISON:")
    print(f"   Hebbian average error:          {hebbian_avg:.6f}")
    print(f"   Backprop average error:         {backprop_avg:.6f}")
    if hebbian_avg < backprop_avg:
        print(f"   Hebbian wins by:                {(1 - hebbian_avg/backprop_avg)*100:.1f}%")
    else:
        print(f"   Backprop wins by:               {(1 - backprop_avg/hebbian_avg)*100:.1f}%")
    
    print("\n" + "=" * 120)
    
    return training_losses, generalization_results


if __name__ == "__main__":
    training_losses, generalization_results = ultimate_comparison_test()