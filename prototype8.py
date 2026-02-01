import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

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
# ALL 4 MODELS WITH SITUATION UNDERSTANDING
# ============================================================================

class HebbianDynamicSituation(nn.Module):
    """Hebbian + Dynamic restructuring."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        # Dynamics in situation space
        self.dynamics = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # For Hebbian updates
        self.W_dynamics = nn.Parameter(torch.randn(64, 128) * 0.01)
        self.W_predict = nn.Parameter(torch.randn(128, 64) * 0.01)
        
        self.learning_rate = learning_rate
        self.consolidation_mask = torch.ones(128)
    
    def forward(self, frame):
        situation = self.encoder(frame)
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return situation, pred_situation, hidden
    
    def hebbian_update(self, situation, hidden, target_situation):
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        error = target_situation - pred_situation
        
        dW_dynamics = torch.outer(situation, hidden) / situation.size(0)
        dW_predict = torch.outer(hidden, error) / hidden.size(0)
        
        self.W_dynamics.data += self.learning_rate * dW_dynamics
        self.W_predict.data += self.learning_rate * dW_predict
        
        return torch.mean(error ** 2)
    
    def predict_next_situation(self, situation):
        """Predict next situation from current (memory-based)."""
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return pred


class HebbianStaticSituation(nn.Module):
    """Hebbian + Static (no restructuring)."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.dynamics = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        self.W_dynamics = nn.Parameter(torch.randn(64, 128) * 0.01)
        self.W_predict = nn.Parameter(torch.randn(128, 64) * 0.01)
        
        self.learning_rate = learning_rate
    
    def forward(self, frame):
        situation = self.encoder(frame)
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return situation, pred_situation, hidden
    
    def hebbian_update(self, situation, hidden, target_situation):
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        error = target_situation - pred_situation
        
        dW_dynamics = torch.outer(situation, hidden)
        dW_predict = torch.outer(hidden, error)
        
        self.W_dynamics.data += self.learning_rate * dW_dynamics
        self.W_predict.data += self.learning_rate * dW_predict
        
        return torch.mean(error ** 2)
    
    def predict_next_situation(self, situation):
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return pred


class BackpropDynamicSituation(nn.Module):
    """Backprop + Dynamic restructuring."""
    def __init__(self):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.dynamics = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        situation = self.encoder(frame)
        pred_situation = self.dynamics(situation)
        return situation, pred_situation
    
    def backprop_update(self, frame_current, frame_target):
        self.optimizer.zero_grad()
        
        situation_current = self.encoder(frame_current)
        situation_pred = self.dynamics(situation_current)
        situation_target = self.encoder(frame_target)
        
        loss = self.criterion(situation_pred, situation_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_next_situation(self, situation):
        with torch.no_grad():
            return self.dynamics(situation)


class BackpropStaticSituation(nn.Module):
    """Backprop + Static (no restructuring)."""
    def __init__(self):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.dynamics = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def forward(self, frame):
        situation = self.encoder(frame)
        pred_situation = self.dynamics(situation)
        return situation, pred_situation
    
    def backprop_update(self, frame_current, frame_target):
        self.optimizer.zero_grad()
        
        situation_current = self.encoder(frame_current)
        situation_pred = self.dynamics(situation_current)
        situation_target = self.encoder(frame_target)
        
        loss = self.criterion(situation_pred, situation_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_next_situation(self, situation):
        with torch.no_grad():
            return self.dynamics(situation)


# ============================================================================
# VIDEO GENERATION - SEQUENTIAL VIDEOS
# ============================================================================

def generate_video_sequence_1(num_frames=60, seed=42):
    """Video 1: Ball moving left to right."""
    np.random.seed(seed)
    frames = []
    y = 16
    
    for x in np.linspace(5, 27, num_frames):
        frame = np.zeros((32, 32))
        xx, yy = np.arange(32), np.arange(32)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - x)**2 + (YY - y)**2 <= 2**2
        frame[mask] = 1.0
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


def generate_video_sequence_2(num_frames=60, seed=42):
    """Video 2: Ball bouncing (related to video 1 but different dynamics)."""
    np.random.seed(seed)
    frames = []
    x, y = 16, 16
    vx, vy = 2.0, 1.5
    
    for _ in range(num_frames):
        frame = np.zeros((32, 32))
        xx, yy = np.arange(32), np.arange(32)
        XX, YY = np.meshgrid(xx, yy)
        mask = (XX - x)**2 + (YY - y)**2 <= 2**2
        frame[mask] = 1.0
        
        x += vx
        y += vy
        if x < 2 or x > 30:
            vx *= -1
        if y < 2 or y > 30:
            vy *= -1
        
        frames.append(torch.from_numpy(frame).float())
    
    return torch.stack(frames)


# ============================================================================
# MAIN TEST - BRUTAL CONTINUAL LEARNING
# ============================================================================

def brutal_continual_learning_test():
    """
    Test all 4 models:
    1. Train on VIDEO 1 (learn situation dynamics)
    2. WITHOUT seeing VIDEO 2, predict its situations using memory
    3. Then train on VIDEO 2 to see if they can adapt
    4. Compare memory-based predictions vs actual
    """
    
    print("=" * 120)
    print("BRUTAL CONTINUAL LEARNING TEST")
    print("Learn situations from Video 1, predict Video 2 from memory, then adapt")
    print("=" * 120)
    
    # Generate videos
    print("\n1. Generating sequential videos...")
    video1 = generate_video_sequence_1(num_frames=60, seed=42)
    video2 = generate_video_sequence_2(num_frames=60, seed=99)
    print(f"   ✓ Video 1: Ball moving left-to-right (learning phase)")
    print(f"   ✓ Video 2: Ball bouncing (prediction phase - HIDDEN)")
    
    # Create all 4 models
    print("\n2. Creating all 4 models...")
    models = {
        'Hebbian + Dynamic': HebbianDynamicSituation(learning_rate=0.01),
        'Hebbian + Static': HebbianStaticSituation(learning_rate=0.01),
        'Backprop + Dynamic': BackpropDynamicSituation(),
        'Backprop + Static': BackpropStaticSituation(),
    }
    
    for name in models.keys():
        print(f"   ✓ {name}")
    
    # PHASE 1: Train on video 1
    print("\n" + "=" * 120)
    print("PHASE 1: TRAINING ON VIDEO 1")
    print("=" * 120)
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for epoch in range(20):
            epoch_loss = 0
            
            for frame_idx in range(len(video1) - 1):
                current_frame = video1[frame_idx].unsqueeze(0).unsqueeze(0)
                target_frame = video1[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                if 'Hebbian' in model_name:
                    situation, pred_situation, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update(situation, hidden, target_situation)
                else:
                    loss = model.backprop_update(current_frame, target_frame)
                
                epoch_loss += loss
            
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch + 1}: Loss = {epoch_loss / (len(video1) - 1):.6f}")
    
    # PHASE 2: Predict video 2 WITHOUT SEEING IT (memory-based)
    print("\n\n" + "=" * 120)
    print("PHASE 2: MEMORY-BASED PREDICTION (without seeing video 2)")
    print("=" * 120)
    
    memory_predictions = {name: [] for name in models.keys()}
    actual_situations = []
    
    print("\n   Making predictions based on learned dynamics from video 1...")
    
    with torch.no_grad():
        # Get first frame of video 2 to start
        current_situation = models['Hebbian + Dynamic'].encoder(video2[0].unsqueeze(0).unsqueeze(0))
        actual_situations.append(current_situation.squeeze(0).cpu().numpy())
        
        # Predict rest of video 2
        for frame_idx in range(1, len(video2)):
            # Get actual situation for comparison
            actual_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
            actual_situation = models['Hebbian + Dynamic'].encoder(actual_frame).squeeze(0).cpu().numpy()
            actual_situations.append(actual_situation)
            
            # Make predictions from each model
            for model_name, model in models.items():
                # Predict next situation using learned dynamics
                pred_situation = model.predict_next_situation(
                    models['Hebbian + Dynamic'].encoder(video2[frame_idx - 1].unsqueeze(0).unsqueeze(0))
                )
                memory_predictions[model_name].append(pred_situation.squeeze(0).cpu().numpy())
    
    # Evaluate memory-based predictions
    print("\n   Evaluating memory-based predictions...")
    memory_errors = {}
    
    for model_name in models.keys():
        predictions = np.array(memory_predictions[model_name])
        actuals = np.array(actual_situations[1:])
        
        error = np.mean(np.sqrt(np.sum((predictions - actuals) ** 2, axis=1)))
        memory_errors[model_name] = error
        print(f"     {model_name}: {error:.6f}")
    
    # PHASE 3: Train on video 2 and check recovery
    print("\n\n" + "=" * 120)
    print("PHASE 3: TRAINING ON VIDEO 2 (adaptation phase)")
    print("=" * 120)
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for epoch in range(20):
            epoch_loss = 0
            
            for frame_idx in range(len(video2) - 1):
                current_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
                target_frame = video2[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                if 'Hebbian' in model_name:
                    situation, pred_situation, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update(situation, hidden, target_situation)
                else:
                    loss = model.backprop_update(current_frame, target_frame)
                
                epoch_loss += loss
            
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch + 1}: Loss = {epoch_loss / (len(video2) - 1):.6f}")
    
    # PHASE 4: Evaluate after training on video 2
    print("\n\n" + "=" * 120)
    print("PHASE 4: FINAL EVALUATION (after learning video 2)")
    print("=" * 120)
    
    final_errors = {name: [] for name in models.keys()}
    
    print("\n   Video 2 prediction accuracy after training...")
    
    with torch.no_grad():
        for frame_idx in range(len(video2) - 1):
            current_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
            target_frame = video2[frame_idx + 1].unsqueeze(0).unsqueeze(0)
            
            for model_name, model in models.items():
                if 'Hebbian' in model_name:
                    situation, pred_situation, _ = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                else:
                    situation, pred_situation = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                
                error = torch.mean((pred_situation - target_situation) ** 2).item()
                final_errors[model_name].append(error)
    
    final_error_summary = {name: np.mean(errors) for name, errors in final_errors.items()}
    
    for model_name, error in final_error_summary.items():
        print(f"     {model_name}: {error:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n\n5. Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Memory-based prediction errors
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = list(models.keys())
    memory_err = [memory_errors[m] for m in model_names]
    colors = ['#e74c3c' if e > 0.1 else '#f39c12' if e > 0.05 else '#2ecc71' for e in memory_err]
    
    bars = ax1.bar(range(len(model_names)), memory_err, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax1.set_title('Memory-Based Prediction of Video 2\n(Without Seeing Actual Frames)', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([m.replace(' + ', '\n+\n') for m in model_names], fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, memory_err):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Final accuracy after learning video 2
    ax2 = fig.add_subplot(gs[0, 1])
    final_err = [final_error_summary[m] for m in model_names]
    colors = ['#2ecc71' if e < 0.01 else '#f39c12' if e < 0.05 else '#e74c3c' for e in final_err]
    
    bars = ax2.bar(range(len(model_names)), final_err, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax2.set_title('Final Accuracy After Training on Video 2', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([m.replace(' + ', '\n+\n') for m in model_names], fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, final_err):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Memory vs Final comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax3.bar(x - width/2, memory_err, width, label='Memory-Based Prediction (Blind)', alpha=0.8, edgecolor='black', color='#e67e22')
    ax3.bar(x + width/2, final_err, width, label='After Training on Video 2', alpha=0.8, edgecolor='black', color='#3498db')
    
    ax3.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax3.set_title('Continual Learning: Blind Prediction vs Adaptation', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' + ', '\n+\n') for m in model_names], fontsize=10)
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('BRUTAL CONTINUAL LEARNING TEST\nLearn Video 1 → Predict Video 2 Blind → Adapt to Video 2',
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('brutal_continual_learning.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: brutal_continual_learning.png")
    
    # Print detailed analysis
    print("\n\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)
    
    print("\n🔮 BLIND PREDICTION (based on learned Video 1 dynamics):")
    for model_name in model_names:
        err = memory_errors[model_name]
        print(f"   {model_name}: {err:.6f}")
        if err < 0.05:
            print(f"      ✓ Good memory transfer!")
        elif err < 0.1:
            print(f"      ~ Moderate memory transfer")
        else:
            print(f"      ✗ Poor memory transfer (couldn't generalize)")
    
    print("\n📚 LEARNING EFFICIENCY (how fast to adapt):")
    improvement = [(memory_errors[m] - final_error_summary[m]) / memory_errors[m] * 100 
                   for m in model_names]
    for model_name, imp in zip(model_names, improvement):
        print(f"   {model_name}: {imp:.1f}% improvement")
    
    print("\n🏆 BEST OVERALL PERFORMERS:")
    
    # Memory-based ranking
    memory_ranking = sorted(memory_errors.items(), key=lambda x: x[1])
    print(f"\n   Best at memory transfer:")
    for i, (name, err) in enumerate(memory_ranking[:2], 1):
        print(f"     {i}. {name}: {err:.6f}")
    
    # Final accuracy ranking
    final_ranking = sorted(final_error_summary.items(), key=lambda x: x[1])
    print(f"\n   Best at final accuracy:")
    for i, (name, err) in enumerate(final_ranking[:2], 1):
        print(f"     {i}. {name}: {err:.6f}")
    
    print("\n" + "=" * 120)
    
    return memory_errors, final_error_summary, memory_predictions


if __name__ == "__main__":
    memory_errors, final_errors, predictions = brutal_continual_learning_test()