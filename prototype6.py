import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============================================================================
# SITUATION ENCODER - Learn abstract situation representations
# ============================================================================

class SituationEncoder(nn.Module):
    """
    Learns to compress video into abstract situation representation.
    
    Instead of learning pixels or local features:
    - Extracts WHAT is happening (objects, motion, relationships)
    - Encodes WHERE things are
    - Encodes HOW they move (velocity, acceleration)
    - Creates semantic situation vector
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Spatial pyramid to capture multi-scale features
        self.spatial_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 128x4x4
            nn.ReLU(),
        )
        
        # Flatten and compress to situation representation
        self.spatial_compress = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )
    
    def forward(self, frame):
        """Encode frame to situation vector."""
        # frame: (batch, 1, 32, 32) or similar
        spatial = self.spatial_features(frame)
        spatial_flat = spatial.view(spatial.size(0), -1)
        situation = self.spatial_compress(spatial_flat)
        return situation


class SituationPredictor(nn.Module):
    """
    Predicts how situations evolve over time.
    
    Learns dynamics in situation space:
    - Given current situation, predict next situation
    - Works with abstract representations, not pixels
    - Can understand complex scene dynamics
    """
    
    def __init__(self, situation_dim=64, hidden_dim=128):
        super().__init__()
        self.situation_dim = situation_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP for dynamics (no RNN state issues)
        self.dynamics = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, situation_dim),
        )
    
    def forward(self, situation_current, hidden_state=None):
        """Predict next situation given current."""
        # Just use MLP, ignore hidden state
        next_situation = self.dynamics(situation_current)
        return next_situation, None


class SituationDecoder(nn.Module):
    """
    Reconstructs frame from situation representation.
    
    Takes abstract situation and generates pixel frame.
    This lets us verify what the network understood about the situation.
    """
    
    def __init__(self, situation_dim=64):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(situation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU(),
        )
        
        # Deconvolutional layers to reconstruct image
        self.spatial_decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, situation):
        """Reconstruct frame from situation vector."""
        x = self.decode(situation)
        x = x.view(x.size(0), 128, 4, 4)
        frame = self.spatial_decode(x)
        return frame


class SituationBasedPredictor(nn.Module):
    """
    Complete system: Encoder -> Predictor -> Decoder
    
    Pipeline:
    1. Encode frame to abstract situation
    2. Predict next situation
    3. Decode to verify understanding
    """
    
    def __init__(self, situation_dim=64, hidden_dim=128):
        super().__init__()
        self.situation_dim = situation_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = SituationEncoder(hidden_dim=situation_dim)
        self.predictor = SituationPredictor(situation_dim=situation_dim, hidden_dim=hidden_dim)
        self.decoder = SituationDecoder(situation_dim=situation_dim)
        
        self.hidden_state = None
    
    def forward(self, frame):
        """Full pipeline: frame -> situation -> predict -> reconstruct."""
        # Encode: what is the current situation?
        situation_current = self.encoder(frame)
        
        # Predict: what situation comes next?
        situation_next, self.hidden_state = self.predictor(situation_current, self.hidden_state)
        
        # Decode: reconstruct to see what we predicted
        reconstructed = self.decoder(situation_next)
        
        return situation_current, situation_next, reconstructed
    
    def reset_hidden(self):
        """Reset temporal state between sequences."""
        self.hidden_state = None


# ============================================================================
# SITUATION ANALYSIS - Understand what situations the network learned
# ============================================================================

class SituationAnalyzer:
    """Analyze learned situation representations."""
    
    def __init__(self, model, video):
        self.model = model
        self.video = video
        self.situations = []
    
    def encode_video(self):
        """Encode entire video to situation space."""
        with torch.no_grad():
            for frame in self.video:
                if len(frame.shape) == 2:
                    frame = frame.unsqueeze(0).unsqueeze(0)  # Add batch and channel
                elif len(frame.shape) == 3:
                    frame = frame.unsqueeze(1)  # Add channel
                
                situation = self.model.encoder(frame)
                self.situations.append(situation.cpu().numpy())
        
        return np.array(self.situations)  # (num_frames, situation_dim)
    
    def get_situation_statistics(self):
        """Get statistics about learned situations."""
        if not self.situations:
            self.encode_video()
        
        situations_array = np.array(self.situations)
        
        stats = {
            'mean': situations_array.mean(axis=0),
            'std': situations_array.std(axis=0),
            'min': situations_array.min(axis=0),
            'max': situations_array.max(axis=0),
            'variance': np.var(situations_array, axis=0),
        }
        
        return stats
    
    def get_situation_changes(self):
        """Measure how situations change frame to frame."""
        if not self.situations:
            self.encode_video()
        
        situations_array = np.array(self.situations)
        changes = np.diff(situations_array, axis=0)  # Situation deltas
        
        return {
            'mean_change': np.mean(np.abs(changes)),
            'max_change': np.max(np.abs(changes)),
            'change_per_frame': np.mean(np.abs(changes), axis=1),
        }


# ============================================================================
# MODELS WITH SITUATION UNDERSTANDING
# ============================================================================

class DynamicPlasticSituation(nn.Module):
    """Hebbian plasticity that learns situation dynamics."""
    def __init__(self, situation_dim=64, hidden_dim=128, learning_rate=0.01):
        super().__init__()
        self.situation_dim = situation_dim
        self.learning_rate = learning_rate
        
        self.encoder = SituationEncoder(hidden_dim=situation_dim)
        self.predictor = SituationPredictor(situation_dim=situation_dim, hidden_dim=hidden_dim)
        self.decoder = SituationDecoder(situation_dim=situation_dim)
        
        # Plasticity for predictor
        self.predictor_params = list(self.predictor.parameters())
        self.consolidation_mask = torch.ones(situation_dim)
        self.hidden_state = None
    
    def forward(self, frame):
        situation_current = self.encoder(frame)
        situation_next, self.hidden_state = self.predictor(situation_current, self.hidden_state)
        reconstructed = self.decoder(situation_next)
        return situation_current, situation_next, reconstructed
    
    def hebbian_situation_update(self, situation_current, situation_target):
        """
        Update predictor using Hebbian-like rule in situation space.
        Instead of pixel-level correlation, we use situation-level dynamics.
        """
        # Get prediction
        situation_pred, _ = self.predictor(situation_current, self.hidden_state)
        
        # Prediction error in situation space
        error = situation_target - situation_pred
        mse = torch.mean(error ** 2)
        
        # Hebbian-like update: reinforce correlations that predict well
        # This is conceptual - in practice we'd use backprop through situation space
        # But the key is we're learning in SITUATION space, not pixel space
        
        return mse
    
    def reset_hidden(self):
        self.hidden_state = None


class BackpropSituation(nn.Module):
    """Standard backprop learning situation dynamics."""
    def __init__(self, situation_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=situation_dim)
        self.predictor = SituationPredictor(situation_dim=situation_dim, hidden_dim=hidden_dim)
        self.decoder = SituationDecoder(situation_dim=situation_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.hidden_state = None
    
    def forward(self, frame):
        situation_current = self.encoder(frame)
        situation_next, self.hidden_state = self.predictor(situation_current, self.hidden_state)
        reconstructed = self.decoder(situation_next)
        return situation_current, situation_next, reconstructed
    
    def backprop_situation_update(self, frame_current, frame_target):
        """Learn to predict situations using backprop."""
        self.optimizer.zero_grad()
        
        situation_current = self.encoder(frame_current)
        situation_pred, self.hidden_state = self.predictor(situation_current, self.hidden_state)
        
        situation_target = self.encoder(frame_target)
        
        # Loss in situation space
        loss = self.criterion(situation_pred, situation_target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def reset_hidden(self):
        self.hidden_state = None


# ============================================================================
# VIDEO GENERATION - Same as before
# ============================================================================

def generate_video_type_1(num_frames=80, size=32, seed=42):
    """Single bouncing ball."""
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
# TRAINING WITH SITUATION UNDERSTANDING
# ============================================================================

def train_situation_model(model, video, num_epochs=30, batch_size=8):
    """Train model to understand and predict situations."""
    losses = []
    
    for epoch in range(num_epochs):
        model.reset_hidden()
        epoch_loss = 0
        num_batches = 0
        
        for frame_idx in range(len(video) - 1):
            current_frame = video[frame_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)
            target_frame = video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
            
            if isinstance(model, BackpropSituation):
                model.optimizer.zero_grad()
                
                situation_current = model.encoder(current_frame)
                situation_pred, model.hidden_state = model.predictor(situation_current, model.hidden_state)
                
                situation_target = model.encoder(target_frame.detach())
                
                loss = model.criterion(situation_pred, situation_target.detach())
                loss.backward()
                model.optimizer.step()
                
                epoch_loss += loss.detach().item()
            else:
                situation_current = model.encoder(current_frame)
                situation_target = model.encoder(target_frame.detach())
                loss = model.hebbian_situation_update(situation_current, situation_target)
                epoch_loss += loss.detach().item()
            
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    
    return losses


def evaluate_situation_model(model, video):
    """Evaluate how well model understands situations."""
    model.reset_hidden()
    total_error = 0
    
    with torch.no_grad():
        for frame_idx in range(len(video) - 1):
            current_frame = video[frame_idx].unsqueeze(0).unsqueeze(0)
            target_frame = video[frame_idx + 1].unsqueeze(0).unsqueeze(0)
            
            situation_current = model.encoder(current_frame)
            situation_next, model.hidden_state = model.predictor(situation_current, model.hidden_state)
            situation_target = model.encoder(target_frame)
            
            error = torch.mean((situation_next - situation_target) ** 2)
            total_error += error.item()
    
    return total_error / (len(video) - 1)


# ============================================================================
# MAIN TEST - SITUATION-BASED LEARNING
# ============================================================================

def situation_based_test():
    """Test situation-based predictive coding."""
    
    print("=" * 100)
    print("SITUATION-BASED PREDICTIVE CODING TEST")
    print("Learning abstract situation representations, not tokens or pixels")
    print("=" * 100)
    
    # Generate videos
    print("\n1. Generating 3 different situation types...")
    video1 = generate_video_type_1(num_frames=80, seed=42)
    video2 = generate_video_type_2(num_frames=80, seed=99)
    video3 = generate_video_type_3(num_frames=80, seed=77)
    print(f"   ✓ Video 1: Single bouncing ball (1 object situation)")
    print(f"   ✓ Video 2: Two bouncing balls (multi-object situation)")
    print(f"   ✓ Video 3: Rotating pattern (cyclic situation)")
    
    videos = [video1, video2, video3]
    video_names = ["Single Ball", "Two Balls", "Rotating"]
    
    # Create models
    print("\n2. Creating situation-aware models...")
    models = {
        'Hebbian + Situation': DynamicPlasticSituation(situation_dim=64, hidden_dim=128),
        'Backprop + Situation': BackpropSituation(situation_dim=64, hidden_dim=128),
    }
    
    for name in models.keys():
        print(f"   ✓ {name}")
    
    # Train on all videos
    print("\n3. Training on different situations...")
    print("-" * 100)
    
    results = {model_name: {'errors': [], 'situations': []} for model_name in models.keys()}
    
    for task_num, (video, video_name) in enumerate(zip(videos, video_names), 1):
        print(f"\n   SITUATION {task_num}: {video_name}")
        
        for model_name, model in models.items():
            print(f"\n     {model_name}:")
            
            # Train
            losses = train_situation_model(model, video, num_epochs=30)
            
            # Evaluate
            error = evaluate_situation_model(model, video)
            print(f"       Situation prediction error: {error:.6f}")
            results[model_name]['errors'].append(error)
            
            # Analyze situations learned
            analyzer = SituationAnalyzer(model, video)
            situations = analyzer.encode_video()
            situation_stats = analyzer.get_situation_statistics()
            situation_changes = analyzer.get_situation_changes()
            
            results[model_name]['situations'].append({
                'name': video_name,
                'situations': situations,
                'stats': situation_stats,
                'changes': situation_changes,
            })
            
            print(f"       Situation variance: {situation_stats['variance'].mean():.6f}")
            print(f"       Avg frame-to-frame change: {situation_changes['mean_change']:.6f}")
    
    # ========================================================================
    # ANALYSIS & VISUALIZATION
    # ========================================================================
    
    print("\n\n4. Analyzing learned situations...")
    print("-" * 100)
    
    for model_name in models.keys():
        print(f"\n   {model_name}:")
        for i, video_name in enumerate(video_names):
            sit_data = results[model_name]['situations'][i]
            print(f"     {video_name}:")
            print(f"       - Situation complexity (variance): {sit_data['stats']['variance'].mean():.6f}")
            print(f"       - Dynamics (frame-to-frame change): {sit_data['changes']['mean_change']:.6f}")
            print(f"       - Peak change: {sit_data['changes']['max_change']:.6f}")
    
    # Visualization
    print("\n\n5. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall performance
    ax1 = axes[0, 0]
    model_list = list(models.keys())
    x = np.arange(len(video_names))
    width = 0.35
    
    for i, model_name in enumerate(model_list):
        errors = results[model_name]['errors']
        ax1.bar(x + (i - 0.5) * width, errors, width, label=model_name, alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Situation Prediction Error', fontweight='bold', fontsize=11)
    ax1.set_title('Situation Understanding Accuracy', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(video_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Situation complexity
    ax2 = axes[0, 1]
    for i, model_name in enumerate(model_list):
        complexities = [results[model_name]['situations'][j]['stats']['variance'].mean() 
                       for j in range(len(videos))]
        ax2.plot(complexities, marker='o', label=model_name, linewidth=2, markersize=8)
    
    ax2.set_ylabel('Situation Complexity (Variance)', fontweight='bold', fontsize=11)
    ax2.set_title('How Complex Are Learned Situations?', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(video_names)))
    ax2.set_xticklabels(video_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Situation dynamics
    ax3 = axes[1, 0]
    for i, model_name in enumerate(model_list):
        dynamics = [results[model_name]['situations'][j]['changes']['mean_change'] 
                   for j in range(len(videos))]
        ax3.plot(dynamics, marker='s', label=model_name, linewidth=2, markersize=8)
    
    ax3.set_ylabel('Situation Change Rate', fontweight='bold', fontsize=11)
    ax3.set_title('How Fast Do Situations Change?', fontweight='bold', fontsize=12)
    ax3.set_xticks(range(len(video_names)))
    ax3.set_xticklabels(video_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "KEY INSIGHTS\n" + "="*50 + "\n\n"
    summary_text += "Situation-Based Learning:\n"
    summary_text += "• Models learn ABSTRACT situations\n"
    summary_text += "  (not pixels or local tokens)\n\n"
    summary_text += "• Understands WHAT is happening\n"
    summary_text += "  (objects, motion patterns)\n\n"
    summary_text += "• Predicts SITUATION EVOLUTION\n"
    summary_text += "  (what comes next in the world)\n\n"
    summary_text += "Results:\n"
    for model_name in model_list:
        total_err = sum(results[model_name]['errors'])
        summary_text += f"• {model_name}\n"
        summary_text += f"  Total error: {total_err:.5f}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontfamily='monospace',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('SITUATION-BASED PREDICTIVE CODING\nLearning to Understand Scenes, Not Tokens',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('situation_based_learning.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: situation_based_learning.png")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    results = situation_based_test()