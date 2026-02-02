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
# BAYESIAN CONFIDENCE SCORING FOR HEBBIAN
# ============================================================================

class BayesianConfidenceScorer:
    """
    Apply Bayes theorem to Hebbian learning.
    
    Bayes: P(model|data) ∝ P(data|model) * P(model)
    
    In Hebbian context:
    - Likelihood: How well did prediction match reality?
    - Prior: How confident were we before?
    - Posterior: How much should we update?
    
    Confidence scoring weights Hebbian updates by how reliable they are.
    """
    
    def __init__(self, prior_confidence=0.5, smoothing=0.1):
        """
        Args:
            prior_confidence: Initial belief in our model (0-1)
            smoothing: How much to smooth confidence updates
        """
        self.prior_confidence = prior_confidence
        self.smoothing = smoothing
        self.confidence_history = []
    
    def compute_likelihood(self, prediction_error):
        """
        Likelihood: P(data|model)
        
        Lower error = higher likelihood that our model is correct
        Uses exponential decay: likelihood = exp(-error)
        """
        # Clamp error for stability
        error_clamped = torch.clamp(prediction_error, min=0, max=10)
        likelihood = torch.exp(-error_clamped)
        return likelihood
    
    def update_confidence(self, likelihood, prior):
        """
        Bayes update: posterior ∝ likelihood * prior
        
        P(model_is_good | prediction_error) = P(error | good_model) * P(good_model) / P(error)
        
        Simplified: posterior = likelihood * prior (unnormalized)
        """
        posterior = likelihood * prior
        
        # Normalize and smooth
        posterior = torch.clamp(posterior, min=0, max=1)
        posterior = prior * (1 - self.smoothing) + posterior * self.smoothing
        
        self.confidence_history.append(posterior.item() if hasattr(posterior, 'item') else float(posterior))
        
        return posterior
    
    def compute_update_weight(self, confidence):
        """
        Weight Hebbian updates by confidence.
        
        High confidence = trust the Hebbian update more
        Low confidence = be conservative
        
        Returns weight in [0, 1]
        """
        return torch.clamp(confidence, min=0, max=1)
    
    def step(self, prediction_error, current_confidence):
        """
        Single Bayesian update step.
        
        Returns: new_confidence, update_weight
        """
        likelihood = self.compute_likelihood(prediction_error)
        new_confidence = self.update_confidence(likelihood, current_confidence)
        update_weight = self.compute_update_weight(new_confidence)
        
        return new_confidence, update_weight


# ============================================================================
# HEBBIAN MODELS WITH BAYESIAN CONFIDENCE
# ============================================================================

class HebbianDynamicConfidence(nn.Module):
    """Hebbian + Dynamic + Bayesian Confidence Scoring."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.W_dynamics = nn.Parameter(torch.randn(64, 128) * 0.01)
        self.W_predict = nn.Parameter(torch.randn(128, 64) * 0.01)
        
        self.learning_rate = learning_rate
        self.consolidation_mask = torch.ones(128)
        
        # Bayesian confidence scorer
        self.confidence_scorer = BayesianConfidenceScorer(prior_confidence=0.5, smoothing=0.1)
        self.confidence = torch.tensor(0.5)  # Initial confidence
        
        self.update_weights_history = []
    
    def forward(self, frame):
        situation = self.encoder(frame)
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return situation, pred_situation, hidden
    
    def hebbian_update_with_confidence(self, situation, hidden, target_situation):
        """
        Hebbian update weighted by Bayesian confidence.
        
        Key insight: Only update strongly when we're confident in our model.
        """
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        error = target_situation - pred_situation
        mse = torch.mean(error ** 2)
        
        # Compute Bayesian confidence
        self.confidence, update_weight = self.confidence_scorer.step(mse, self.confidence)
        self.update_weights_history.append(update_weight.item())
        
        # Handle tensor dimensions
        if situation.dim() > 1:
            situation = situation.squeeze(0)
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)
        if error.dim() > 1:
            error = error.squeeze(0)
        
        # Compute Hebbian updates
        dW_dynamics = torch.outer(situation, hidden)
        dW_predict = torch.outer(hidden, error)
        
        # Weight updates by confidence
        dW_dynamics = dW_dynamics * update_weight
        dW_predict = dW_predict * update_weight
        
        # Apply with consolidation
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_dynamics.data += self.learning_rate * dW_dynamics * consolidation_factor
        self.W_predict.data += self.learning_rate * dW_predict * consolidation_factor.T
        
        return mse
    
    def predict_next_situation(self, situation):
        """Predict next situation from current."""
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return pred


class HebbianStaticConfidence(nn.Module):
    """Hebbian + Static + Bayesian Confidence Scoring."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
        self.W_dynamics = nn.Parameter(torch.randn(64, 128) * 0.01)
        self.W_predict = nn.Parameter(torch.randn(128, 64) * 0.01)
        
        self.learning_rate = learning_rate
        
        self.confidence_scorer = BayesianConfidenceScorer(prior_confidence=0.5, smoothing=0.1)
        self.confidence = torch.tensor(0.5)
        
        self.update_weights_history = []
    
    def forward(self, frame):
        situation = self.encoder(frame)
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return situation, pred_situation, hidden
    
    def hebbian_update_with_confidence(self, situation, hidden, target_situation):
        """Hebbian update weighted by Bayesian confidence."""
        pred_situation = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        error = target_situation - pred_situation
        mse = torch.mean(error ** 2)
        
        self.confidence, update_weight = self.confidence_scorer.step(mse, self.confidence)
        self.update_weights_history.append(update_weight.item())
        
        if situation.dim() > 1:
            situation = situation.squeeze(0)
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)
        if error.dim() > 1:
            error = error.squeeze(0)
        
        dW_dynamics = torch.outer(situation, hidden)
        dW_predict = torch.outer(hidden, error)
        
        dW_dynamics = dW_dynamics * update_weight
        dW_predict = dW_predict * update_weight
        
        self.W_dynamics.data += self.learning_rate * dW_dynamics
        self.W_predict.data += self.learning_rate * dW_predict
        
        return mse
    
    def predict_next_situation(self, situation):
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return pred


class HebbianDynamic(nn.Module):
    """Hebbian + Dynamic (baseline, no confidence)."""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.encoder = SituationEncoder(hidden_dim=64)
        self.decoder = SituationDecoder(situation_dim=64)
        
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
        mse = torch.mean(error ** 2)
        
        if situation.dim() > 1:
            situation = situation.squeeze(0)
        if hidden.dim() > 1:
            hidden = hidden.squeeze(0)
        if error.dim() > 1:
            error = error.squeeze(0)
        
        dW_dynamics = torch.outer(situation, hidden)
        dW_predict = torch.outer(hidden, error)
        
        consolidation_factor = self.consolidation_mask.unsqueeze(0)
        self.W_dynamics.data += self.learning_rate * dW_dynamics * consolidation_factor
        self.W_predict.data += self.learning_rate * dW_predict * consolidation_factor.T
        
        return mse
    
    def predict_next_situation(self, situation):
        hidden = torch.matmul(situation.unsqueeze(0), self.W_dynamics).squeeze(0)
        hidden = torch.relu(hidden)
        pred = torch.matmul(hidden.unsqueeze(0), self.W_predict).squeeze(0)
        return pred


class BackpropDynamic(nn.Module):
    """Backprop + Dynamic (baseline for comparison)."""
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
# VIDEO GENERATION
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
    """Video 2: Ball bouncing."""
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
# MAIN TEST
# ============================================================================

def bayesian_hebbian_test():
    """Test Hebbian with Bayesian confidence scoring vs baselines."""
    
    print("=" * 120)
    print("BAYESIAN HEBBIAN LEARNING TEST")
    print("Hebbian + Bayesian Confidence Scoring vs Hebbian Baseline vs Backprop")
    print("=" * 120)
    
    # Generate videos
    print("\n1. Generating videos...")
    video1 = generate_video_sequence_1(num_frames=60, seed=42)
    video2 = generate_video_sequence_2(num_frames=60, seed=99)
    print(f"   ✓ Video 1: Linear motion")
    print(f"   ✓ Video 2: Bouncing motion")
    
    # Create 4 models
    print("\n2. Creating 4 models...")
    models = {
        'Hebbian+Dynamic+Confidence': HebbianDynamicConfidence(learning_rate=0.01),
        'Hebbian+Static+Confidence': HebbianStaticConfidence(learning_rate=0.01),
        'Hebbian+Dynamic (Baseline)': HebbianDynamic(learning_rate=0.01),
        'Backprop+Dynamic': BackpropDynamic(),
    }
    
    for name in models.keys():
        print(f"   ✓ {name}")
    
    # PHASE 1: Train on video 1
    print("\n" + "=" * 120)
    print("PHASE 1: TRAINING ON VIDEO 1")
    print("=" * 120)
    
    training_losses = {name: [] for name in models.keys()}
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for epoch in range(20):
            epoch_loss = 0
            
            for frame_idx in range(len(video1) - 1):
                current_frame = video1[frame_idx].unsqueeze(0).unsqueeze(0)
                target_frame = video1[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                if 'Confidence' in model_name:
                    situation, _, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update_with_confidence(situation, hidden, target_situation)
                elif 'Hebbian' in model_name:
                    situation, _, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update(situation, hidden, target_situation)
                else:
                    loss = model.backprop_update(current_frame, target_frame)
                
                epoch_loss += loss
            
            avg_loss = epoch_loss / (len(video1) - 1)
            training_losses[model_name].append(float(avg_loss))
            
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch + 1}: {avg_loss:.6f}")
    
    # PHASE 2: Blind prediction
    print("\n\n" + "=" * 120)
    print("PHASE 2: MEMORY-BASED BLIND PREDICTION")
    print("=" * 120)
    
    memory_errors = {name: [] for name in models.keys()}
    
    print("\n   Making predictions without seeing video 2...")
    
    with torch.no_grad():
        for frame_idx in range(1, len(video2)):
            current_frame = video2[frame_idx - 1].unsqueeze(0).unsqueeze(0)
            target_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
            
            for model_name, model in models.items():
                current_situation = model.encoder(current_frame)
                pred_situation = model.predict_next_situation(current_situation)
                target_situation = model.encoder(target_frame)
                
                error = torch.mean((pred_situation - target_situation) ** 2).item()
                memory_errors[model_name].append(error)
    
    memory_error_avg = {name: np.mean(errors) for name, errors in memory_errors.items()}
    
    for model_name, error in memory_error_avg.items():
        print(f"     {model_name}: {error:.6f}")
    
    # PHASE 3: Train on video 2
    print("\n\n" + "=" * 120)
    print("PHASE 3: TRAINING ON VIDEO 2 (ADAPTATION)")
    print("=" * 120)
    
    for model_name, model in models.items():
        print(f"\n   {model_name}:")
        
        for epoch in range(20):
            epoch_loss = 0
            
            for frame_idx in range(len(video2) - 1):
                current_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
                target_frame = video2[frame_idx + 1].unsqueeze(0).unsqueeze(0)
                
                if 'Confidence' in model_name:
                    situation, _, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update_with_confidence(situation, hidden, target_situation)
                elif 'Hebbian' in model_name:
                    situation, _, hidden = model.forward(current_frame)
                    target_situation = model.encoder(target_frame)
                    loss = model.hebbian_update(situation, hidden, target_situation)
                else:
                    loss = model.backprop_update(current_frame, target_frame)
                
                epoch_loss += loss
            
            if (epoch + 1) % 5 == 0:
                print(f"     Epoch {epoch + 1}: {epoch_loss / (len(video2) - 1):.6f}")
    
    # PHASE 4: Final evaluation
    print("\n\n" + "=" * 120)
    print("PHASE 4: FINAL ACCURACY")
    print("=" * 120)
    
    final_errors = {name: [] for name in models.keys()}
    
    with torch.no_grad():
        for frame_idx in range(len(video2) - 1):
            current_frame = video2[frame_idx].unsqueeze(0).unsqueeze(0)
            target_frame = video2[frame_idx + 1].unsqueeze(0).unsqueeze(0)
            
            for model_name, model in models.items():
                current_situation = model.encoder(current_frame)
                pred_situation = model.predict_next_situation(current_situation)
                target_situation = model.encoder(target_frame)
                
                error = torch.mean((pred_situation - target_situation) ** 2).item()
                final_errors[model_name].append(error)
    
    final_error_avg = {name: np.mean(errors) for name, errors in final_errors.items()}
    
    for model_name, error in final_error_avg.items():
        print(f"   {model_name}: {error:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n\n5. Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name in models.keys():
        ax1.plot(training_losses[model_name], label=model_name, linewidth=2, marker='o', markersize=3)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_title('Phase 1: Training on Video 1', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory-based prediction
    ax2 = fig.add_subplot(gs[0, 1])
    model_names = list(models.keys())
    memory_errs = [memory_error_avg[m] for m in model_names]
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c']
    
    bars = ax2.bar(range(len(model_names)), memory_errs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax2.set_title('Phase 2: Blind Prediction (No Video 2)', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([m.replace('+', '\n+') for m in model_names], fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, memory_errs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Final accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    final_errs = [final_error_avg[m] for m in model_names]
    
    bars = ax3.bar(range(len(model_names)), final_errs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Prediction Error', fontweight='bold', fontsize=11)
    ax3.set_title('Phase 4: Final Accuracy', fontweight='bold', fontsize=12)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([m.replace('+', '\n+') for m in model_names], fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, final_errs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Confidence tracking
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(models['Hebbian+Dynamic+Confidence'].confidence_scorer, 'confidence_history'):
        conf_history = models['Hebbian+Dynamic+Confidence'].confidence_scorer.confidence_history
        ax4.plot(conf_history, label='Confidence', linewidth=2, color='#3498db')
    ax4.set_ylabel('Confidence', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Update Step', fontweight='bold', fontsize=11)
    ax4.set_title('Bayesian Confidence Evolution (Video 1)', fontweight='bold', fontsize=12)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Update weight tracking
    ax5 = fig.add_subplot(gs[1, 1])
    update_weights = models['Hebbian+Dynamic+Confidence'].update_weights_history
    ax5.plot(update_weights, label='Update Weight', linewidth=2, color='#e74c3c')
    ax5.set_ylabel('Update Weight', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Update Step', fontweight='bold', fontsize=11)
    ax5.set_title('Hebbian Update Weight (Video 1)', fontweight='bold', fontsize=12)
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Comparison summary
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(model_names))
    width = 0.25
    
    ax6.bar(x - width, memory_errs, width, label='Memory Prediction', alpha=0.8, color='#f39c12', edgecolor='black')
    ax6.bar(x, final_errs, width, label='Final Accuracy', alpha=0.8, color='#3498db', edgecolor='black')
    
    ax6.set_ylabel('Error', fontweight='bold', fontsize=11)
    ax6.set_title('Continual Learning: Blind vs Final', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([m.replace('+', '\n+') for m in model_names], fontsize=8)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('BAYESIAN HEBBIAN LEARNING TEST\nHebbian+Confidence vs Hebbian Baseline vs Backprop',
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('bayesian_hebbian_test.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved: bayesian_hebbian_test.png")
    
    # Print detailed analysis
    print("\n\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)
    
    print("\n🔮 CONFIDENCE IMPACT:")
    conf_benefit_dynamic = (memory_error_avg['Hebbian+Dynamic (Baseline)'] - memory_error_avg['Hebbian+Dynamic+Confidence']) / memory_error_avg['Hebbian+Dynamic (Baseline)'] * 100
    conf_benefit_static = (memory_error_avg['Hebbian+Static (Baseline)'] - memory_error_avg['Hebbian+Static+Confidence']) / memory_error_avg['Hebbian+Static (Baseline)'] * 100 if 'Hebbian+Static (Baseline)' in memory_error_avg else 0
    
    print(f"   Confidence helps Dynamic: {conf_benefit_dynamic:+.1f}%")
    if conf_benefit_static != 0:
        print(f"   Confidence helps Static: {conf_benefit_static:+.1f}%")
    
    print("\n📊 BLIND PREDICTION RANKING:")
    ranked_memory = sorted(memory_error_avg.items(), key=lambda x: x[1])
    for i, (name, err) in enumerate(ranked_memory, 1):
        print(f"   {i}. {name}: {err:.6f}")
    
    print("\n🎯 FINAL ACCURACY RANKING:")
    ranked_final = sorted(final_error_avg.items(), key=lambda x: x[1])
    for i, (name, err) in enumerate(ranked_final, 1):
        print(f"   {i}. {name}: {err:.6f}")
    
    print("\n" + "=" * 120)
    
    return training_losses, memory_error_avg, final_error_avg


if __name__ == "__main__":
    training_losses, memory_errors, final_errors = bayesian_hebbian_test()