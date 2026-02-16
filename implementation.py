import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
