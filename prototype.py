import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PlasticLayer(nn.Module):
    """Single layer with Hebbian plasticity."""
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


def train_on_video(model, video, num_epochs=50, batch_size=8):
    """Train model on video sequences."""
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Train on consecutive frame pairs
        for start_idx in range(0, len(video) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(video) - 1)
            
            current_frames = video[start_idx:end_idx].reshape(-1, 1024)  # Flatten to 1024
            next_frames = video[start_idx+1:end_idx+1].reshape(-1, 1024)
            
            # Forward
            hidden, reconstruction = model.forward(current_frames)
            
            # Learn
            loss = model.hebbian_update(current_frames, hidden, next_frames)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def test_prediction(model, video, num_examples=5):
    """Visualize what model predicts."""
    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3*num_examples))
    
    with torch.no_grad():
        for ex in range(num_examples):
            idx = ex * 15
            current = video[idx].reshape(1, 1024)
            actual_next = video[idx + 1]
            
            hidden, predicted = model.forward(current)
            predicted = predicted.reshape(32, 32)
            
            axes[ex, 0].imshow(video[idx].reshape(32, 32), cmap='gray')
            axes[ex, 0].set_title('Current')
            axes[ex, 0].axis('off')
            
            axes[ex, 1].imshow(predicted.numpy(), cmap='gray')
            axes[ex, 1].set_title('Predicted')
            axes[ex, 1].axis('off')
            
            axes[ex, 2].imshow(actual_next.reshape(32, 32), cmap='gray')
            axes[ex, 2].set_title('Actual Next')
            axes[ex, 2].axis('off')
            
            error = torch.abs(actual_next.reshape(1, 1024) - predicted.reshape(1, 1024)).reshape(32, 32)
            axes[ex, 3].imshow(error.numpy(), cmap='hot')
            axes[ex, 3].set_title('Error')
            axes[ex, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100, bbox_inches='tight')
    print("Saved predictions.png")


if __name__ == "__main__":
    print("Generating video...")
    video = generate_bouncing_ball_video(num_frames=100, size=32)
    print(f"Video shape: {video.shape}")
    
    print("\nCreating model...")
    model = PlasticLayer(input_size=1024, hidden_size=128, learning_rate=0.01)
    
    print("\nTraining...")
    losses = train_on_video(model, video, num_epochs=50, batch_size=8)
    
    print(f"\nInitial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    print("\nVisualizing predictions...")
    test_prediction(model, video, num_examples=5)
    
    print("\nDone!")