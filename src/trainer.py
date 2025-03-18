import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import os

class PDETrainer:
    """Trainer for Physics-Informed Neural Networks."""
    
    def __init__(
        self,
        model: nn.Module,
        pde: 'PDEBase',
        optimizer_config: Dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize trainer.

        :param model: PINN model
        :param pde: PDE instance
        :param optimizer_config: Optimizer configuration
        :param device: Device to train on
        :param checkpoint_dir: Directory to save checkpoints
        """
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.pde = pde
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.get('learning_rate', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        
        # Setup learning rate schedulers
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'residual_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = optimizer_config.get('patience', 10)
        self.patience_counter = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _compute_validation_loss(self, num_points: int = 1000) -> Dict[str, float]:
        """
        Compute validation loss on a separate set of points.
        
        :param num_points: Number of validation points
        :return: Dictionary of validation losses
        """
        self.model.eval()
        with torch.no_grad():
            # Generate validation points
            x_val, t_val = self.pde.generate_collocation_points(num_points)
            x_val = x_val.to(self.device)
            t_val = t_val.to(self.device)
            
            # Compute losses
            residual_loss = self.pde.compute_residual_loss(self.model, x_val, t_val)
            boundary_loss = self.pde.compute_boundary_loss(self.model)
            initial_loss = self.pde.compute_initial_loss(self.model)
            
            total_loss = residual_loss + boundary_loss + initial_loss
            
            return {
                'total_loss': total_loss.item(),
                'residual_loss': residual_loss.item(),
                'boundary_loss': boundary_loss.item(),
                'initial_loss': initial_loss.item()
            }
    
    def _update_learning_rate(self, val_loss: float):
        """
        Update learning rate based on validation loss.
        
        :param val_loss: Current validation loss
        """
        # Update ReduceLROnPlateau scheduler
        self.scheduler.step(val_loss)
        
        # Update cosine scheduler
        self.cosine_scheduler.step()
        
        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
        self.logger.info(f"Current learning rate: {current_lr:.6f}")
    
    def train(
        self,
        num_epochs: int,
        batch_size: int,
        num_points: int,
        validation_frequency: int = 10,
        save_frequency: int = 50
    ):
        """
        Train the model.
        
        :param num_epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param num_points: Number of collocation points
        :param validation_frequency: Frequency of validation
        :param save_frequency: Frequency of checkpoint saving
        """
        self.logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            
            # Training loop with progress bar
            pbar = tqdm(range(num_points // batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
            for _ in pbar:
                # Generate batch of collocation points
                x_batch, t_batch = self.pde.generate_collocation_points(batch_size)
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Compute losses
                residual_loss = self.pde.compute_residual_loss(self.model, x_batch, t_batch)
                boundary_loss = self.pde.compute_boundary_loss(self.model)
                initial_loss = self.pde.compute_initial_loss(self.model)
                
                # Total loss
                total_loss = residual_loss + boundary_loss + initial_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimize
                self.optimizer.step()
                
                # Update progress bar
                epoch_losses.append(total_loss.item())
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.6f}",
                    'residual': f"{residual_loss.item():.6f}",
                    'boundary': f"{boundary_loss.item():.6f}",
                    'initial': f"{initial_loss.item():.6f}"
                })
            
            # Compute average epoch loss
            avg_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(avg_loss)
            
            # Validation
            if (epoch + 1) % validation_frequency == 0:
                val_losses = self._compute_validation_loss()
                self.history['val_loss'].append(val_losses['total_loss'])
                self.history['residual_loss'].append(val_losses['residual_loss'])
                self.history['boundary_loss'].append(val_losses['boundary_loss'])
                self.history['initial_loss'].append(val_losses['initial_loss'])
                
                # Update learning rate
                self._update_learning_rate(val_losses['total_loss'])
                
                # Log validation metrics
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_loss:.6f} - "
                    f"Val Loss: {val_losses['total_loss']:.6f} - "
                    f"Residual: {val_losses['residual_loss']:.6f} - "
                    f"Boundary: {val_losses['boundary_loss']:.6f} - "
                    f"Initial: {val_losses['initial_loss']:.6f}"
                )
                
                # Early stopping check
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.patience_counter = 0
                    self.save_checkpoint(f"best_model.pth")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.logger.info("Early stopping triggered!")
                        break
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        :param filename: Name of the checkpoint file
        """
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load training checkpoint.
        
        :param filename: Name of the checkpoint file
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        :return: Dictionary of training metrics
        """
        return self.history
    
    def plot_training_history(self):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot component losses
        plt.subplot(2, 2, 2)
        plt.plot(self.history['residual_loss'], label='Residual Loss')
        plt.plot(self.history['boundary_loss'], label='Boundary Loss')
        plt.plot(self.history['initial_loss'], label='Initial Loss')
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.history['learning_rate'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
