"""Classes for training and running inference on probes with memory optimization."""
import os
import sys
import gc
import numpy as np

from torch import optim
import torch
from tqdm import tqdm

class ProbeRegimen:
  """Memory-optimized regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """

  def __init__(self, args):
    self.args = args
    self.max_epochs = args['probe_training']['epochs']
    self.params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    # Add gradient accumulation steps (e.g., update weights every 4 batches)
    self.grad_accumulation_steps = args['probe_training'].get('grad_accumulation_steps', 1)
    # Add support for mixed precision training
    self.use_mixed_precision = args['probe_training'].get('use_mixed_precision', False)
    if self.use_mixed_precision and torch.cuda.is_available():
      self.scaler = torch.cuda.amp.GradScaler()

  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """
    self.optimizer = optim.Adam(probe.parameters(), lr=0.001)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=0)

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset):
    """ Trains a probe until a convergence criterion is met with memory optimization.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.set_optimizer(probe)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      # Initialize counters for the epoch
      epoch_train_loss = 0
      epoch_dev_loss = 0
      epoch_train_loss_count = 0
      epoch_dev_loss_count = 0
      
      # Training loop with gradient accumulation
      probe.train()
      batch_count = 0
      
      for batch in tqdm(train_dataset, desc='[training batch]'):
        # Clear gradients only at the beginning of accumulation steps
        if batch_count % self.grad_accumulation_steps == 0:
          self.optimizer.zero_grad()
        
        observation_batch, label_batch, length_batch, _ = batch
        
        # Use mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
          with torch.cuda.amp.autocast():
            word_representations = model(observation_batch)
            predictions = probe(word_representations)
            batch_loss, count = loss(predictions, label_batch, length_batch)
            # Scale the loss to handle gradient accumulation
            batch_loss = batch_loss / self.grad_accumulation_steps
            self.scaler.scale(batch_loss).backward()
        else:
          word_representations = model(observation_batch)
          predictions = probe(word_representations)
          batch_loss, count = loss(predictions, label_batch, length_batch)
          # Scale the loss to handle gradient accumulation
          batch_loss = batch_loss / self.grad_accumulation_steps
          batch_loss.backward()
        
        # Use numpy scalar to avoid memory leak
        batch_loss_np = float(batch_loss.item() * self.grad_accumulation_steps)
        count_np = float(count.item())
        epoch_train_loss += batch_loss_np * count_np
        epoch_train_loss_count += count_np
        
        batch_count += 1
        
        # Update weights after accumulation steps
        if batch_count % self.grad_accumulation_steps == 0 or batch_count == len(train_dataset):
          if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler.step(self.optimizer)
            self.scaler.update()
          else:
            self.optimizer.step()
          
          # Explicitly delete tensors to free memory
          del word_representations, predictions, batch_loss
          
        # Periodically force garbage collection
        if batch_count % (self.grad_accumulation_steps * 10) == 0:
          gc.collect()
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
      
      # Evaluation loop
      probe.eval()
      with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in tqdm(dev_dataset, desc='[dev batch]'):
          observation_batch, label_batch, length_batch, _ = batch
          
          word_representations = model(observation_batch)
          predictions = probe(word_representations)
          batch_loss, count = loss(predictions, label_batch, length_batch)
          
          # Use numpy scalar to avoid memory leak
          batch_loss_np = float(batch_loss.item())
          count_np = float(count.item())
          epoch_dev_loss += batch_loss_np * count_np
          epoch_dev_loss_count += count_np
          
          # Explicitly delete tensors to free memory
          del word_representations, predictions, batch_loss
      
      # Force garbage collection after evaluation
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      
      # Calculate epoch losses
      train_loss = epoch_train_loss / max(1.0, epoch_train_loss_count)
      dev_loss = epoch_dev_loss / max(1.0, epoch_dev_loss_count)
      
      self.scheduler.step(dev_loss)
      tqdm.write(f'[epoch {epoch_index}] Train loss: {train_loss:.6f}, Dev loss: {dev_loss:.6f}')
      
      # Save model if improved
      if dev_loss < min_dev_loss - 0.0001:
        torch.save(probe.state_dict(), self.params_path)
        min_dev_loss = dev_loss
        min_dev_loss_epoch = epoch_index
        tqdm.write('Saving probe parameters')
      elif min_dev_loss_epoch < epoch_index - 4:
        tqdm.write('Early stopping')
        break

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object 

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    
    with torch.no_grad():  # Disable gradient computation during inference
      for batch in tqdm(dataset, desc='[predicting]'):
        observation_batch, label_batch, length_batch, _ = batch
        
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        # Store predictions in CPU numpy arrays immediately to free GPU memory
        predictions_by_batch.append(predictions.cpu().numpy())
        
        # Free memory explicitly
        del word_representations, predictions
        
        # Periodically force garbage collection
        if len(predictions_by_batch) % 10 == 0:
          gc.collect()
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predictions_by_batch