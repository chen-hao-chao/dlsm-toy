import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def get_step_fn(std_value, train, conditioned=False, weighting=0):
  """Construct a one-step training/evaluation function.
  Args: 
    std_value: (float/tuple) The standard deviation(s) for the noises.
    train: (bool) The indication for training. It is set as True for training mode.
    conditioned: (bool) The indication for the noise-conditioned model. It is set as
                        True if the model is conditioned on different standard deviations.
    weighting: (int) The power of the balancing coefficient for the DSM loss. For example, 
                     if wighting=2, the coefficient is 1/std^(2*wighting).
  Returns:
    step_fn: (func) A one-step training/evaluation function.
  """
  def loss_fn(score_model, batch):
    """Compute the loss function.
    Args:
      score_model: (nn.Module) A parameterized score model.
      batch: (tensor) A mini-batch of training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    if conditioned:
      # Calculate standard deviation
      std_value_max, std_value_min = std_value
      t = torch.rand(batch.shape[0], device=batch.device)
      std_value_cond = (std_value_min * (std_value_max / std_value_min) ** t)
      std = std_value_cond[:, None]
      # Perturb the data
      z = torch.randn_like(batch, device=batch.device)
      perturbed_batch = batch + std * z
      # Make predictions
      score = score_model(perturbed_batch, std_value_cond)
      
    else:
      # Calculate standard deviation
      std = torch.empty(batch.shape, device=batch.device).fill_(std_value)
      # Perturb the data
      z = torch.randn_like(batch, device=batch.device)
      perturbed_batch = batch + std * z
      # Make predictions
      score = score_model(perturbed_batch)
    
    # Calculate the losses
    losses = torch.square(score * (std ** weighting) + z * (std ** (weighting-1)) )
    loss = torch.mean(losses)

    return loss
    
  def step_fn(score_model, optimizer, batch):
    """Running one step of training or evaluation.
    Args:
      score_model: (nn.Module) A parameterized score model.
      optimizer: (torch.optim) An optimizer function that can update score_model with '.step()' function.
      batch: (tensor) A mini-batch of training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    if train:
      optimizer.zero_grad()
      score_model.train()
      loss = loss_fn(score_model, batch)
      loss.backward()
      optimizer.step()
    else:
      with torch.no_grad():
        score_model.eval()
        loss = loss_fn(score_model, batch)
    return loss

  return step_fn

def get_classifier_step_fn(std_value, train, loss_type='total', conditioned=False, weighting_dlsm=0, weighting_ce=0, coef=1.0, eps=1e-8):
  """Construct a one-step training/evaluation function.
  Args: 
    std_value: (float/tuple) The standard deviation(s) for the noises.
    loss_type: (str) The indication for the type of loss.
    train: (bool) The indication for training. It is set as True for training mode.
    conditioned: (bool) The indication for the noise-conditioned model. It is set as
                        True if the model is conditioned on different standard deviations.
    weighting_dlsm: (int) The power of the balancing coefficient for the DLSM loss. For example, 
                     if weighting_dlsm=2, the coefficient is 1/std^(2*2).
    weighting_ce: (int) The power of the balancing coefficient for the CE loss. For example, 
                     if weighting_ce=0, the coefficient is 1/std^(2*0).
    coef: (float) The coefficient for balancing the DLSM and the CE losses.
    eps: (float) An exetremely small value. It is used for preventing overflow.
  Returns:
    step_fn: (func) A one-step training/evaluation function.
  """
  def loss_fn(classifier_model, score_model, batch, labels):
    """Compute the loss function.
    Args:
      score_model: (nn.Module) A parameterized score model.
      classifier_model: (nn.Module) A parameterized classifier.
      batch: (tensor) A mini-batch of training data.
      labels: (tensor) A mini-batch of labels of the training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    # Define softmax and ce functions
    loss_ce_fn = torch.nn.CrossEntropyLoss(reduce=False)
    sm = nn.Softmax(dim=1)

    # Get standard deviation
    if conditioned:
      std_value_max, std_value_min = std_value
      t = torch.rand(batch.shape[0], device=batch.device)
      std_value_cond = (std_value_min * (std_value_max / std_value_min) ** t)
      std = std_value_cond[:, None]
      # Perturb the images
      z = torch.randn_like(batch, device=batch.device)
      perturbed_batch = batch + std * z
      # Forward pass
      with torch.no_grad():
        score_model.eval()
        score = score_model(perturbed_batch, std_value_cond)
      perturbed_batch_var = Variable(perturbed_batch.clone(), requires_grad=True)
      out = classifier_model(perturbed_batch_var, std_value_cond)

    else:
      std_value_cond = torch.empty(batch.shape[0], device=batch.device).fill_(std_value)
      std = std_value_cond[:, None]
      # Perturb the images
      z = torch.randn_like(batch, device=batch.device)
      perturbed_batch = batch + std * z
      # Forward pass
      with torch.no_grad():
        score_model.eval()
        score = score_model(perturbed_batch)
      perturbed_batch_var = Variable(perturbed_batch.clone(), requires_grad=True)
      out = classifier_model(perturbed_batch_var)

    # Calculate the losses
    if loss_type == 'total' or loss_type == 'dlsm':
      # Calculate the dlsm loss
      log_prob_class = torch.log(sm(out)+ eps)
      label_mask = F.one_hot(labels, num_classes=2)
      grads_prob_class, = torch.autograd.grad(log_prob_class, perturbed_batch_var, 
                          grad_outputs=label_mask,
                          create_graph=True)
      loss_dlsm = torch.mean(0.5 * torch.square(grads_prob_class * (std ** weighting_dlsm) + score * (std ** weighting_dlsm) + z * (std ** (weighting_dlsm-1)) ))

    if loss_type == 'total' or loss_type == 'ce':
      # Calculate the ce loss
      loss_ce = torch.mean(loss_ce_fn(out, labels)*(std_value_cond ** (-2 * weighting_ce)))
    
    loss = (loss_dlsm + coef * loss_ce) if loss_type == 'total' else (loss_dlsm if loss_type == 'dlsm' else loss_ce)
    return loss

  def step_fn(classifier_model, score_model, optimizer, batch, labels):
    """Running one step of training or evaluation.
    Args:
      score_model: (nn.Module) A parameterized score model.
      classifier_model: (nn.Module) A parameterized classifier.
      optimizer: (torch.optim) An optimizer function that can update score_model with '.step()' function.
      batch: (tensor) A mini-batch of training data.
      labels: (tensor) A mini-batch of labels of the training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    if train:
      optimizer.zero_grad()
      classifier_model.train()
      loss = loss_fn(classifier_model, score_model, batch, labels)
      loss.backward()
      optimizer.step()
    else:
      with torch.no_grad():
        classifier_model.eval()
        loss = loss_fn(classifier_model, score_model, batch, labels)
    return loss
    
  return step_fn