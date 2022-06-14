import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def get_step_fn(std_value, train, conditioned=False, weighting=0):
  def loss_fn(score_model, batch):
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

def get_classifier_step_fn(std_value, train, loss_type='total', conditioned=False, weighting_dlsm = 0, weighting_ce = 0):
  def loss_fn(classifier_model, score_model, batch, labels, coef):
    # Define functions
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
      std = torch.empty(batch.shape, device=batch.device).fill_(std_value)
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
    if loss_type == 'total':
      # Calculate dlsm loss
      log_prob_class = torch.log(sm(out)+ 1e-8)
      label_mask = F.one_hot(labels, num_classes=2)
      grads_prob_class, = torch.autograd.grad(log_prob_class, perturbed_batch_var, 
                          grad_outputs=label_mask,
                          create_graph=True)
      loss_dlsm = torch.mean(0.5 * torch.square(grads_prob_class * (std ** weighting_dlsm) + score * (std ** weighting_dlsm) + z * (std ** (weighting_dlsm - 1)) ))
      loss_ce = torch.mean(loss_ce_fn(out, labels))
      loss = (loss_dlsm + coef*loss_ce)

    elif loss_type == 'dlsm':
      # Calculate dlsm loss
      log_prob_class = torch.log(sm(out)+ 1e-8)
      label_mask = F.one_hot(labels, num_classes=2)
      grads_prob_class, = torch.autograd.grad(log_prob_class, perturbed_batch_var, 
                          grad_outputs=label_mask,
                          create_graph=True)
      loss_dlsm = torch.mean(0.5 * torch.square(grads_prob_class * (std ** weighting_dlsm) + score * (std ** weighting_dlsm) + z * (std ** (weighting_dlsm - 1)) ))
      with torch.no_grad():
        loss_ce = torch.mean(loss_ce_fn(out, labels))
      loss = loss_dlsm

    elif loss_type == 'ce':
      # Calculate ce loss
      loss_ce = torch.mean(loss_ce_fn(out, labels))
      loss = loss_ce

    return loss, loss_ce.clone()

  def step_fn(classifier_model, score_model, optimizer, batch, labels, coef=1):
    if train:
      optimizer.zero_grad()
      classifier_model.train()
      loss, loss_ce = loss_fn(classifier_model, score_model, batch, labels, coef=coef)
      loss.backward()
      optimizer.step()
    else:
      with torch.no_grad():
        classifier_model.eval()
        loss, loss_ce = loss_fn(classifier_model, score_model, batch, labels, coef=coef)
    return loss, loss_ce
    
  return step_fn