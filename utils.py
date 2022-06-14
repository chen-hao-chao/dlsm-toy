import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
matplotlib.use('Agg')

from prdc import compute_prdc

import datasets

def get_scores(x, score_model, noise_conditioned=False, std_value=7.5):
  with torch.no_grad():
    std = torch.ones((x.shape[0],)).to(x.device) * std_value
    score = score_model(x, std) if noise_conditioned else score_model(x)
  return score

def get_conditional_scores(x, score_model, classifier_model, class_id=None, noise_conditioned=False, std_value=7.5):
  with torch.no_grad():
    # Get prior scores
    std = torch.ones((x.shape[0],)).to(x.device) * std_value
    score = score_model(x, std) if noise_conditioned else score_model(x)
    grad = torch.zeros(score.shape).to(x.device)
    # Get likelihood scores
    torch.set_grad_enabled(True)
    x_grad = Variable(x.detach().clone(), requires_grad=True)
    sm = nn.Softmax(dim=1)
    pred = sm(classifier_model(x_grad, std)) if noise_conditioned else sm(classifier_model(x_grad))
    pred = torch.log(pred)
    pred[torch.arange(pred.shape[0]), class_id].sum().backward()
    torch.set_grad_enabled(False)
    grad = x_grad.grad
  
  return score, grad

def oracle_score_denominator(point, batch, sigma):
  return torch.exp(  - ( ( (batch[:, 0]-point[0])**2 + (batch[:, 1]-point[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )

def oracle_score_numerator(point, batch, sigma):
  p = torch.exp(  - ( ( (batch[:, 0]-point[0])**2 + (batch[:, 1]-point[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )  
  diff = ( batch-point ) /  (sigma**2)
  diff[:, 0] *= p
  diff[:, 1] *= p
  return diff

def get_oracle_score(points, batch, sigma, eps=1e-8):
  points = torch.tensor(points)
  mean_numerator = torch.mean( oracle_score_numerator(points, batch, sigma = sigma), dim=0)
  mean_denominator = torch.mean( oracle_score_denominator(points, batch, sigma = sigma), dim=0) 
  return mean_numerator / (mean_denominator + eps)

def get_oracle_scores(points, config):
  config.data.dataset = 'inter_twinning_moon_upper'
  upper_ds = datasets.get_dataset(config)
  config.data.dataset = 'inter_twinning_moon_lower'
  lower_ds = datasets.get_dataset(config)
  iter_upper_ds = iter(upper_ds)
  iter_lower_ds = iter(lower_ds)
  batch_upper = torch.tensor(next(iter_upper_ds)['position']._numpy()).to(config.device)
  batch_lower = torch.tensor(next(iter_lower_ds)['position']._numpy()).to(config.device)

  config.data.dataset = 'inter_twinning_moon'
  config.training.batch_size = config.training.batch_size * 2
  ds = datasets.get_dataset(config)
  iter_ds = iter(ds)
  batch_full = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device)

  score_oracle_upper = np.zeros(points.shape)
  score_oracle_lower = np.zeros(points.shape)
  for i in range(points.shape[0]):
    score_oracle_upper[i, :] = get_oracle_score(points[i], batch_upper, config.model.std).cpu().numpy()
    score_oracle_lower[i, :] = get_oracle_score(points[i], batch_lower, config.model.std).cpu().numpy()
  
  score_oracle_full = np.zeros(points.shape)
  for i in range(points.shape[0]):
    score_oracle_full[i, :] = get_oracle_score(points[i], batch_full, config.model.std).cpu().numpy()
  
  score_oracle_likelihood_upper = score_oracle_upper - score_oracle_full
  score_oracle_likelihood_lower = score_oracle_lower - score_oracle_full

  return score_oracle_upper, score_oracle_lower, score_oracle_likelihood_upper, score_oracle_likelihood_lower, score_oracle_full


def calculate_prdc(config, samples_upper, samples_lower):
  with torch.no_grad():
    config.data.dataset = 'inter_twinning_moon_upper'
    upper_ds = datasets.get_dataset(config)
    config.data.dataset = 'inter_twinning_moon_lower'
    lower_ds = datasets.get_dataset(config)
    iter_upper_ds = iter(upper_ds)
    iter_lower_ds = iter(lower_ds)
    batch_upper = torch.tensor(next(iter_upper_ds)['position']._numpy())
    batch_lower = torch.tensor(next(iter_lower_ds)['position']._numpy())
    z = torch.randn_like(batch_upper)
    batch_upper = batch_upper + z * config.model.std
    batch_lower = batch_lower + z * config.model.std

    nearest_k = 5
    metrics_upper = compute_prdc(real_features=batch_upper.cpu().detach().numpy(),
                            fake_features=samples_upper.cpu().detach().numpy(),
                            nearest_k=nearest_k)
    metrics_lower = compute_prdc(real_features=batch_lower.cpu().detach().numpy(),
                            fake_features=samples_lower.cpu().detach().numpy(),
                            nearest_k=nearest_k)
  return metrics_upper, metrics_lower

def draw_and_save_data_points(w, h, samples_upper, samples_lower, filename):
  cond_upper = torch.zeros(samples_upper.shape[0], dtype=torch.long)
  cond_lower = torch.ones(samples_lower.shape[0], dtype=torch.long)

  figure(figsize=(w/4, h/4), dpi=100)
  plt.xlim((-w, w))
  plt.ylim((-h, h))
  data = np.concatenate((samples_upper.cpu().numpy(), samples_lower.cpu().numpy()), axis=0)
  label = np.concatenate((cond_upper.cpu().numpy(), cond_lower.cpu().numpy()), axis=0)
  plot_data = np.vstack((data.T, label)).T
  df = pd.DataFrame(data=plot_data, columns=("x", "y", "label"))
  sn.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.8)
  plt.xticks([])
  plt.yticks([])
  plt.legend('')
  plt.savefig(filename)

def draw_and_save_vector_field(x, y, w, h, scores, filename):
  figure(figsize=(w/4, h/4), dpi=100)
  plt.quiver(x, y, scores[:,0], scores[:,1])
  plt.xticks([])
  plt.yticks([])
  plt.savefig(filename)

def plot_vector_field(config, score_fn, dir_file, w=40, h=25, density=35, noise_conditioned=False, std_value=7.5):
  logging.info("Plotting Vector Field")
  with torch.no_grad():
    x, y = np.meshgrid(np.linspace(-w, w, density, dtype=np.float32), np.linspace(-h, h, density, dtype=np.float32))
    points = np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)
    points = torch.from_numpy(points).to(config.device)
    if noise_conditioned:
      std = torch.ones((points.shape[0],)).to(config.device) * std_value
      points_vf = score_fn(points, std).cpu().numpy()
    else:
      points_vf = score_fn(points).cpu().numpy()

    draw_and_save_vector_field(x, y, w, h, points_vf, dir_file)

def plot_vector_field_likelihood(config, classifier_model, dir_file, w=40, h=25, density=35, class_id=0, noise_conditioned=False, std_value=7.5):
  
  def calculate_vector_field_likelihood(classifier_model, points, class_id, noise_conditioned, std_value):
    with torch.no_grad():
      torch.set_grad_enabled(True)
      points_grad = Variable(points.detach().clone(), requires_grad=True)
      sm = nn.Softmax(dim=1)
      std = torch.ones((points.shape[0],)).to(points.device) * std_value
      pred = sm(classifier_model(points_grad, std)) if noise_conditioned else sm(classifier_model(points_grad))
      pred = torch.log(pred + 1e-8)
      pred[torch.arange(pred.shape[0]), class_id].sum().backward()
      torch.set_grad_enabled(False)    
      grad = points_grad.grad
      points_vf = grad.cpu().numpy()
    return points_vf

  logging.info("Plotting Vector Field")
  x, y = np.meshgrid(np.linspace(-w, w, density, dtype=np.float32), np.linspace(-h, h, density, dtype=np.float32))
  points = np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)
  points = torch.from_numpy(points).to(config.device)
  points_vf = calculate_vector_field_likelihood(classifier_model, points, class_id, noise_conditioned, std_value)

  draw_and_save_vector_field(x, y, w, h, points_vf, dir_file)