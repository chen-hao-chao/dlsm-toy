import os
import logging
import numpy as np

import datasets
import losses_plot
from models import simple_classifier_fn, simple_score_fn
from utils import plot_vector_field_likelihood
from utils import get_conditional_scores, get_oracle_scores
import torch
import tensorflow as tf
import torch.optim as optim
from torch.utils import tensorboard

def train(config, workdir):
  """Execute the training procedure for the classifier.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  visualization_dir = os.path.join(workdir, "visualization")
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  tf.io.gfile.makedirs(visualization_dir)
  tf.io.gfile.makedirs(checkpoint_dir)
  
  # Add tensorboard writter
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize the classifier and the score model
  if config.model.noise_conditioned:
    classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)
    score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
    train_step_fn = losses_plot.get_classifier_step_fn((config.model.std_max, config.model.std_min), train=True,
                                                    loss_type=config.training.loss, conditioned=True,
                                                      weighting_dlsm=config.model.weighting_dlsm, weighting_ce=config.model.weighting_ce)
  else:
    classifier_model = simple_classifier_fn.simple_classifier_fn(config).to(config.device)
    score_model = simple_score_fn.simple_score_fn(config).to(config.device)
    train_step_fn = losses_plot.get_classifier_step_fn(config.model.std, train=True, loss_type=config.training.loss)

  # Load a pretrained score model
  checkpoint = torch.load(config.training.score_restore_path, map_location=config.device)
  score_model.load_state_dict(checkpoint['model'], strict=False)

  # Initialize the optimizer
  optimizer = optim.Adam(classifier_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                          eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  
  # Build the data iterators
  ds = datasets.get_dataset(config)
  iter_ds = iter(ds)

  w = config.sampling.width
  h = config.sampling.height
  d = config.sampling.density
  x, y = np.meshgrid(np.linspace(-w, w, d, dtype=np.float32), np.linspace(-h, h, d, dtype=np.float32))
  points = torch.tensor(np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)).to(config.device)
  
  # Calculate the oracle scores
  grad_gt_upper, grad_gt_lower, grad_gt_likelihood_upper, grad_gt_likelihood_lower, grad_gt_full = get_oracle_scores(points, config)

  # Start training
  logging.info("Start training.")
  for step in range(config.training.n_iters + 1):
    # Get data
    data = next(iter_ds)
    batch = torch.from_numpy(data['position']._numpy()).to(config.device).float()
    labels = torch.from_numpy(data['label']._numpy()).to(config.device).long()
    
    # Execute one training step
    loss, loss_ce = train_step_fn(classifier_model, score_model, optimizer, batch, labels, coef=config.model.coef)

    # Plot every 100 step
    if step % 100 == 0:
      logging.info("step: %d, loss: %.5e" % (step, loss.item()))
      writer.add_scalar("loss", loss, step)
      logging.info("step: %d, loss_ce: %.5e" % (step, loss_ce.item()))
      writer.add_scalar("loss_ce", loss_ce, step)
      
      # Calculate the estimated scores
      score, grad_upper = get_conditional_scores(points, score_model, classifier_model, class_id=0,
                                                  noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
      score, grad_lower = get_conditional_scores(points, score_model, classifier_model, class_id=1,
                                                  noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
      score = score.cpu().numpy()
      grad_upper = grad_upper.cpu().numpy() * config.model.scaling_factor
      grad_lower = grad_lower.cpu().numpy() * config.model.scaling_factor
      posterior_upper = score + grad_upper
      posterior_lower = score + grad_lower

      # Calculate the expectation of DL metric
      Dp_upper = np.sqrt(((grad_gt_upper[:,0] - posterior_upper[:,0])**2) + ((grad_gt_upper[:,1] - posterior_upper[:,1])**2))
      Dp_lower = np.sqrt(((grad_gt_lower[:,0] - posterior_lower[:,0])**2) + ((grad_gt_lower[:,1] - posterior_lower[:,1])**2))
      Dl_upper = np.sqrt(((grad_gt_likelihood_upper[:,0] - grad_upper[:,0])**2) + ((grad_gt_likelihood_upper[:,1] - grad_upper[:,1])**2))
      Dl_lower = np.sqrt(((grad_gt_likelihood_lower[:,0] - grad_lower[:,0])**2) + ((grad_gt_likelihood_lower[:,1] - grad_lower[:,1])**2))
      Dl_total = 0.5*np.mean(Dl_upper) + 0.5*np.mean(Dl_lower)
      print("Expectation of Dp (upper cresent): ", np.mean(Dp_upper))
      print("Expectation of Dp (lower cresent): ", np.mean(Dp_lower))
      print("Expectation of Dl (upper cresent): ", np.mean(Dl_upper))
      print("Expectation of Dl (lower cresent): ", np.mean(Dl_lower))
      print("Expectation of total Dl: ", Dl_total)
      writer.add_scalar("Dl_upper", np.mean(Dl_upper), step)
      writer.add_scalar("Dl_lower", np.mean(Dl_lower), step)
      writer.add_scalar("Dp_upper", np.mean(Dp_upper), step)
      writer.add_scalar("Dp_lower", np.mean(Dp_lower), step)
      writer.add_scalar("total Dl", Dl_total, step)

    # Save a checkpoint periodically
    if step != 0 and step % config.training.snapshot_freq == 0:
      # Save the checkpoints
      save_step = step // config.training.snapshot_freq
      torch.save({'model': classifier_model.state_dict(),}, os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'))