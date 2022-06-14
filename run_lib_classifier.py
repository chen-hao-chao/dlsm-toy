import os
import logging
import numpy as np

import datasets
import losses
from models import simple_classifier_fn, simple_score_fn
from utils import plot_vector_field_likelihood

import torch
import tensorflow as tf
import torch.optim as optim

def train(config, workdir):
  """Execute the training procedure for the classifier.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs.
  visualization_dir = os.path.join(workdir, "visualization")
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tf.io.gfile.makedirs(visualization_dir)
  tf.io.gfile.makedirs(checkpoint_dir)
  
  # Initialize the classifier and the score model.
  if config.model.noise_conditioned:
    classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)
    score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
    train_step_fn = losses.get_classifier_step_fn((config.model.std_max, config.model.std_min), train=True,
                                                    loss_type=config.training.loss, conditioned=True,
                                                      weighting_dlsm=config.model.weighting_dlsm, weighting_ce=config.model.weighting_ce,
                                                        coef=config.model.coef)
  else:
    classifier_model = simple_classifier_fn.simple_classifier_fn(config).to(config.device)
    score_model = simple_score_fn.simple_score_fn(config).to(config.device)
    train_step_fn = losses.get_classifier_step_fn(config.model.std, train=True, loss_type=config.training.loss, coef=config.model.coef)

  # Load a pretrained score model.
  checkpoint = torch.load(config.training.score_restore_path, map_location=config.device)
  score_model.load_state_dict(checkpoint['model'], strict=False)

  # Initialize the optimizer.
  optimizer = optim.Adam(classifier_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                          eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  
  # Build the data iterators.
  ds = datasets.get_dataset(config)
  iter_ds = iter(ds)

  # Start training.
  logging.info("Start training.")
  for step in range(config.training.n_iters + 1):
    # Get data.
    data = next(iter_ds)
    batch = torch.from_numpy(data['position']._numpy()).to(config.device).float()
    labels = torch.from_numpy(data['label']._numpy()).to(config.device).long()
    
    # Execute one training step.
    loss = train_step_fn(classifier_model, score_model, optimizer, batch, labels)

    # Print the loss periodically.
    if step % config.training.log_freq == 0:
      logging.info("step: %d, loss: %.3e" % (step, loss.item()))

    # Save a checkpoint periodically.
    if step != 0 and step % config.training.snapshot_freq == 0:
      # Save the checkpoint and plot the vector field.
      save_step = step // config.training.snapshot_freq
      torch.save({'model': classifier_model.state_dict(),}, os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'))
      plot_vector_field_likelihood(config, classifier_model, os.path.join(visualization_dir, "vf_"+str(save_step)+"_upper_cresent.png"),
                                    w=40, h=25, density=35, class_id=0, noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
      plot_vector_field_likelihood(config, classifier_model, os.path.join(visualization_dir, "vf_"+str(save_step)+"_lower_cresent.png"),
                                    w=40, h=25, density=35, class_id=1, noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)