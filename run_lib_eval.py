import os
import logging

import numpy as np
import tensorflow as tf
import torch

import sampling
from utils import get_scores, get_conditional_scores, get_oracle_scores
from utils import draw_and_save_vector_field, draw_and_save_data_points, calculate_prdc
from models import simple_classifier_fn, simple_score_fn

def eval_sampling(config, workdir):
  save_plot = config.eval.save_plot

  # Create directories for figures.
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  if config.eval.post_sm:
    # Initialize the score models.
    if config.model.noise_conditioned:
      score_model_upper = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
      score_model_lower = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
    else:
      score_model_upper = simple_score_fn.simple_score_fn(config).to(config.device)
      score_model_lower = simple_score_fn.simple_score_fn(config).to(config.device)

    checkpoint = torch.load(config.sampling.score_lower_restore_path, map_location=config.device)
    score_model_lower.load_state_dict(checkpoint['model'], strict=False)
    checkpoint = torch.load(config.sampling.score_upper_restore_path, map_location=config.device)
    score_model_upper.load_state_dict(checkpoint['model'], strict=False)

    # Sample the data points according to the estimated scores.
    sampling_shape = (config.training.batch_size, 2)
    sampling_fn = sampling.get_sampling_fn(config, sampling_shape)
    samples_upper = sampling_fn(score_model_upper, classifier_model=None, class_id=0, posterior_sm=True)
    samples_lower = sampling_fn(score_model_lower, classifier_model=None, class_id=1, posterior_sm=True)

    # Calculate Precision, Recall, Density, and Coverage.
    metrics_upper, metrics_lower = calculate_prdc(config, samples_upper, samples_lower)    
    print("Class upper cresent: (class-wise PRDC)")
    print(metrics_upper)
    print("Class lower cresent: (class-wise PRDC)")
    print(metrics_lower)

    if save_plot:
      # Plot the sampled points.
      w = config.sampling.width
      h = config.sampling.height
      draw_and_save_data_points(w, h, samples_upper, samples_lower, os.path.join(sample_dir, "sampled_points.png"))

  else:
    # Initialize the score model and the classifier.
    if config.model.noise_conditioned:
      score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
      classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)
    else:
      score_model = simple_score_fn.simple_score_fn(config).to(config.device)
      classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)  

    checkpoint = torch.load(config.sampling.score_restore_path, map_location=config.device)
    score_model.load_state_dict(checkpoint['model'], strict=False)
    checkpoint = torch.load(config.sampling.classifier_restore_path, map_location=config.device)
    classifier_model.load_state_dict(checkpoint['model'], strict=False)
    
    # Sample the data points according to the estimated scores.
    sampling_shape = (config.training.batch_size, 2)
    sampling_fn = sampling.get_sampling_fn(config, sampling_shape)
    samples_upper = sampling_fn(score_model, classifier_model, class_id=0, posterior_sm=False)
    samples_lower = sampling_fn(score_model, classifier_model, class_id=1, posterior_sm=False)

    # Calculate Precision, Recall, Density, and Coverage.
    metrics_upper, metrics_lower = calculate_prdc(config, samples_upper, samples_lower)    
    print("Class upper cresent: (class-wise PRDC)")
    print(metrics_upper)
    print("Class lower cresent: (class-wise PRDC)")
    print(metrics_lower)

    if save_plot:
      # Plot the sampled points and the data points.
      w = config.sampling.width
      h = config.sampling.height
      draw_and_save_data_points(w, h, batch_upper, batch_lower, os.path.join(sample_dir, "data_points.png"))
      draw_and_save_data_points(w, h, samples_upper, samples_lower, os.path.join(sample_dir, "sampled_points.png"))
      
def eval_distance(config, workdir):
    """Calculate the score errors.
    Args:
      config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
      workdir: (str) Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Create directories for experimental logs.
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    # Evaluation configuration.
    w = config.sampling.width
    h = config.sampling.height
    d = config.sampling.density
    x, y = np.meshgrid(np.linspace(-w, w, d, dtype=np.float32), np.linspace(-h, h, d, dtype=np.float32))
    points = torch.tensor(np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)).to(config.device)
    
    # Calculate the oracle scores.
    grad_gt_upper, grad_gt_lower, grad_gt_likelihood_upper, grad_gt_likelihood_lower, grad_gt_full = get_oracle_scores(points, config)
    
    if config.eval.post_sm:
      # Initialize the score models.
      if config.model.noise_conditioned:
        score_model_upper = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
        score_model_lower = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
      else:
        score_model_upper = simple_score_fn.simple_score_fn(config).to(config.device)
        score_model_lower = simple_score_fn.simple_score_fn(config).to(config.device)

      checkpoint = torch.load(config.sampling.score_lower_restore_path, map_location=config.device)
      score_model_lower.load_state_dict(checkpoint['model'], strict=False)
      checkpoint = torch.load(config.sampling.score_upper_restore_path, map_location=config.device)
      score_model_upper.load_state_dict(checkpoint['model'], strict=False)

      # Calculate the estimated scores.
      posterior_upper = get_scores(points, score_model_upper, noise_conditioned=config.model.noise_conditioned, std_value=config.model.std).cpu().numpy()
      posterior_lower = get_scores(points, score_model_lower, noise_conditioned=config.model.noise_conditioned, std_value=config.model.std).cpu().numpy()
      
      # Calculate the expectation of the DP metric.
      Dp_upper = np.sqrt(((grad_gt_upper[:,0] - posterior_upper[:,0])**2) + ((grad_gt_upper[:,1] - posterior_upper[:,1])**2))
      Dp_lower = np.sqrt(((grad_gt_lower[:,0] - posterior_lower[:,0])**2) + ((grad_gt_lower[:,1] - posterior_lower[:,1])**2))
      logging.info("Expectation of Dp (upper cresent): %.3e" % (np.mean(Dp_upper)))
      logging.info("Expectation of Dp (lower cresent): %.3e" % (np.mean(Dp_lower)))
      
      # Plot the vector fields.
      draw_and_save_vector_field(x, y, w, h, posterior_upper, os.path.join(sample_dir, "posterior_upper.png"))
      draw_and_save_vector_field(x, y, w, h, posterior_lower, os.path.join(sample_dir, "posterior_lower.png"))

    else:
      # Initialize the score model and the classifier.
      if config.model.noise_conditioned:
        score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
        classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)
      else:
        score_model = simple_score_fn.simple_score_fn(config).to(config.device)
        classifier_model = simple_classifier_fn.simple_noise_conditioned_classifier_fn(config).to(config.device)  
      checkpoint = torch.load(config.sampling.score_restore_path, map_location=config.device)
      score_model.load_state_dict(checkpoint['model'], strict=False)    
      checkpoint = torch.load(config.sampling.classifier_restore_path, map_location=config.device)
      classifier_model.load_state_dict(checkpoint['model'], strict=False)

      # Calculate the estimated scores.
      score, grad_upper = get_conditional_scores(points, score_model, classifier_model, class_id=0,
                                                  noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
      score, grad_lower = get_conditional_scores(points, score_model, classifier_model, class_id=1,
                                                  noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
      score = score.cpu().numpy()
      grad_upper = grad_upper.cpu().numpy() * config.model.scaling_factor
      grad_lower = grad_lower.cpu().numpy() * config.model.scaling_factor
      posterior_upper = score + grad_upper
      posterior_lower = score + grad_lower

      # Calculate the expectation of the DP and DL metrics.
      D_full = np.sqrt(((grad_gt_full[:,0] - score[:,0])**2) + ((grad_gt_full[:,1] - score[:,1])**2))
      Dp_upper = np.sqrt(((grad_gt_upper[:,0] - posterior_upper[:,0])**2) + ((grad_gt_upper[:,1] - posterior_upper[:,1])**2))
      Dp_lower = np.sqrt(((grad_gt_lower[:,0] - posterior_lower[:,0])**2) + ((grad_gt_lower[:,1] - posterior_lower[:,1])**2))
      Dl_upper = np.sqrt(((grad_gt_likelihood_upper[:,0] - grad_upper[:,0])**2) + ((grad_gt_likelihood_upper[:,1] - grad_upper[:,1])**2))
      Dl_lower = np.sqrt(((grad_gt_likelihood_lower[:,0] - grad_lower[:,0])**2) + ((grad_gt_likelihood_lower[:,1] - grad_lower[:,1])**2))
      logging.info("Expectation of D (full): %.3e" % (np.mean(D_full)))
      logging.info("Expectation of Dp (upper cresent): %.3e" % (np.mean(Dp_upper)))
      logging.info("Expectation of Dp (lower cresent): %.3e" % (np.mean(Dp_lower)))
      logging.info("Expectation of Dp (upper cresent): %.3e" % (np.mean(Dl_upper)))
      logging.info("Expectation of Dp (lower cresent): %.3e" % (np.mean(Dl_lower)))

      # Plot the vector fields.
      draw_and_save_vector_field(x, y, w, h, posterior_upper, os.path.join(sample_dir, "posterior_upper.png"))
      draw_and_save_vector_field(x, y, w, h, posterior_lower, os.path.join(sample_dir, "posterior_lower.png"))
      draw_and_save_vector_field(x, y, w, h, grad_gt_upper, os.path.join(sample_dir, "grad_gt_upper.png"))
      draw_and_save_vector_field(x, y, w, h, grad_gt_lower, os.path.join(sample_dir, "grad_gt_lower.png"))

def evaluate(config, workdir):
  """Execute the evaluation procedure.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  if config.eval.type == "sampling":
    eval_sampling(config, workdir)
  elif config.eval.type == "distance":
    eval_distance(config, workdir)
  else:
    raise NotImplementedError(f"Evaluation type {config.eval.type} unknown.")
