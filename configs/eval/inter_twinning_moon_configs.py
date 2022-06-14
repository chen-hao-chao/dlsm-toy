from configs.default_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.batch_size = 10000

  # sampling
  sampling = config.sampling
  sampling.classifier_restore_path = './classifier_cond_ce/checkpoints/checkpoint_40000.pth'
  sampling.score_restore_path = './score_cond/checkpoints/checkpoint_40000.pth'
  sampling.score_upper_restore_path = './score_cond_upper/checkpoints/checkpoint_40000.pth'
  sampling.score_lower_restore_path = './score_cond_lower/checkpoints/checkpoint_40000.pth'
  sampling.type = 'vesde_pc_sampler'
  sampling.noise_std = 1.0
  sampling.num_steps = 1000
  sampling.epsilon = 3.5

  # data
  data = config.data
  data.dataset = 'inter_twinning_moon'

  # model
  model = config.model
  model.noise_conditioned = True
  model.scaling_factor = 1

  # Evaluation
  eval = config.eval 
  eval.type = "distance"
  eval.post_sm = False
  eval.save_plot = False

  return config
