from configs.default_toy_configs import get_default_configs

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.n_iters = 40001
  training.batch_size = 4000
  training.snapshot_freq = 5000

  # sampling
  sampling = config.sampling
  sampling.conditional = False
  sampling.classifier_restore_path = None
  sampling.score_restore_path = None
  sampling.num_steps = 50

  # data
  data = config.data
  data.dataset = 'inter_twinning_moon_upper'

  # optimization
  optim = config.optim
  optim.lr = 6.5e-4

  # model
  model = config.model
  model.type = 'score_model'
  model.noise_conditioned = True
  model.weighting_dsm = 2

  return config
