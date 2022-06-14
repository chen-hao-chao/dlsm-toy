from configs.default_toy_configs import get_default_configs

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.n_iters = 40001
  training.batch_size = 4000
  training.loss = 'dlsm'
  training.score_restore_path = './score_cond/checkpoints/checkpoint_40000.pth'

  # data
  data = config.data
  data.dataset = 'inter_twinning_moon'

  # optimize
  optim = config.optim
  optim.lr = 2e-5

  # model
  model = config.model
  model.type = 'classifier'
  model.noise_conditioned = True
  model.weighting_dlsm = 2
  model.coef = 1.0

  return config
