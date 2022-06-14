import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.n_iters = 40001
  training.batch_size = 2000
  training.snapshot_freq = 5000
  training.log_freq = 500

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.type = 'vesde_pc_sampler'
  sampling.noise_std = 1.0
  sampling.num_steps = 1000
  sampling.epsilon = 0.01
  sampling.width = 40
  sampling.height = 25
  sampling.density = 35

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'inter_twinning_moon'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.type = 'score_model'
  model.nf = 16
  model.std = 7.5
  model.std_max = 10.0
  model.std_min = 1e-3
  model.classes = 2
  model.weighting_dsm = 2
  model.weighting_dlsm = 2
  model.weighting_ce = 0
  model.coef = 1.0
  model.scaling_factor = 1

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.lr = 5e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8

  # Evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.type = "sampling"

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config