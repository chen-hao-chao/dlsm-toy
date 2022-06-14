import torch
from utils import get_scores, get_conditional_scores

def get_sampling_fn(config, shape):
  def langevin_dynamics(score_model, classifier_model, class_id, posterior_sm=False):
    x = torch.randn(shape).to(config.device)
    for i in range(config.sampling.num_steps):
      with torch.no_grad():
        if not posterior_sm:
          score, grad = get_conditional_scores(x, score_model, classifier_model, class_id=class_id,
                                                    noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
          posterior_score = score + grad * config.model.scaling_factor
        else:
          score = get_scores(x, score_model, 
                                  noise_conditioned=config.model.noise_conditioned, std_value=config.model.std)
          posterior_score = score

        noise = torch.randn_like(x) * config.sampling.noise_std
        step_size_noise = torch.ones(shape).to(config.device) * config.sampling.epsilon
        step_size_grad = torch.ones(shape).to(config.device) * ( (config.sampling.epsilon ** 2) / 2 )
        x = x + step_size_grad * posterior_score + step_size_noise * noise
    return x
  
  def vesde_pc_sampler(score_model, classifier_model, class_id, posterior_sm=False):
    x = torch.randn(shape).to(config.device)
    for i in range(config.sampling.num_steps):
      with torch.no_grad():
        # Predictor
        std_value_last = (config.model.std_min * (config.model.std_max / config.model.std_min) ** (i / config.sampling.num_steps))
        std_value_current = (config.model.std_min * (config.model.std_max / config.model.std_min) ** (i / config.sampling.num_steps))

        if not posterior_sm:
          score, grad = get_conditional_scores(x, score_model, classifier_model, class_id=class_id,
                                                    noise_conditioned=config.model.noise_conditioned, std_value=std_value_last)
          posterior_score = score + grad * config.model.scaling_factor
        else:
          score = get_scores(x, score_model, 
                                  noise_conditioned=config.model.noise_conditioned, std_value=std_value_last)
          posterior_score = score
        noise = torch.randn_like(x)
        std_diff = (std_value_last ** 2) - (std_value_current ** 2)
        step_size_grad = torch.ones(shape).to(config.device) * std_diff
        step_size_noise = torch.ones(shape).to(config.device) * (std_diff ** 0.5)
        x = x + step_size_grad * posterior_score + step_size_noise * noise
        
        # Corrector
        if not posterior_sm:
          score, grad = get_conditional_scores(x, score_model, classifier_model, class_id=class_id,
                                                    noise_conditioned=config.model.noise_conditioned, std_value=std_value_last)
          posterior_score = score + grad * config.model.scaling_factor
        else:
          score = get_scores(x, score_model, 
                                  noise_conditioned=config.model.noise_conditioned, std_value=std_value_last)
          posterior_score = score
        noise = torch.randn_like(x) * config.sampling.noise_std
        step_size_noise = torch.ones(shape).to(config.device) * config.sampling.epsilon
        step_size_grad = torch.ones(shape).to(config.device) * (( config.sampling.epsilon * 2 ) ** 0.5)
        x = x + step_size_grad * posterior_score + step_size_noise * noise

    return x
  
  if config.sampling.type == 'vesde_pc_sampler':
    return vesde_pc_sampler
  else:
    return langevin_dynamics