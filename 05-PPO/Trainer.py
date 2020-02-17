from torch import optim
from Model import *
from typing import Dict, List
class Trainer:
    def __init__(self, model: Model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    def train(self,
              samples: Dict[str, np.ndarray],
              learning_rate: float,
              clip_range: float):
        sampled_obs = samples['obs']
        sampled_action = samples['actions']
        sampled_return = samples['values'] + samples['advantages']
        sampled_normalized_advantage = Trainer._normalize(samples['advantages'])
        sampled_neg_log_pi = samples['neg_log_pis']
        sampled_value = samples['values']
        pi, value = self.model(sampled_obs)
        neg_log_pi = -pi.log_prob(sampled_action)
        ratio: torch.Tensor = torch.exp(sampled_neg_log_pi - neg_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()
        loss: torch.Tensor = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        approx_kl_divergence = .5 * ((neg_log_pi - sampled_neg_log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_reward,
                vf_loss,
                entropy_bonus,
                approx_kl_divergence,
                clip_fraction]

    @staticmethod
    def _normalize(adv: np.ndarray):
        return (adv - adv.mean()) / (adv.std() + 1e-8)