"""Custom Adam optimizer with optional bias correction toggle.

Implements the Adam algorithm (Kingma & Ba, 2015) with a flag to
disable bias correction, used for the Figure 4 ablation study.
"""
import math
import torch
from torch.optim import Optimizer


class AdamUncorrected(Optimizer):
    """Adam optimizer with configurable bias correction.

    This implementation follows Algorithm 1 from Kingma & Ba (2015) but adds
    a ``bias_correction`` flag that, when set to False, skips the bias
    correction step (lines 3-4 of the algorithm). This is used to demonstrate
    the importance of bias correction in the paper's Figure 4 ablation.

    Args:
        params: Iterable of parameters to optimize.
        lr (float): Learning rate (alpha). Default: 1e-3.
        betas (Tuple[float, float]): Coefficients for computing running
            averages of gradient (beta1) and its square (beta2).
            Default: (0.9, 0.999).
        eps (float): Term added to denominator for numerical stability.
            Default: 1e-8.
        bias_correction (bool): If True, apply bias correction to moment
            estimates. If False, use raw (biased) estimates. Default: True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, bias_correction=bias_correction)
        super(AdamUncorrected, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else:
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
