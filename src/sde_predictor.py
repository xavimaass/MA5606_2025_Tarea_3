import abc
import torch
import math

## Euler Maruyama Predictor 
class PredictorPT(abc.ABC):
    def __init__(self, sde_obj):
        self.sde = sde_obj

    @abc.abstractmethod
    def update_fn(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

class EulerMaruyamaPredictorPT(PredictorPT):
    def __init__(self, sde_obj):
        super().__init__(sde_obj)

    def update_fn(self, t, x): # t is current time, x is current state
        dt = 1.0 / self.sde.N # Timestep size (reverse SDE uses N steps over T)

        # Get drift and diffusion from the SDE (could be reverse SDE)
        # model prediction (score) is handled inside sde.sde_coeff if it's an RSDE instance
        drift, diffusion_g = self.sde.sde_coeff(t, x)

        z = torch.randn_like(x)
        x_mean = x + dt * drift # Euler step for mean
        x_next = x_mean + diffusion_g * math.sqrt(dt) * z # Add noise scaled by g(t) and sqrt(dt)

        return x_next, x_mean