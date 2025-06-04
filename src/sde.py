import abc
from typing import Any
import torch
import math

## SDE abstract class
class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""
    def __init__(self, N: int):
        super().__init__()
        self.N = N # Number of discretization time steps

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        """Return drift and diffusion coefficients at time t for state x."""
        pass

    @abc.abstractmethod
    def marginal_prob(self, t, x0):
        """Parameters to determine the marginal distribution of the SDE, p_t(x|x_0).
           Returns mean and std of x_t given x_0.
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, p_T(x)."""
        pass

    def reverse(self, model: torch.nn.Module) -> Any:
        """Create the reverse-time SDE.
        Args:
          model (torch.nn.Module): The score model.
        Returns:
          sde_backward (Any): The backward SDE.
        """
        N = self.N
        T_end = self.T
        sde_coeff_orig = self.sde_coeff 

        def get_reverse_drift_fn(score_model_fn):
            def reverse_drift_fn(t, y):
                # y is the state of the reverse process at time t (reverse time)
                # Corresponds to x at time T_end - t (forward time)
                forward_t = T_end - t
                drift_orig, diffusion_orig = sde_coeff_orig(forward_t, y)

                with torch.no_grad(): # Score prediction should not compute gradients here
                    score = score_model_fn(forward_t, y) # model(t,x) already

                reverse_drift = -drift_orig + (diffusion_orig ** 2) * score
                return reverse_drift
            return reverse_drift_fn

        class RSDE(self.__class__):
            def __init__(self_rsde):
                self_rsde.N = N
                self_rsde.model_fn = model # The PyTorch model
                self_rsde.reverse_drift_fn = get_reverse_drift_fn(self_rsde.model_fn)
                self_rsde.param_drift = 'score' # model estimates the score

            @property
            def T(self_rsde):
                return T_end

            def sde_coeff(self_rsde, t, y): # t is reverse time, y is state
                # Diffusion coefficient is the same as forward SDE at time T-t
                _, diffusion = sde_coeff_orig(T_end - t, y)
                drift = self_rsde.reverse_drift_fn(t, y)
                return drift, diffusion

            def marginal_prob(self, t, x0): # Should not be called on RSDE usually
                raise NotImplementedError("marginal_prob not defined for RSDE")

            def prior_sampling(self, shape): # Sampling initial Y_0 from p_T (forward)
                # This is the prior of the FORWARD SDE, used to start the REVERSE SDE
                return super().prior_sampling(shape)


        sde_backward = RSDE()
        return sde_backward


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