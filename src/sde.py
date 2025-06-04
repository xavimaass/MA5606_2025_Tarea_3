import abc
from typing import Any
import torch

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


class OrnsteinUhlenbeck(SDE):
    def __init__(self, N=100, device='cpu'):
        super().__init__(N)
        self.device = device

    @property
    def T(self):
        return 1.0 # End time for the SDE

    def _get_t_tensor(self, t, x0):
        target_ndim = x0.ndim
        
        if isinstance(t, (float, int)):
            # If t is a scalar (float/int), create a tensor of the correct shape
            shape_for_t = list(x0.shape[:1]) + [1] * (target_ndim - 1)
            return torch.full(shape_for_t, float(t), device=x0.device, dtype=x0.dtype)
            
        if torch.is_tensor(t):
            if t.ndim == 0: # scalar tensor (e.g., torch.tensor(0.5))
                shape_for_t = list(x0.shape[:1]) + [1] * (target_ndim - 1)
                return t.expand(shape_for_t)
                
            elif t.ndim == 1 and t.shape[0] == x0.shape[0]:
                # Reshape (batch,) to (batch, 1, 1, ...) based on x0's dimensions
                new_shape = list(t.shape) + [1] * (target_ndim - 1)
                return t.reshape(new_shape)
                
            elif t.ndim == target_ndim: # Already has the same number of dimensions as x0
                return t
            else:
                # Fallback for unexpected t shapes, could raise an error or try to unsqueeze
                # For safety, let's explicitly add dimensions until it matches x0's ndim
                t_tensor = t
                while t_tensor.ndim < target_ndim:
                    t_tensor = t_tensor.unsqueeze(-1)
                # Ensure the batch dimension matches if it's the only one that doesn't
                if t_tensor.shape[0] != x0.shape[0] and t_tensor.shape[0] == 1:
                    return t_tensor.expand(x0.shape[0], *t_tensor.shape[1:])
                elif t_tensor.shape[0] != x0.shape[0]:
                    raise ValueError(f"Batch dimension mismatch after preparing t_tensor: {t_tensor.shape} vs {x0.shape}")
        raise TypeError("t must be a float, int, or a torch.Tensor")

    def sde_coeff(self, t, x):
        drift = -0.5 * x
        # Diffusion is constant 1, so g(t) = 1.
        # It needs to have the same shape as x for element-wise operations later,
        # or be broadcastable. For safety, make it same shape.
        diffusion = torch.ones_like(x)
        return drift, diffusion

    def marginal_prob(self, t, x0):
        # Ensure t is a tensor, (batch_size, 1)
        t_tensor = self._get_t_tensor(t, x0)

        # For dX = - (1/2) X dt + dB_t  (alpha = 1/2, beta = 1 from Sohl-Dickstein notation for variance preserving)
        mean_coeff = torch.exp(-0.5 * t_tensor)
        mean = mean_coeff * x0

        std_val = torch.sqrt(1.0 - torch.exp(-1.0 * t_tensor))
        ones_for_std = torch.ones_like(x0)
        std = std_val * ones_for_std
        return mean, std

    def prior_sampling(self, shape):
        # Samples from N(0, I) as X_T should be close to stationary distribution if T is large enough.
        # For OU dX = - (1/2)X dt + dB_t, stationary dist is N(0,1).
        return torch.randn(shape, device=self.device)