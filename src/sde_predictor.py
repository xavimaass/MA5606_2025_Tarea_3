import abc
import torch
from torch.utils.data import Dataset
from sde import SDE

## Euler Maruyama Predictor 
class PredictorPT(abc.ABC):
    def __init__(self, sde_obj):
        self.sde = sde_obj

    @abc.abstractmethod
    def update_fn(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class SamplerPT:
    def __init__(self, eps: float): # eps: stopping time for backward diffusion
        self.eps = eps

    def get_sampling_fn(self, sde_obj: SDE, initial_dataset_for_sampling: Dataset, predictor_obj: PredictorPT):

        update_fn = predictor_obj.update_fn

        def sampling_fn(num_samples: int, device_):
            if isinstance(initial_dataset_for_sampling.dimension, int): # 2D case
                shape_sample = (num_samples, initial_dataset_for_sampling.dimension)
            elif isinstance(initial_dataset_for_sampling.dimension, tuple): # Image case
                shape_sample = (num_samples, *initial_dataset_for_sampling.dimension)
            else:
                raise TypeError("Dataset dimension type not supported for prior_sampling.")

            if num_samples <= len(initial_dataset_for_sampling):
                 y_current = torch.Tensor(initial_dataset_for_sampling[:num_samples]).to(device_)
                 if y_current.shape[0] != num_samples: # If dataset slicing didn't work as expected
                     y_current = sde_obj.prior_sampling(shape_sample).to(device_) # Fallback to SDE prior
            else: # Not enough samples in dataset, use SDE's prior sampling
                print(f"Warning: Requested {num_samples} samples, but dataset has {len(initial_dataset_for_sampling)}. Using SDE prior sampling.")
                y_current = sde_obj.prior_sampling(shape_sample).to(device_)


            timesteps_val = torch.linspace(0, sde_obj.T - self.eps, sde_obj.N, device=device_)

            x_hist_list = [] # Store history of samples

            for i in range(sde_obj.N):
                t_current = timesteps_val[i]
                # update_fn expects t as (batch_size, 1) or scalar to be broadcast
                vec_t = torch.ones(y_current.shape[0], device=device_) * t_current

                y_next, _ = update_fn(vec_t, y_current) # Pass vec_t instead of t_current directly
                y_current = y_next

                if i % (sde_obj.N // 50) == 0 or i == sde_obj.N - 1 : # Store some history
                    x_hist_list.append(y_current.cpu().clone())

            x_hist_tensor = torch.stack(x_hist_list) if x_hist_list else torch.empty(0)

            return y_current.cpu(), sde_obj.N, timesteps_val.cpu(), x_hist_tensor

        return sampling_fn
