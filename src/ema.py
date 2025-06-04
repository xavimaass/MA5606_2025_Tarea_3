
# EMA (Exponential Moving Average) for model parameters
# PyTorch doesn't have a built-in generic EMA updater like Optax or Haiku's recipes often show.
# We can implement it manually or use a utility if available.
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {} # Stores EMA parameters

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, target_model): # Apply EMA params to a target model
        for name, param in target_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name]

    def copy_to_model(self, target_model): # For when you want to use the EMA model
        self.apply_shadow(target_model)