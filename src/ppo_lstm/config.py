from dataclasses import dataclass

@dataclass
class Args:
    project_name: str = 'ppo_lstm'
    env_id: str = 'CartPole-v1'
    torch_deterministic: bool = True
    total_steps: int = int(1e6) 
    seed: int = 0
    num_steps: int = 256
    num_envs: int = 16
    seq_len: int = 16 # sequence length for LSTM; We cut up num_steps into pieces of seq_len

    minibatch_size: int = 256
    buffer_size: int = int(1e5) 
    debug_probes: bool = False

    n_epochs: int = 0 # will be calculated
    batch_size: int = 0
    num_iterations: int = 0
    num_minibatches: int = 0

    update_epochs: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.05
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 2e-3
    eps_max: float = 1e-3
    eps_min: float = 1e-5
    max_kl: float = 0.015

    hidden_size: int = 64

    def __post_init__(self):
        self.n_epochs = self.total_steps // (self.num_steps * self.num_envs)
        self.batch_size = self.num_steps * self.num_envs
        self.num_iterations = self.total_steps // self.batch_size
        self.num_minibatches = self.batch_size // (self.minibatch_size * self.num_envs)
        assert self.num_steps % self.minibatch_size == 0, "num_steps must be divisible by minibatch_size"