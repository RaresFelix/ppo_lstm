from dataclasses import dataclass, asdict

@dataclass
class Args:
    # Project and Environment Configuration
    project_name: str = 'ppo_lstm'
    env_id: str = 'MiniGrid-MemoryS9-v0'
    view_size: int = 3  # 0 for full observation
    one_hot: bool = True  # use one-hot encoding for partial observation
    use_pixels: bool = False  # will make env output pixels & use CNNs
    torch_deterministic: bool = True
    seed: int = 0

    use_wandb: bool = True 
    wandb_project: str = 'ppo_lstm'
    wandb_group: str = 'sweeps_3x3_S9_random'

    # Core Training Parameters
    total_steps: int = int(2e6)
    num_envs: int = 64
    num_steps: int = 128
    seq_len: int = 8  # sequence length for LSTM
    update_epochs: int = 64

    rand_move_eps: float = 5e-3# frequency of completely random moves

    # Network Architecture
    hidden_size: int = 8
    hidden_layer_size: int = 64

    # Optimization Parameters
    learning_rate: float = 2e-3
    betas: tuple = (0.9, 0.999)
    max_grad_norm: float = 0.5

    # PPO Specific Parameters
    clip_range: float = 0.2
    max_kl: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    entropy_coef: float = 0.0001

    # Buffer and Batch Settings
    buffer_size: int = int(512)
    minibatch_size: int = 1024

    # Logging and Saving
    debug_probes: bool = False
    record_video: bool = False
    save_freq: int = int(1048576)
    save_dir: str = "checkpoints"

    # Derived Parameters (calculated in post_init)
    n_epochs: int = 0
    batch_size: int = 0
    num_iterations: int = 0
    num_minibatches: int = 0

    gpu_id: int = 0

    ema_decay: float = 0.995  # decay rate for exponential moving average of returns

    def __post_init__(self):
        self.n_epochs = self.total_steps // (self.num_steps * self.num_envs)

        assert self.num_steps % self.seq_len == 0, "num_steps must be divisible by seq_len"
        self.batch_size = self.num_steps * self.num_envs
        self.num_iterations = self.total_steps // self.batch_size
        self.num_minibatches = self.batch_size // (self.minibatch_size * self.seq_len)

        assert self.num_iterations != 0, "num_iterations must be greater than 0"
        assert self.num_minibatches != 0, "num_minibatches must be greater than 0"
        assert self.batch_size % (self.minibatch_size * self.seq_len) == 0, "batch_size must be divisible by minibatch_size * seq_len"

    @classmethod
    def from_wandb_config(cls, config):
        # Convert wandb config dict to Args instance
        # Only include keys that exist in Args
        valid_params = {k: v for k, v in config.items() if k in cls.__dataclass_fields__}
        return cls(**valid_params)