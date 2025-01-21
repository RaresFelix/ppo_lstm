from dataclasses import dataclass

@dataclass
class Args:
    project_name: str = 'ppo_lstm'
    env_id: str = 'MiniGrid-MemoryS7_5x5-v0'
    torch_deterministic: bool = True
    total_steps: int = int(5e7) 
    seed: int = 0
    num_steps: int = 128
    betas: tuple = (0.9, 0.999)
    num_envs: int = 128
    seq_len: int = 8 # sequence length for LSTM; We cut up num_steps into pieces of seq_len
    record_video: bool = False

    minibatch_size: int = 2048
    buffer_size: int = int(512) 
    debug_probes: bool = False

    n_epochs: int = 0 # will be calculated
    batch_size: int = 0
    num_iterations: int = 0
    num_minibatches: int = 0

    update_epochs: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_range: float = 0.05
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 5e-4
    eps_max: float = 1e-3
    eps_min: float = 1e-5
    max_kl: float = 0.02

    hidden_size: int = 256
    hidden_layer_size: int = 512  # Add this new parameter
    save_freq: int = int(1048576)  # Save model every N steps
    save_dir: str = "checkpoints"  # Base directory for model checkpoints

    def __post_init__(self):
        self.n_epochs = self.total_steps // (self.num_steps * self.num_envs)

        assert self.num_steps % self.seq_len == 0, "num_steps must be divisible by seq_len"
        self.batch_size = self.num_steps * self.num_envs
        self.num_iterations = self.total_steps // self.batch_size
        self.num_minibatches = self.batch_size // (self.minibatch_size * self.seq_len)

        assert self.num_iterations != 0, "num_iterations must be greater than 0"
        assert self.num_minibatches != 0, "num_minibatches must be greater than 0"
        assert self.batch_size % (self.minibatch_size * self.seq_len) == 0, "batch_size must be divisible by minibatch_size * seq_len"