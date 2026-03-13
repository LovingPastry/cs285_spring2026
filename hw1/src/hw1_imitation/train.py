"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger,evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 800
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None
    # Optional path to a saved checkpoint .pkl file for continued training.
    resume_ckpt_path: Path | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    if config.resume_ckpt_path is not None:
        loaded = torch.load(
            config.resume_ckpt_path,
            map_location=device,
            weights_only=False,
        )
        if not isinstance(loaded, torch.nn.Module):
            raise TypeError(
                f"Expected a torch.nn.Module in checkpoint, got {type(loaded)!r}"
            )
        model = loaded.to(device)
        print(f"Resumed model from checkpoint: {config.resume_ckpt_path}")

    # Compile the model for faster training
    model = torch.compile(model)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    import torch.optim as optim
    import time
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    step = 0
    if config.resume_ckpt_path is not None:
        step_match = re.search(r"checkpoint_step_(\d+)\.pkl$", config.resume_ckpt_path.name)
        if step_match is not None:
            step = int(step_match.group(1))
            print(f"Resumed global step from checkpoint filename: {step}")

    for epoch in range(config.num_epochs):
        start_epoch_time = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        
        for state, action_chunk in loader:
            start_batch_time = time.time()
            
            # Move data to device
            state = state.to(device)
            action_chunk = action_chunk.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.compute_loss(state, action_chunk)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track batch statistics
            batch_time = time.time() - start_batch_time
            samples_per_second = config.batch_size / batch_time
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Log training metrics
            if step % config.log_interval == 0:
                row = {
                    'loss': loss.item(),
                    'epoch': epoch,
                    'step': step,
                    'samples_per_second': samples_per_second,
                    'batch_time': batch_time,
                    'learning_rate': config.lr
                }
                logger.log(row, step)
                wandb.log(row, step=step)
            
            # Run evaluation
            if step % config.eval_interval == 0:
                evaluate_policy(
                    model,
                    normalizer,
                    device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=step,
                    logger=logger
                )
            
            step += 1
        
        # Log epoch-level metrics
        epoch_time = time.time() - start_epoch_time
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        epoch_samples_per_second = (len(loader) * config.batch_size) / epoch_time
        
        epoch_row = {
            'epoch_loss': avg_epoch_loss,
            'epoch_time': epoch_time,
            'epoch_samples_per_second': epoch_samples_per_second,
            'epoch': epoch
        }
        logger.log(epoch_row, step)
        wandb.log(epoch_row, step=step)

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
