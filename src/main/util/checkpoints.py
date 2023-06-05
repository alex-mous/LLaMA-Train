"""
Load and save checkpoints for model and optimizer
"""

import os


def generate_checkpoint_name(checkpoints_base_path: str, epoch: int, new_type: bool):
    """
    Generate a checkpoint name for the given model type and epoch

    :param checkpoints_base_path: Path to directory to store checkpoints in
    :param model: PyTorch model
    :return: Checkpoint path and name
    """
    return os.path.join(checkpoints_base_path, f"chkpt-{epoch}" + ("-light" if new_type else "") + ".pt")
