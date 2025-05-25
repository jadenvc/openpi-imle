"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import torch
import itertools

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, ds, max_len):
        self._ds      = ds
        self._max_len = max_len

    def __len__(self):
        # pretend it’s only max_len long
        return min(len(self._ds), self._max_len)

    def __getitem__(self, idx):
        if idx >= self._max_len:
            raise IndexError
        return self._ds[idx]


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # print repack transform inputs and outputs
    print("Repack transform inputs:", data_config.repack_transforms.inputs, flush=True)
    print("Repack transform outputs:", data_config.repack_transforms.outputs, flush=True)

    dataset = _data_loader.create_dataset(data_config, config.model)
    print(dataset[0].keys(), flush=True)
    from openpi.transforms import compose
    repack_fn = compose(data_config.repack_transforms.inputs)
    # breakpoint()
    repacked = repack_fn(dataset[0])
    # breakpoint()
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None, debug: bool = False):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)
    print("✅ metadata loaded, entering stats loop…", flush=True)

    if max_frames:
        dataset = SubsetDataset(dataset, max_frames)

    num_frames = len(dataset)
    shuffle = False

    print(f"Number of frames: {num_frames}", flush=True)

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    print(num_frames, flush=True)

    # breakpoint()

    num_workers = 8
    if debug:
        num_workers = 0

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}


    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
