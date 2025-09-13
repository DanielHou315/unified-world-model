import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.file_utils import glob_all
from datasets.utils.sampler import TrajectorySampler
from datasets.utils.obs_utils import unflatten_obs


class FactoryDataset(Dataset):
    def __init__(
        self,
        name: str,
        hdf5_path_globs: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        val_ratio: float = 0.0,
        subsample_ratio: float = 1.0,
        flip_rgb: bool = False,
        max_episodes: int = 10000,
    ):
        self.name = name
        self.seq_len = seq_len
        self.flip_rgb = flip_rgb

        # Parse observation and action shapes
        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self._action_shape = tuple(shape_meta["action"]["shape"])
        # Compressed buffer to store episode data
        self.buffer = self._init_buffer(hdf5_path_globs, buffer_path, max_episodes)

        # Create training-validation split
        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.val_mask = val_mask
        self.train_mask = ~val_mask

        # Apply subsample_ratio to training episodes
        if subsample_ratio < 1.0 or max_episodes < num_episodes:
            train_indices = np.where(self.train_mask)[0]
            num_train_episodes = len(train_indices)
            num_subsampled = round(num_train_episodes * subsample_ratio)
            num_subsampled = max(1, num_subsampled)  # Ensure at least one episode
            num_subsampled = min(num_subsampled, max_episodes)
            # Create a new mask with only the subsampled training episodes
            subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
            rng = np.random.default_rng(seed=1)
            sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
            subsampled_train_mask[sampled_indices] = True
            self.train_mask = subsampled_train_mask

        # Sampler to draw sequences from buffer
        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

    def _init_buffer(self, hdf5_path_globs, buffer_path, max_episodes):
        hdf5_paths = glob_all(hdf5_path_globs)

        # Create metadata
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        # Compute buffer capacity
        capacity = 0
        num_episodes = 0
        filtered_hdf5_paths = []
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                trial_success = f["trial_success"][:]
                if f["rgb"].ndim == 4:
                    if trial_success == 0:
                        print(f"prefiltering: Trial {hdf5_path} failed, skipping")
                        continue
                    this_traj_len = f["rgb"].shape[0]
                    num_episodes += 1
                    capacity +=  this_traj_len
                    filtered_hdf5_paths.append(hdf5_path)
                elif f["rgb"].ndim == 5:
                    for i in range(f["rgb"].shape[0]):
                        if trial_success[i] == 0:
                            print(f"prefiltering: Trial {hdf5_path} {i} failed, skipping")
                            continue
                        this_traj_len = f["rgb"].shape[1]
                        num_episodes += 1
                        capacity +=  this_traj_len
                        filtered_hdf5_paths.append(hdf5_path)
                if num_episodes >= max_episodes:
                    break
        print("Total episodes: ", num_episodes)
        print("Total capacity: ", capacity)
        # Initialize buffer
        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )

        # If buffer is restored from disk, return it
        if buffer.restored:
            return buffer
        def filter_action_ema(actions):
            ema_factor = 0.2
            ema_actions = actions.copy()
            for i in range(actions.shape[1]-1):
                ema_actions[i+1] = ema_factor * actions[i+1] + (1-ema_factor) * ema_actions[i]
            return ema_actions

        # Otherwise, load episodes to buffer
        pbar = tqdm(total=num_episodes, desc="Loading episodes to buffer")
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                trial_success = f["trial_success"][:]
                
                if f["rgb"].ndim == 4:
                    # Single trajectory per file
                    if trial_success == 0:
                        print(f"Trial {hdf5_path} failed, skipping")
                        continue
                        
                    episode = {}
                    for key in self._image_shapes.keys():
                        if self.flip_rgb:
                            episode[f"obs.{key}"] = f[key][:, ::-1]
                        else:
                            episode[f"obs.{key}"] = f[key][:]
                    for key in self._lowdim_shapes.keys():
                        episode[f"obs.{key}"] = f[key][:]
                    # if "gear" in self.name:
                    #     print("filtering action")
                    #     actions = filter_action_ema(f["action"][:])
                    #     episode["action"] = actions
                    # else:
                    episode["action"] = f["action"][:]
                    buffer.add_episode(episode)
                    pbar.update(1)
                    
                elif f["rgb"].ndim == 5:
                    # Multiple trajectories per file
                    for i in range(f["rgb"].shape[0]):
                        if trial_success[i] == 0:
                            print(f"Trial {hdf5_path} {i} failed, skipping")
                            continue
                            
                        episode = {}
                        for key in self._image_shapes.keys():
                            if self.flip_rgb:
                                episode[f"obs.{key}"] = f[key][i, :, ::-1]
                            else:
                                episode[f"obs.{key}"] = f[key][i, :]
                        for key in self._lowdim_shapes.keys():
                            episode[f"obs.{key}"] = f[key][i, :]
                        # if "gear" in self.name:
                        #     print("filtering action")
                        #     actions = filter_action_ema(f["action"][i, :])
                        #     episode["action"] = actions
                        # else:
                        episode["action"] = f["action"][i, :]
                        buffer.add_episode(episode)
                        pbar.update(1)
                if buffer.num_episodes >= max_episodes:
                    print(f"Reached max episodes: {max_episodes}")
                    break
        pbar.close()
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return f"<FactoryDataset>\nname: {self.name}\nnum_samples: {len(self)}\n{self.buffer}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Sample a sequence of observations and actions from the dataset.
        data = self.sampler.sample_sequence(idx)

        # Convert data to torch tensors
        data = {k: torch.from_numpy(v) for k, v in data.items()}

        # Unflatten observations
        data = unflatten_obs(data)
        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, self.val_mask)
        return val_set
