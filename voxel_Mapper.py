import copy
import functools
import threading
from pathlib import Path

import open3d as o3d
import torch
import logging
import torch.optim
import numpy as np
import numba
import argparse
import torch.multiprocessing as mp


@numba.jit
def _get_valid_idx(base_idx: np.ndarray, query_idx: np.ndarray):
    mask = np.zeros((base_idx.shape[0],), dtype=np.bool_)
    for vi, v in enumerate(base_idx):
        if query_idx[np.searchsorted(query_idx, v)] != v:
            mask[vi] = True
    return mask


class MeshExtractCache:
    def __init__(self, device):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.device = device
        self.clear_updated_vec()

    def clear_updated_vec(self):
        self.updated_vec_id = torch.empty((0,), device=self.device, dtype=torch.long)

    def clear_all(self):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.clear_updated_vec()


class MapVisuals:
    def __init__(self):
        self.mesh = []
        self.blocks = []
        self.samples = []
        self.uncertainty = []


class DenseIndexedMap:
    def __init__(self, args: argparse.Namespace, device: torch.device, latent_dim: int):
        """
        Initialize a densely indexed latent map.
        For easy manipulation, invalid indices are -1, and occupied indices are >= 0.

        :param latent_dim:  size of latent dim
        :param device:      device type of the map (some operations can still on host)
        """
        mp.set_start_method('forkserver', force=True)

        self.voxel_size = args.voxel_size
        self.n_xyz = np.ceil((np.asarray(args.bound_max) - np.asarray(args.bound_min)) / args.voxel_size).astype(
            int).tolist()
        logging.info(f"Map size Nx = {self.n_xyz[0]}, Ny = {self.n_xyz[1]}, Nz = {self.n_xyz[2]}")

        self.args = args
        self.bound_min = torch.tensor(args.bound_min, device=device).float()
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz, device=device)
        self.latent_dim = latent_dim
        self.device = device
        self.integration_offsets = [torch.tensor(t, device=self.device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        # Directly modifiable from outside.
        self.extract_mesh_std_range = None

        self.mesh_update_affected = [torch.tensor([t], device=self.device)
                                     for t in [[-1, 0, 0], [1, 0, 0],
                                               [0, -1, 0], [0, 1, 0],
                                               [0, 0, -1], [0, 0, 1]]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float32)

        self.cold_vars = {
            "n_occupied": 0,
            "indexer": torch.ones(np.product(self.n_xyz), device=device, dtype=torch.long) * -1,
            # -- Voxel Attributes --
            # 1. Latent Vector (Geometry)
            "latent_vecs": torch.empty((1, self.latent_dim), dtype=torch.float32, device=device),
            # 2. Position
            "latent_vecs_pos": torch.ones((1,), dtype=torch.long, device=device) * -1,
            # 3. Confidence on its geometry
            "voxel_obs_count": torch.zeros((1,), dtype=torch.float32, device=device),
            # 4. Optimized mark
            "voxel_optimized": torch.zeros((1,), dtype=torch.bool, device=device)
        }
        self.backup_var_names = ["indexer", "latent_vecs", "latent_vecs_pos", "voxel_obs_count"]
        self.backup_vars = {}
        self.modifying_lock = threading.Lock()
        # Allow direct visit by variable
        for p in self.cold_vars.keys():
            setattr(DenseIndexedMap, p, property(
                fget=functools.partial(DenseIndexedMap._get_var, name=p),
                fset=functools.partial(DenseIndexedMap._set_var, name=p)
            ))

        self.meshing_thread = None
        self.meshing_thread_id = -1
        self.meshing_stream = torch.cuda.Stream()
        self.mesh_cache = MeshExtractCache(self.device)
        self.latent_vecs.zero_()

    # def __del__(self):
    #     self.optimize_process.kill()

    def save(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('wb') as f:
            torch.save(self.cold_vars, f)

    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('rb') as f:
            self.cold_vars = torch.load(f)

    def _get_var(self, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            return self.backup_vars[name]
        else:
            return self.cold_vars[name]

    def _set_var(self, value, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            self.backup_vars[name] = value
        else:
            self.cold_vars[name] = value

    def _inflate_latent_buffer(self, count: int):
        target_n_occupied = self.n_occupied + count
        if self.latent_vecs.size(0) < target_n_occupied:
            new_size = self.latent_vecs.size(0)
            while new_size < target_n_occupied:
                new_size *= 2
            new_vec = torch.empty((new_size, self.latent_dim), dtype=torch.float32, device=self.device)
            new_vec[:self.latent_vecs.size(0)] = self.latent_vecs
            new_vec_pos = torch.ones((new_size,), dtype=torch.long, device=self.device) * -1
            new_vec_pos[:self.latent_vecs.size(0)] = self.latent_vecs_pos
            new_voxel_conf = torch.zeros((new_size,), dtype=torch.float32, device=self.device)
            new_voxel_conf[:self.latent_vecs.size(0)] = self.voxel_obs_count
            new_voxel_optim = torch.zeros((new_size,), dtype=torch.bool, device=self.device)
            new_voxel_optim[:self.latent_vecs.size(0)] = self.voxel_optimized
            new_vec[self.latent_vecs.size(0):].zero_()
            self.latent_vecs = new_vec
            self.latent_vecs_pos = new_vec_pos
            self.voxel_obs_count = new_voxel_conf
            self.voxel_optimized = new_voxel_optim

        new_inds = torch.arange(self.n_occupied, target_n_occupied, device=self.device, dtype=torch.long)
        self.n_occupied = target_n_occupied
        return new_inds

    def _linearize_id(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        return xyz[:, 2] + self.n_xyz[-1] * xyz[:, 1] + (self.n_xyz[-1] * self.n_xyz[-2]) * xyz[:, 0]

    def _unlinearize_id(self, idx: torch.Tensor):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        return torch.stack([idx // (self.n_xyz[1] * self.n_xyz[2]),
                            (idx // self.n_xyz[2]) % self.n_xyz[1],
                            idx % self.n_xyz[2]], dim=-1)

    def _mark_updated_vec_id(self, new_vec_id: torch.Tensor):
        """
        :param new_vec_id: (B,) updated id (indexed in latent vectors)
        """
        self.mesh_cache.updated_vec_id = torch.cat([self.mesh_cache.updated_vec_id, new_vec_id])
        self.mesh_cache.updated_vec_id = torch.unique(self.mesh_cache.updated_vec_id)

    def allocate_block(self, idx: torch.Tensor):
        """
        :param idx: (N, 3) or (N, ), if the first one, will call linearize id.
        NOTE: this will not check index overflow!
        """
        if idx.ndimension() == 2 and idx.size(1) == 3:
            idx = self._linearize_id(idx)
        new_id = self._inflate_latent_buffer(idx.size(0))
        self.latent_vecs_pos[new_id] = idx
        self.indexer[idx] = new_id

    STATUS_CONF_BIT = 1 << 0  # 1
    STATUS_SURF_BIT = 1 << 1  # 2

    def integrate_keyframe(self, surface_xyz: torch.Tensor, do_optimize: bool = False, async_optimize: bool = False):
        """
        :param surface_xyz:  (N, 3) x, y, z
        :param do_optimize: whether to do optimization (this will be slow though)
        :param async_optimize: whether to spawn a separate job to optimize.
            Note: the optimization is based on the point at this function call.
                  optimized result will be updated on the next function call after it's ready.
            Caveat: If two optimization thread are started simultaneously, results may not be accurate.
                    Although we explicitly ban this, user can also trigger this by call the function with async_optimize = True+False.
                    Please use consistent `async_optimize` during a SLAM session.
        :return:
        """
        assert surface_xyz.device == self.device, \
            f"Device of map {self.device} and input observation " \
            f"{surface_xyz.device} must be the same."

        # This lock prevents meshing thread reading error.
        self.modifying_lock.acquire()

        # -- 1. Allocate new voxels --
        surface_xyz_zeroed = surface_xyz - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        surface_grid_id = torch.ceil(surface_xyz_normalized).long() - 1
        surface_grid_id = self._linearize_id(surface_grid_id)  # Func

        # Remove the observations where it is sparse.
        unq_mask = None
        if self.args.prune_min_vox_obs > 0:
            _, unq_inv, unq_count = torch.unique(surface_grid_id, return_counts=True, return_inverse=True)
            unq_mask = (unq_count > self.args.prune_min_vox_obs)[unq_inv]
            surface_xyz_normalized = surface_xyz_normalized[unq_mask]
            surface_grid_id = surface_grid_id[unq_mask]

        # Identify empty cells, fill the indexer.
        invalid_surface_ind = self.indexer[surface_grid_id] == -1
        if invalid_surface_ind.sum() > 0:
            invalid_flatten_id = torch.unique(surface_grid_id[invalid_surface_ind])
            # We expand this because we want to create some dummy voxels which helps the mesh extraction.
            invalid_flatten_id = self._expand_flatten_id(invalid_flatten_id, ensure_valid=False)  # Func
            invalid_flatten_id = invalid_flatten_id[self.indexer[invalid_flatten_id] == -1]
            self.allocate_block(invalid_flatten_id)  # Func

        self.modifying_lock.release()
        return unq_mask

    def _expand_flatten_id(self, base_flatten_id: torch.Tensor, ensure_valid: bool = True):
        expanded_flatten_id = [base_flatten_id]
        updated_pos = self._unlinearize_id(base_flatten_id)
        for affected_offset in self.mesh_update_affected:
            rs_id = updated_pos + affected_offset
            for dim in range(3):
                rs_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
            rs_id = self._linearize_id(rs_id)
            if ensure_valid:
                rs_id = rs_id[self.indexer[rs_id] != -1]
            expanded_flatten_id.append(rs_id)
        expanded_flatten_id = torch.unique(torch.cat(expanded_flatten_id))
        return expanded_flatten_id

    def get_sdf(self, xyz: torch.Tensor):
        """
        Get the sdf value of the requested positions with computation graph built.
        :param xyz: (N, 3)
        :return: sdf: (M,), std (M,), valid_mask: (N,) with M elements being 1.
        """
        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        with torch.no_grad():
            grid_id = torch.ceil(xyz_normalized.detach()).long() - 1
            sample_latent_id = self.indexer[self._linearize_id(grid_id)]
            sample_valid_mask = sample_latent_id != -1
            # Prune validity by ignore-count.
            valid_valid_mask = self.voxel_obs_count[sample_latent_id[sample_valid_mask]] > self.args.ignore_count_th
            sample_valid_mask[sample_valid_mask.clone()] = valid_valid_mask
            valid_latent = self.latent_vecs[sample_latent_id[sample_valid_mask]]

        valid_xyz_rel = xyz_normalized[sample_valid_mask] - grid_id[sample_valid_mask] - self.relative_network_offset

        '''sdf, std = net_util.forward_model(self.model.decoder,
                                          latent_input=valid_latent, xyz_input=valid_xyz_rel, no_detach=True)
        
        return sdf.squeeze(-1), std.squeeze(-1), sample_valid_mask'''