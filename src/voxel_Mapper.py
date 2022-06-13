"""
Implements a voxel hashing based dense map data structure.
Adapted from Di-Fusion's implementation of voxel hashing.
https://github.com/huangjh-pub/di-fusion/tree/merged/pytorch
"""


import functools
from pathlib import Path

import torch
import logging
import torch.optim
import numpy as np
import numba
import torch.multiprocessing as mp
import copy


@numba.jit
def _get_valid_idx(base_idx: np.ndarray, query_idx: np.ndarray):
    mask = np.zeros((base_idx.shape[0],), dtype=np.bool_)
    for vi, v in enumerate(base_idx):
        if query_idx[np.searchsorted(query_idx, v)] != v:
            mask[vi] = True
    return mask


class MapVisuals:
    def __init__(self):
        self.mesh = []
        self.blocks = []
        self.samples = []
        self.uncertainty = []


class DenseIndexedMap:
    def __init__(self, name: str, cfg: dict, bound: torch.Tensor, shape: list):
        """
        Initialize a densely indexed latent map.
        For easy manipulation, invalid indices are -1, and occupied indices are >= 0.

        :param name:  name
        :param cfg:  config
        :param bound:  map bounds
        :param shape: shape of the voxel grid
        """
        self.cfg = cfg
        device = cfg["mapping"]["device"]
        self.device = device
        self.bound = bound
        self.bound.share_memory_()

        self.store_idx = 0
        self.name = name

        self.voxel_size = cfg["grid_len"][name.replace("grid_", "")]
        self.n_xyz = shape
        logging.info(f"Map size Nx = {self.n_xyz[0]}, Ny = {self.n_xyz[1]}, Nz = {self.n_xyz[2]}")

        self.bound_min = self.bound[:, 0].float().to(self.device)
        self.bound_min.share_memory_()
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz, device=device)
        self.bound_max.share_memory_()
        self.latent_dim = cfg['model']['c_dim']
        self.integration_offsets = [torch.tensor(t, device=self.device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        self.prune_min_vox_obs = cfg["mapping"]["prune_min_vox_obs"]
        self.ignore_count_th = cfg["mapping"]["ignore_count_th"]
        # Directly modifiable from outside.
        self.extract_mesh_std_range = None

        self.mesh_update_affected = [torch.tensor([t], device=self.device)
                                     for t in [[-1, 0, 0], [1, 0, 0],
                                               [0, -1, 0], [0, 1, 0],
                                               [0, 0, -1], [0, 0, 1]]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float32)
        self.relative_network_offset.share_memory_()

        self.cold_vars = {
            "n_occupied": 0,
            "indexer": torch.ones(np.product(self.n_xyz), device=device, dtype=torch.long) * -1,
            # -- Voxel Attributes --
            # 1. Latent Vector (Geometry)
            "latent_vecs": torch.empty((32768, self.latent_dim), dtype=torch.float32, device=device),
            # 2. Position
            "latent_vecs_pos": torch.ones((32768,), dtype=torch.long, device=device) * -1,
            # 3. Confidence on its geometry
            "voxel_obs_count": torch.zeros((32768,), dtype=torch.float32, device=device),
            # 4. Optimized mark
            "voxel_optimized": torch.zeros((32768,), dtype=torch.bool, device=device)
        }
        for key in self.cold_vars.keys():
            if isinstance(self.cold_vars[key], torch.Tensor):
                self.cold_vars[key].share_memory_()
        self.backup_var_names = ["indexer", "latent_vecs", "latent_vecs_pos", "voxel_obs_count"]
        self.backup_vars = {}
        for p in self.cold_vars.keys():
            setattr(DenseIndexedMap, p, property(
                fget=functools.partial(DenseIndexedMap._get_var, name=p),
                fset=functools.partial(DenseIndexedMap._set_var, name=p)
            ))

    def save(self, path, key):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('wb') as f:
            torch.save(torch.sum(self.cold_vars['indexer']!=-1), f)

    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('rb') as f:
            self.cold_vars = torch.load(f)

    def _get_var(self, name):
        return self.cold_vars[name]

    def _set_var(self, value, name):
        self.cold_vars[name] = value

    def _inflate_latent_buffer(self, count: int):
        target_n_occupied = self.cold_vars['n_occupied'] + count
        new_inds = torch.arange(self.cold_vars['n_occupied'], target_n_occupied, device=self.device, dtype=torch.long).share_memory_()
        self.cold_vars['n_occupied'] = target_n_occupied
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

    def allocate_block(self, idx: torch.Tensor, key):
        """
        :param idx: (N, 3) or (N, ), if the first one, will call linearize id.
        NOTE: this will not check index overflow!
        """
        if idx.ndimension() == 2 and idx.size(1) == 3:
            idx = self._linearize_id(idx)
        new_id = self._inflate_latent_buffer(idx.size(0))
        self.cold_vars['latent_vecs_pos'][new_id] = idx
        self.cold_vars['indexer'][idx] = new_id

        # torch.save(torch.sum(self.cold_vars['indexer']!=-1), 'output/Replica/'+key+'_'+str(self.store_idx)+'.pt')
        
        self.store_idx = self.store_idx+1

    STATUS_CONF_BIT = 1 << 0  # 1
    STATUS_SURF_BIT = 1 << 1  # 2

    def integrate_keyframe(self, surface_xyz: torch.Tensor, key):
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
        surface_xyz = surface_xyz.to(self.device)
        # -- 1. Allocate new voxels --
        surface_xyz_zeroed = surface_xyz - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        surface_grid_id = torch.ceil(surface_xyz_normalized).long() - 1
        surface_grid_id = self._linearize_id(surface_grid_id)  # Func

        # Remove the observations where it is sparse.
        unq_mask = None
        if self.prune_min_vox_obs > 0:
            _, unq_inv, unq_count = torch.unique(surface_grid_id, return_counts=True, return_inverse=True)
            unq_mask = (unq_count > self.prune_min_vox_obs)[unq_inv]
            surface_xyz_normalized = surface_xyz_normalized[unq_mask]
            surface_grid_id = surface_grid_id[unq_mask]

        # Identify empty cells, fill the indexer.
        invalid_surface_ind = self.cold_vars['indexer'][surface_grid_id] == -1
        if invalid_surface_ind.sum() > 0:
            invalid_flatten_id = torch.unique(surface_grid_id[invalid_surface_ind])
            # We expand this because we want to create some dummy voxels which helps the mesh extraction.
            invalid_flatten_id = self._expand_flatten_id(invalid_flatten_id, ensure_valid=False)  # Func
            invalid_flatten_id = invalid_flatten_id[self.cold_vars['indexer'][invalid_flatten_id] == -1]
            self.allocate_block(invalid_flatten_id, key)  # Func

        # self.modifying_lock.release()
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
                rs_id = rs_id[self.cold_vars['indexer'][rs_id] != -1]
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
            sample_latent_id = self.cold_vars['indexer'][self._linearize_id(grid_id)]
            sample_valid_mask = sample_latent_id != -1
            # Prune validity by ignore-count.
            valid_valid_mask = self.cold_vars['voxel_obs_count'][sample_latent_id[sample_valid_mask]] > self.ignore_count_th
            sample_valid_mask[sample_valid_mask.clone()] = valid_valid_mask
            valid_latent = self.cold_vars['latent_vecs'][sample_latent_id[sample_valid_mask]]

        valid_xyz_rel = xyz_normalized[sample_valid_mask] - grid_id[sample_valid_mask] - self.relative_network_offset

        '''sdf, std = net_util.forward_model(self.model.decoder,
                                          latent_input=valid_latent, xyz_input=valid_xyz_rel, no_detach=True)
        
        return sdf.squeeze(-1), std.squeeze(-1), sample_valid_mask'''

    def point_to_encoding(self, xyz):
        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        with torch.no_grad():
            grid_id = torch.ceil(xyz_normalized.detach()).long() - 1
            linear_id = self._linearize_id(grid_id)
            indices = self.cold_vars['indexer'][linear_id]
            point_embeddings = self.cold_vars['latent_vecs'][indices]

        return point_embeddings

    def interpolate_point(self, xyz):
        """
        Outputs point encodings using tri-linear interpolation from stored voxel encodings
        :param xyz: points (N, 3)
        :return out: interpolated encodings (M, 3). Returned only for points which lie
                     inside initialized voxels and map bounds
        :return mask: mask denoting returned points encodings (N, )
        """
        # Mask out points outside map bounds
        mask_x = (xyz[:, 0] < self.bound[0][1]) & (xyz[:, 0] > self.bound[0][0])
        mask_y = (xyz[:, 1] < self.bound[1][1]) & (xyz[:, 1] > self.bound[1][0])
        mask_z = (xyz[:, 2] < self.bound[2][1]) & (xyz[:, 2] > self.bound[2][0])
        mask = mask_x & mask_y & mask_z

        # Interpolate points
        out, voxel_mask = self.interpolate_bounded_point(xyz[mask])

        # Combine masks for map bounds and voxels
        mask_combined = torch.zeros(xyz.shape[0], dtype=bool).to(mask.device)
        mask_combined[mask] = mask[mask] & voxel_mask
        return out, mask_combined

    def interpolate_bounded_point(self, xyz):
        """
        Trilinearly interpolates points
        :param xyz: points (N, 3)
        :return out: interpolated encodings (M, 3). Returned only for points which lie
                     inside initialized voxels
        :return mask: mask denoting returned points encodings (N, )
        """
        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        xyz_normalized = xyz_normalized

        grid = self.cold_vars["latent_vecs"]

        low_ids = torch.floor(xyz_normalized - 0.5).int()
        d = xyz_normalized - low_ids

        offsets = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0]
        ], dtype=int).to(xyz.device).unsqueeze(0)

        corners = (torch.tile(low_ids.unsqueeze(1), (1, 8, 1)) + torch.tile(offsets, (low_ids.shape[0], 1, 1))).long()
        corners = torch.max(
            torch.min(
                corners,
                torch.tile(
                    (torch.tensor(self.n_xyz) - 1).reshape((1, 1, -1)),
                    (corners.shape[0], corners.shape[1], 1)
                ).to(corners.device),
            ),
            torch.tile(
                (torch.tensor([0, 0, 0])).reshape((1, 1, -1)),
                (corners.shape[0], corners.shape[1], 1)
            ).to(corners.device)
        )
        corners_linearized = self._linearize_id(corners.reshape(-1, 3)).reshape(-1, 8).long()
        corners_idx = self.cold_vars["indexer"][corners_linearized.reshape(-1)].reshape(-1, 8)

        mask = torch.sum((corners_idx == -1), dim=-1) == 0
        corners_idx = corners_idx[mask]
        d = d[mask]

        c00 = grid[corners_idx[:, 0], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 7], :] * d[:, 0].unsqueeze(1)
        c01 = grid[corners_idx[:, 1], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 6], :] * d[:, 0].unsqueeze(1)
        c10 = grid[corners_idx[:, 3], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 4], :] * d[:, 0].unsqueeze(1)
        c11 = grid[corners_idx[:, 2], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 5], :] * d[:, 0].unsqueeze(1)

        c0 = c00 * (1 - d[:, 1]).unsqueeze(1) + c10 * d[:, 1].unsqueeze(1)
        c1 = c01 * (1 - d[:, 1]).unsqueeze(1) + c11 * d[:, 1].unsqueeze(1)

        c = c0 * (1 - d[:, 2]).unsqueeze(1) + c1 * d[:, 2].unsqueeze(1)

        return c, mask
