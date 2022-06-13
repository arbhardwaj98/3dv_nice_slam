"""
Replaced grid based map with dense voxel hashing based map
"""

from src.Tracker import Tracker
from src.utils.Logger import Logger
from src.utils.Renderer import Renderer
from src.utils.Mesher import Mesher
from src.Mapper import Mapper
from src.voxel_Mapper import DenseIndexedMap
from src.utils.datasets import get_dataset
from src import config
import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    '''  NICE_SLAM main class.

    Mainly allocate shared resources, and dispatch mapping and tracking process.
    '''

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg, nice=self.nice)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)
        if self.nice:
            self.load_pretrain(cfg)
            self.grid_init(cfg)
        else:
            self.dense_map_dict = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        # Store dense map tensors in shared memory
        for key, val in self.dense_map_dict.items():
            for key2, val2 in val.cold_vars.items():
                if isinstance(val2, torch.Tensor):
                    val2 = val2.to(self.cfg['mapping']['device'])
                    val2.share_memory_()
                    self.dense_map_dict[key].cold_vars[key2] = val2
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under \
                {self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under \
                {self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx * self.fx
            self.fy = sy * self.fy
            self.cx = sx * self.cx
            self.cy = sy * self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge'] * 2
            self.W -= self.cfg['cam']['crop_edge'] * 2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound']) * self.scale)
        bound_divisable = cfg['grid_len']['bound_divisable']
        # enlarge the bound a bit to allow it divisable by bound_divisable
        self.bound[:, 1] = (((self.bound[:, 1] - self.bound[:, 0]) /
                             bound_divisable).int() + 1) * bound_divisable + self.bound[:, 0]
        if self.nice:
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound * self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders 

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8 + 7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8 + 5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.
        c_dim = 32, representation encoding dimension
        Creates dense maps each for fine, mid, coarse, color

        Args:
            cfg (dict): parsed config dict.
        """
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        dense_map = {}
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1] - self.bound[:, 0]

        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_dense_map = DenseIndexedMap(
                coarse_key,
                self.cfg,
                self.bound,
                list(map(int, (xyz_len * self.coarse_bound_enlarge / coarse_grid_len).tolist()))
            )
            dense_map[coarse_key] = coarse_dense_map

        middle_key = 'grid_middle'
        middle_dense_map = DenseIndexedMap(
            middle_key,
            self.cfg,
            self.bound,
            list(map(int, (xyz_len / middle_grid_len).tolist()))
        )
        dense_map[middle_key] = middle_dense_map

        fine_key = 'grid_fine'
        fine_dense_map = DenseIndexedMap(
            fine_key,
            self.cfg,
            self.bound,
            list(map(int, (xyz_len / fine_grid_len).tolist()))
        )
        dense_map[fine_key] = fine_dense_map

        color_key = 'grid_color'
        color_dense_map = DenseIndexedMap(
            color_key,
            self.cfg,
            self.bound,
            list(map(int, (xyz_len / color_grid_len).tolist()))
        )
        dense_map[color_key] = color_dense_map

        self.dense_map_dict = dense_map

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank,))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank,))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank,))
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Save dense maps
        if not os.path.exists(f"output/{self.dataset}/"):
            os.mkdir(f"output/{self.dataset}/")
        for key in self.dense_map_dict.keys():
            self.dense_map_dict[key].save(f"output/{self.dataset}/{key}_dense_map_dict.pt", key)


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
