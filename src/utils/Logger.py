import os

import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, slam
                 ):
        self.ckptsdir = slam.ckptsdir
        self.dense_map_dict = slam.dense_map_dict
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.verbose = slam.verbose

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        save_dict = {
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            # 'keyframe_dict':keyframe_dict,
            'selected_keyframes': selected_keyframes,
            'idx': idx,
        }
        for key in self.dense_map_dict.keys():
            cold_vars = dict()
            for key2, val2 in self.dense_map_dict[key].cold_vars.items():
                cold_vars[f"{key}_{key2}"] = val2
            save_dict.update(cold_vars)
        torch.save(save_dict, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
