"""
Updated NICE-SLAM decoder. Retrieves latent representations
from dense voxel array instead of grid based voxel map.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] + 
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips 
            else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, dense_map_dict):

        dense_map = dense_map_dict['grid_' + self.name]
        inter_p, voxel_mask = dense_map.interpolate_point(xyz=p.squeeze(0))
        decoder_input = torch.zeros(p.squeeze(0).shape[0], dense_map.latent_dim).to(p.device)
        if not inter_p.shape[0] == 0:
            decoder_input[voxel_mask, :] = inter_p.type(decoder_input.dtype)

        if self.concat_feature:
            inter_p_middle, voxel_mask_middle = dense_map_dict["grid_middle"].interpolate_point(xyz=p.squeeze(0))
            voxel_mask = torch.logical_and(voxel_mask, voxel_mask_middle)
            decoder_input_middle = torch.zeros(p.squeeze(0).shape[0], dense_map.latent_dim).to(p.device)
            decoder_input_middle[voxel_mask_middle, :] = inter_p_middle.type(decoder_input_middle.dtype)
            decoder_input_unmasked = torch.cat([decoder_input, decoder_input_middle], dim=1)
            decoder_input = torch.zeros(p.squeeze(0).shape[0], 2 * dense_map.latent_dim).to(p.device)
            decoder_input[voxel_mask] = decoder_input_unmasked[voxel_mask]

        p = p.float()
        embedded_pts = self.embedder(p)

        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](decoder_input)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out2 = self.output_linear(h)
        if not self.color:
            out2 = out2.squeeze(-1)

        return out2, voxel_mask


class MLP_no_xyz(nn.Module):
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear', color=False, skips=[2], grid_len=0.16):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] + 
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips 
            else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(grid_feature, vgrid, padding_mode='border',
                          align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, dense_map_dict, **kwargs):
        dense_map = dense_map_dict['grid_' + self.name]
        inter_p, voxel_mask = dense_map.interpolate_point(xyz=p.squeeze(0))
        decoder_input = torch.zeros(p.squeeze(0).shape[0], dense_map.latent_dim).to(p.device)
        if not inter_p.shape[0] == 0:
            decoder_input[voxel_mask, :] = inter_p.type(decoder_input.dtype)

        h = decoder_input.to(dense_map.device)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([decoder_input, h], -1)
        out2 = self.output_linear(h)
        if not self.color:
            out2 = out2.squeeze(-1)

        return out2, voxel_mask


class NICE(nn.Module):
    ''' 
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.
    '''

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
        super().__init__()

        if coarse:
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len)

        self.middle_decoder = MLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size, 
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method)
        self.fine_decoder = MLP(name='fine', dim=dim, c_dim=c_dim*2, color=False, 
                                skips=[2], n_blocks=5, hidden_size=hidden_size, 
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True, 
                                 skips=[2], n_blocks=5, hidden_size=hidden_size, 
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

    def forward(self, p, dense_map_dict, stage='middle', **kwargs):
        """
            Output occupancy/color in different stage.
        """
        device = f'cuda:{p.get_device()}'
        if stage == 'coarse':
            occ2, voxel_mask = self.coarse_decoder(p, dense_map_dict)
            occ2 = occ2.squeeze(0)
            raw2 = torch.zeros(occ2.shape[0], 4).to(device).float()
            raw2[..., -1] = occ2
            return raw2, voxel_mask
        elif stage == 'middle':
            middle_occ2, voxel_mask = self.middle_decoder(p, dense_map_dict)
            middle_occ2 = middle_occ2.squeeze(0)
            raw2 = torch.zeros(middle_occ2.shape[0], 4).to(device).float()
            raw2[..., -1] = middle_occ2
            return raw2, voxel_mask
        elif stage == 'fine':
            fine_occ2, voxel_mask = self.fine_decoder(p, dense_map_dict)
            raw2 = torch.zeros(fine_occ2.shape[0], 4).to(device).float()
            middle_occ2, voxel_mask_middle = self.middle_decoder(p, dense_map_dict)
            middle_occ2 = middle_occ2.squeeze(0)
            raw2[..., -1] = fine_occ2 + middle_occ2
            return raw2, voxel_mask
        elif stage == 'color':
            fine_occ2, voxel_mask = self.fine_decoder(p, dense_map_dict)
            raw2, voxel_mask_color = self.color_decoder(p, dense_map_dict)
            middle_occ2, voxel_mask_middle = self.middle_decoder(p, dense_map_dict)
            middle_occ2 = middle_occ2.squeeze(0)
            raw2[..., -1] = fine_occ2 + middle_occ2
            return raw2, voxel_mask
