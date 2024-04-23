from vit_pytorch.cross_vit import MultiScaleEncoder, Rearrange
from einops import repeat
import ProSTGrid
import torch
import torch.nn as nn
from util import _bilinear_interpolate_no_torch_5D
import torchgeometry as tgm
from posevec2mat import pose_vec2mat, inv_pose_vec, raydist_range
device = torch.device("cuda")

def transform_nan_check(dist_min, dist_max, transform_mat4x4, transform_mat3x4):
    if torch.isnan(dist_min).any() or torch.isnan(dist_max).any() \
                or torch.isnan(transform_mat4x4).any() or (torch.abs(transform_mat3x4) > 1000).any()\
                or (torch.abs(dist_min) > 1000).any() or (torch.abs(dist_max) > 1000).any():
        return False
    else:
        return True

# Copied from vit_pytorch.cross_vit.py and modified to fix patch dim assuming input channels = 3, where it should be 2 here

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        n_channels = 2
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = n_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# Copied from vit_pytorch.cross_vit.py

class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits


class RegiNet_CrossViTv2_SW(nn.Module):
    def __init__(self):
        super(RegiNet_CrossViTv2_SW, self).__init__()

        # Define 3D convolutional layers        
        self._3D_conv = nn.Sequential(
            nn.Conv3d(1, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 3, 3, 1, 1)
        )

        self._2Dconv_3x1 = nn.Conv2d(3, 1, 3, 1, 1)

        self._2Dconv_encode = CrossViT(
                                image_size = 128,
                                # channels = 2,
                                num_classes = 1000,
                                depth = 3,
                                sm_dim = 16,            # high res dimension
                                sm_patch_size = 8,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 256,   # high res feedforward dimension
                                lg_dim = 64,            # low res dimension
                                lg_patch_size = 64,      # low res patch size
                                lg_enc_depth = 3,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 256,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 4,    # cross attention heads
                            )

        self._out_layer = nn.Linear(1000, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
        x_3d = x_exp + self._3D_conv(x)

        BATCH_SIZE = theta.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat4x4 = tgm.rtvec_to_pose(theta)
        transform_mat3x4 = transform_mat4x4[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)

        if not transform_nan_check(dist_min, dist_max, transform_mat4x4, transform_mat3x4):
            return False

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                         src, det, pix_spacing, step_size, False)

        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)

        x_3d = _bilinear_interpolate_no_torch_5D(x_3d, grid_trans)

        x_2d = torch.sum(x_3d, dim=-1)

        x_2d = self._2Dconv_3x1(x_2d)
        x_y_cat = torch.cat((x_2d, y), dim=1)

        out = self._2Dconv_encode(x_y_cat)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        return out, True
