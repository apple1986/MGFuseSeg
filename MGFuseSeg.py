from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from positional_encoding import SinusoidalPositionalEncoding
import kornia as K
import numpy as np

class PatchEmbed(nn.Module):
    ## patch_size：块的大小，这里每一个块由4x4像素数目组成
    ## embed_dim：把每一个块线性映射的通道维数，这里设置96维度的特征通道数，也就是说4x4x3的块，经过核大小为4x4x3
    ## 以及步长为4的卷积操作变成1x1x96的一个表示
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) ## 一个数变成两个同样数的元组，如56-->(56, 56)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
    
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinMixerBlock(nn.Module):
    def __init__(self, dim, input_resolution, window_size=7, shift_size=0,window_mlp_ratio=2.0,
                 channel_mlp_ratio=2.0,drop=0., drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = Mlp(in_features=(self.window_size**2),hidden_features=int((self.window_size**2) * window_mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * channel_mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        attn_windows = self.mlp1(x_windows.transpose(1,2)).transpose(1,2)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0: # 把移位的块的特征又返回过去
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))  
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, window_size,
                 window_mlp_ratio=2.0,channel_mlp_ratio=2.0, 
                 drop=0., attn_drop=0.,drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([SwinMixerBlock(dim=dim, input_resolution=input_resolution, 
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     window_mlp_ratio=window_mlp_ratio,
                                     channel_mlp_ratio=channel_mlp_ratio,
                                     drop=drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

################
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
####################
class BGFuse(nn.Module):
    def __init__(self, in_ch, out_ch, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.out_ch = out_ch
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1x1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.pus = nn.PixelUnshuffle(2)
        self.ps = nn.PixelShuffle(2)
        self.sg = nn.Sigmoid()
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)

    def forward(self, x1, x2): # x1小尺度， x2大尺度
        B, N, C = x1.shape
        x1 = x1.view(B, self.H, self.W, C).permute(0,3,1,2)
        x1 = self.conv1x1(x1) # 维度与细粒度特征保持一致
        x1_w = self.sg(x1)
        
        x2_w = x2.view(B, self.H*2, self.W*2, self.out_ch).permute(0,3,1,2)
        ## pixel unshuffle
        x2_w = self.pus(x2_w).reshape(B, self.out_ch, 4, self.H, self.W)
        x1_w = x1_w.unsqueeze(2)
        x2_w = x2_w*(1.0+x1_w)
        x2_w = x2_w.reshape(B, self.out_ch*4, self.H, self.W)
        x2_w = self.ps(x2_w)
        x2_w = x2_w.reshape(B, self.out_ch, -1).permute(0, 2, 1)
        x2_w = self.norm2(x2_w)
        
        x1 = self.up(x1)
        x1 = x1.reshape(B, self.out_ch, -1).permute(0,2,1)
        x1 = self.norm1(x1)        

        out = torch.cat([x2_w, x1], dim=2)

        return out 

#########################
class LGFuse(nn.Module):
    def __init__(self, in_ch_m, in_ch_c, out_ch):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_ch_c)
        self.norm2 = nn.LayerNorm(in_ch_m)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)   

        self.conv1x1 = nn.Conv2d(in_ch_m, in_ch_c, kernel_size=1)
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2): ## x1: mlp, x2: conv

        x1 = self.up(x1)        
        ## 全局增强
        x1_d = self.conv1x1(x1)
        x_u = x1_d + x2
        # 这里利用池化获取全局信息
        avg_out = torch.mean(x_u, dim=1, keepdim=True)
        max_out, _ = torch.max(x_u, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv7x7(x)
        x_u =  self.sigmoid(x)
        x_u = x2 * x_u

        x_u = self.norm1(x_u.permute(0,2,3,1)).permute(0,3,1,2)
        x_1 = self.norm2(x1.permute(0,2,3,1)).permute(0,3,1,2)
    
        x = torch.cat([x_1, x_u], dim=1)

        return x


######################
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class MGFuseSeg(nn.Module): 
    def __init__(self, img_size=224, patch_size=4, in_chans=3, classes=9,
                 embed_dim=32*3, depths=[2, 2, 6, 2], window_size=7, 
                 window_mlp_ratio=2.0, channel_mlp_ratio=4.0, drop_rate=0.,
                 drop_path_rate=0.1,norm_layer=nn.LayerNorm, patch_norm=True, ape=True):
        """
        patch_size: 下采样的比例，一个Patch（token）由原始输入图像中的4个像素聚合而成
        embed_dim： 表示MLP中通道维度, change the embed_dim=32*(1,2, or 3), for MGFuseSeg32, MGFuseSeg64 and MGFuseSeg96
        depths: 一个Block由多少个MLP模块组成
        window_mlp_ratio： 空间维度上扩展比例
        channel_mlp_ratio：通道维度上扩展比例
        """
        super().__init__()

        self.inc = inconv(in_chans, embed_dim//4)
        self.down1 = down(embed_dim//4, embed_dim//2)
        self.down2 = down(embed_dim//2, embed_dim)


        self.num_classes = classes
        self.num_layers = len(depths)
        self.num_blocks = [0, 1, 2, 3]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) ##最后解码输出特征数
        num_patches = (img_size//patch_size)**2 #self.patch_embed.num_patches,patch总数
        self.ape = ape
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = SinusoidalPositionalEncoding(d_model=embed_dim, dropout=0, max_len=num_patches)

        patches_resolution = [img_size//patch_size, img_size//patch_size]#self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block_1 = BasicLayer(dim=int(embed_dim * 1),
                               input_resolution=(patches_resolution[0] // (2 ** self.num_blocks[0]),
                                                 patches_resolution[1] // (2 ** self.num_blocks[0])),
                               depth=depths[self.num_blocks[0]],
                               window_size = window_size,
                               window_mlp_ratio=window_mlp_ratio,
                               channel_mlp_ratio=channel_mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:self.num_blocks[0]]):sum(depths[:self.num_blocks[0] + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (self.num_blocks[0] < self.num_layers - 1) else None)
        self.block_2 = BasicLayer(dim=int(embed_dim * 2),
                               input_resolution=(patches_resolution[0] // (2 ** self.num_blocks[1]),
                                                 patches_resolution[1] // (2 ** self.num_blocks[1])),
                               depth=depths[self.num_blocks[1]],
                               window_size = window_size,
                               window_mlp_ratio=window_mlp_ratio,
                               channel_mlp_ratio=channel_mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:self.num_blocks[1]]):sum(depths[:self.num_blocks[1] + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (self.num_blocks[1] < self.num_layers - 1) else None)
        self.block_3 = BasicLayer(dim=int(embed_dim * 4),
                               input_resolution=(patches_resolution[0] // (2 ** self.num_blocks[2]),
                                                 patches_resolution[1] // (2 ** self.num_blocks[2])),
                               depth=depths[self.num_blocks[2]],
                               window_size = 7, #window_size
                               window_mlp_ratio=window_mlp_ratio,
                               channel_mlp_ratio=channel_mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:self.num_blocks[2]]):sum(depths[:self.num_blocks[2] + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (self.num_blocks[2] < self.num_layers - 1) else None)

        ############decoder
        self.up1_cat = BGFuse(embed_dim*8, embed_dim*4, H=7, W=7)
        self.block4 = BasicLayer(dim=int(embed_dim * 8),
                               input_resolution=(patches_resolution[0] // (2 ** self.num_blocks[2]),
                                                 patches_resolution[1] // (2 ** self.num_blocks[2])),
                               depth=depths[self.num_blocks[3]],
                               window_size = window_size,
                               window_mlp_ratio=window_mlp_ratio,
                               channel_mlp_ratio=channel_mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:self.num_blocks[3]]):sum(depths[:self.num_blocks[3] + 1])],
                               norm_layer=norm_layer,
                               downsample= None)
        self.up2_cat = BGFuse(embed_dim*8, embed_dim*2, H=14, W=14)
        
        self.block5 = BasicLayer(dim=int(embed_dim * 4),
                               input_resolution=(patches_resolution[0] // (2 ** self.num_blocks[1]),
                                                 patches_resolution[1] // (2 ** self.num_blocks[1])),
                               depth=depths[self.num_blocks[3]],
                               window_size = window_size,
                               window_mlp_ratio=window_mlp_ratio,
                               channel_mlp_ratio=channel_mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:self.num_blocks[3]]):sum(depths[:self.num_blocks[3] + 1])],
                               norm_layer=norm_layer,
                               downsample= None)

        self.up3 = LGFuse(embed_dim*4, embed_dim, embed_dim//2) 
        self.conv1 = double_conv(embed_dim*4+embed_dim, embed_dim//2)
        
        self.up4 = LGFuse(embed_dim//2, embed_dim//2, embed_dim//4)  
        self.conv2 = double_conv(embed_dim//2+embed_dim//2, embed_dim//4)

        self.up5 = LGFuse(embed_dim//4, embed_dim//4, embed_dim//4)
        self.conv3 = double_conv(embed_dim//4+embed_dim//4, embed_dim//4)
        self.seg = outconv(embed_dim//4, self.num_classes)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_features(self, x):
        x_c1 = self.inc(x) # 8x224x224
        x_c2 = self.down1(x_c1) # 16x112x112
        x_c3 = self.down2(x_c2) # 32x56x56
        
        B, C, H, W = x_c3.shape
        x_m = x_c3.view(B, C, -1).permute(0,2,1)
        if self.ape:
            x_m = x_m + self.absolute_pos_embed(x_m)

        x0 = self.pos_drop(x_m) # 32x56x56
        x1 = self.block_1(x0) # 64x28x28
        x2 = self.block_2(x1) # 128x514x14       
        x3 = self.block_3(x2) # 256x7x7  

        x2_cat = self.up1_cat(x3, x2)
        x2_cat = self.block4(x2_cat)

        x1_cat = self.up2_cat(x2_cat, x1)
        x1_cat = self.block5(x1_cat)


        x1_cat = x1_cat.view(B, 28, 28, self.embed_dim*4).permute(0,3,1,2)
        x_m = x_m.view(B, 56, 56, self.embed_dim).permute(0,3,1,2)
        x_f1 = self.up3(x1_cat, x_m)
        x_f1 = self.conv1(x_f1)

        x_f2 = self.up4(x_f1, x_c2)
        x_f2 = self.conv2(x_f2)

        x_f3 = self.up5(x_f2, x_c1)
        x_f3 = self.conv3(x_f3)
        seg = self.seg(x_f3)

        return seg

    def forward(self, x):
        x = self.extract_features(x)
        return x

####################
def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


from ptflops import get_model_complexity_info

if __name__ == '__main__':
    
    net = MGFuseSeg(img_size=224,in_chans=1)
    net.eval()
    net(torch.randn(1, 1, 224, 224)) #NCHW
    ## cal para FLOPS
    macs, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs ))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params ))
    # print(net)
    print("MGFuseSeg\n",
            sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
    total_paramters = netParams(net)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    
