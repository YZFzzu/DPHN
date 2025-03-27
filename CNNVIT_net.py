import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def get_vit_feature_layer(save_output):
    layer0=save_output.outputs[7][:, 1:, :]
    layer1=save_output.outputs[8][:, 1:, :]
    layer2=save_output.outputs[9][:, 1:, :]
    layer3=save_output.outputs[10][:, 1:, :]
    layer4=save_output.outputs[11][:, 1:, :]

    return layer0,layer1,layer2,layer3,layer4

def unified_dimensions(cnn_layerx):
    cnn_dis_fpooled = F.adaptive_max_pool2d(cnn_layerx, (14, 14)) 
    cnn_dis_flattend_pooled = torch.flatten(cnn_dis_fpooled, start_dim=2).permute(0, 2, 1)  
    return cnn_dis_flattend_pooled

def fused(fused_features_layer0,fused_features_layer1,fused_features_layer2,fused_features_layer3,fused_features_layer4):
    b, n, c = fused_features_layer0.shape
    h, w = 14, 14
    fused_features = torch.cat(
        (fused_features_layer0.permute(0, 2, 1).view(b, c, h, w), 
         fused_features_layer1.permute(0, 2, 1).view(b, c, h, w),
         fused_features_layer2.permute(0, 2, 1).view(b, c, h, w),
         fused_features_layer3.permute(0, 2, 1).view(b, c, h, w),
         fused_features_layer4.permute(0, 2, 1).view(b, c, h, w),
         ), dim=1)  # (16,768*5,14,14)
    return fused_features

def get_resnet_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0],
            save_output.outputs[1],
            save_output.outputs[2]
        ),
        dim=1
    )
    return feat
def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[7][:,1:,:],
            save_output.outputs[8][:,1:,:],
            save_output.outputs[9][:,1:,:],
            save_output.outputs[10][:, 1:, :],
            save_output.outputs[11][:, 1:, :]
        ),
        dim=2
    )
    return feat


class ModifiedTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # to_q, to_kv, to_out
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x, extra_kv=None):
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        if extra_kv is not None:
            # 将ResNet50特征作为额外的键值对
            extra_k, extra_v = self.to_kv(extra_kv).chunk(2, dim=-1)
            k = torch.cat([k, extra_k], dim=-2)
            v = torch.cat([v, extra_v], dim=-2)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ChannelAttention_Conv(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ChannelAttention_Conv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.conv(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = avg_out + max_out
        return self.sigmoid(out)



class ImageQualityPredictor(nn.Module):
    def __init__(self, in_channels=768*5, reduced_channels=1024, out_channels=256):
        super(ImageQualityPredictor, self).__init__()
        self.reduce_channels = nn.Conv2d(in_channels, out_channels=256, kernel_size=1)
        #self.channel_attention = ChannelAttention_FC(in_channels=512)
        self.channel_attention = ChannelAttention_Conv(in_channels=512)
        #self.spatial_attention = SpatialAttention(kernel_size=7)
        self.conv1 =nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=reduced_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduced_channels, out_channels=512, kernel_size=3, padding=1),
        )
        self.conv2 =nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Sequential(
            nn.Conv2d(out_channels, out_channels=1, kernel_size=1),
            nn.Flatten(start_dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(14 * 14, 1),
        )

    def forward(self, vit_dis,vit_ref, fused_features):
        vit_dis = self.reduce_channels(vit_dis) 
        vit_ref = self.reduce_channels(vit_ref) 
        fused_features = self.reduce_channels(fused_features)
        features = torch.cat((vit_ref-vit_dis,vit_ref, fused_features), dim=1)
        features = self.conv1(features)

        channel_weights = self.channel_attention(features)
        channel_attended_features = features * channel_weights
       
        out = self.conv2(channel_attended_features)
        out=self.flatten(out)
        out = self.fc(out)
        return out
###
