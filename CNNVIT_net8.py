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
##特征提取
'''feat = torch.cat(..., dim=2): 将 save_output.outputs 中的前五个元素（即 ViT 模型特定层的输出）进行拼接。
每个输出的第 0 维（即第一个维度）是批量大小，第 1 维是序列长度，第 2 维是特征维度。
[:,1:,:] 表示去除每个输出的第一个序列元素（通常是分类令牌），保留其余序列元素及其特征。dim=2 表示沿着特征维度进行拼接。'''

def get_vit_feature_layer(save_output):
    layer0=save_output.outputs[7][:, 1:, :]
    layer1=save_output.outputs[8][:, 1:, :]
    layer2=save_output.outputs[9][:, 1:, :]
    layer3=save_output.outputs[10][:, 1:, :]
    layer4=save_output.outputs[11][:, 1:, :]

    return layer0,layer1,layer2,layer3,layer4

def unified_dimensions(cnn_layerx):
    # 池化
    cnn_dis_fpooled = F.adaptive_max_pool2d(cnn_layerx, (14, 14))  # (16,768,14,14)
    cnn_dis_flattend_pooled = torch.flatten(cnn_dis_fpooled, start_dim=2).permute(0, 2, 1)  # (16,196,768)
    return cnn_dis_flattend_pooled

def fused(fused_features_layer0,fused_features_layer1,fused_features_layer2,fused_features_layer3,fused_features_layer4):
    b, n, c = fused_features_layer0.shape
    h, w = 14, 14
    fused_features = torch.cat(
        (fused_features_layer0.permute(0, 2, 1).view(b, c, h, w),  # (16,768,14,14)
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
###

##特征融合
class ModifiedTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 根据ViT的特征维度初始化to_q, to_kv, to_out
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

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)#(16,12,196,320)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)#(16,12,392,320)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)#(16,12,392,320)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale#(16,8,196,392)
        attn = dots.softmax(dim=-1)#(16,12,196,392)
        out = torch.matmul(attn, v)#(16,12,196,320)
        out = rearrange(out, 'b h n d -> b n (h d)')#(16,196,768)
        return self.to_out(out)
###

##一维卷积版通道注意力
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
        # 将张量从 (batch_size, channels, 1, 1) 转换为 (batch_size, 1, channels) 以进行一维卷积
        avg_out = self.conv(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 将两个注意力图相加
        out = avg_out + max_out
        # 将通道注意力权重应用于输入特征图
        return self.sigmoid(out)
###

##预测
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
        vit_dis = self.reduce_channels(vit_dis)  # (16,512,14,14)
        vit_ref = self.reduce_channels(vit_ref)  # (16,512,14,14)
        fused_features = self.reduce_channels(fused_features)  # (16,512,14,14)
        # 拼接融合特征和重建特征
        features = torch.cat((vit_ref-vit_dis,vit_ref, fused_features), dim=1)# (16,1536,14,14)
        features = self.conv1(features)#(16,512,14,14)
        # 通道注意力机制
        channel_weights = self.channel_attention(features)# (16,512,1,1)
        channel_attended_features = features * channel_weights# (16,512,14,14)
        # 空间注意力机制
        #spatial_weights = self.spatial_attention(channel_attended_features)  # (16,1,14,14)
        #spatial_attended_features = channel_attended_features * spatial_weights  # (16,512,14,14)
        # 卷积操作
        out = self.conv2(channel_attended_features)# (16,256,14,14)
        out=self.flatten(out)# (16,14*14)
        out = self.fc(out)#(16,1)
        return out
###
