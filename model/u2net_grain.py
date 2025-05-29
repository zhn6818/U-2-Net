import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class REBNCONV(nn.Module):
    """基础的卷积-BN-ReLU模块"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """结合通道和空间注意力的CBAM模块"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EdgeEnhanceModule(nn.Module):
    """边缘增强模块"""
    def __init__(self, in_ch):
        super(EdgeEnhanceModule, self).__init__()
        self.edge_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # 计算边缘响应
        edge_x = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        edge_y = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 边缘增强
        edge_enhanced = self.relu(self.bn(self.edge_conv(x)))
        edge_weight = torch.sigmoid(edge_magnitude)
        
        return x + edge_enhanced * edge_weight

def _upsample_like(src, tar):
    """上采样函数，使src与tar具有相同的空间尺寸 - 使用双线性插值作为备选"""
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src

def _transpose_upsample(src, tar, in_ch, out_ch):
    """使用转置卷积进行上采样"""
    scale_factor = tar.shape[2] // src.shape[2]
    if scale_factor == 2:
        return nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)(src)
    elif scale_factor == 4:
        return nn.ConvTranspose2d(in_ch, out_ch, 8, stride=4, padding=2)(src)
    elif scale_factor == 8:
        return nn.ConvTranspose2d(in_ch, out_ch, 16, stride=8, padding=4)(src)
    else:
        return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

class TransposeUpsampler(nn.Module):
    """转置卷积上采样模块"""
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(TransposeUpsampler, self).__init__()
        if scale_factor == 2:
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        elif scale_factor == 4:
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 8, stride=4, padding=2)
        elif scale_factor == 8:
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 16, stride=8, padding=4)
        else:
            # 对于其他比例，使用连续的2倍上采样
            layers = []
            current_ch = in_ch
            for _ in range(int(np.log2(scale_factor))):
                layers.append(nn.ConvTranspose2d(current_ch, out_ch, 4, stride=2, padding=1))
                current_ch = out_ch
            self.upsample = nn.Sequential(*layers)
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, target_size=None):
        x = self.upsample(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # 如果需要精确匹配目标尺寸，进行微调
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x

### 改进的RSU模块 - RSU5G (Grain-optimized) ###
class RSU5G(nn.Module):
    """针对晶界分割优化的RSU5模块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5G, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        # 编码器部分 - 减少下采样层数
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        # 瓶颈层使用空洞卷积
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=4)
        
        # 解码器部分
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        
        # 转置卷积上采样层
        self.upsample4d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample3d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample2d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        
        # 注意力模块
        self.cbam = CBAM(out_ch)
        
        # 边缘增强模块
        self.edge_enhance = EdgeEnhanceModule(out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        # 编码器
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        # 瓶颈层
        hx4 = self.rebnconv4(hx3)
        hx5 = self.rebnconv5(hx4)
        
        # 解码器 - 使用转置卷积上采样
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = self.upsample4d(hx4d, target_size=hx3.shape[2:])
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upsample3d(hx3d, target_size=hx2.shape[2:])
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upsample2d(hx2d, target_size=hx1.shape[2:])
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        # 残差连接
        output = hx1d + hxin
        
        # 应用注意力机制
        output = self.cbam(output)
        
        # 边缘增强
        output = self.edge_enhance(output)
        
        return output

### 改进的RSU4模块 ###
class RSU4G(nn.Module):
    """针对晶界分割优化的RSU4模块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4G, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=4)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        
        # 转置卷积上采样层
        self.upsample3d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample2d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = self.upsample3d(hx3d, target_size=hx2.shape[2:])
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upsample2d(hx2d, target_size=hx1.shape[2:])
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        output = hx1d + hxin
        output = self.cbam(output)
        
        return output

### 改进的RSU4F模块 ###
class RSU4FG(nn.Module):
    """针对晶界分割优化的RSU4F模块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4FG, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        output = hx1d + hxin
        output = self.cbam(output)
        
        return output

##### U²-Net-Grain: 针对晶界分割优化的网络 ####
class U2NET_GRAIN(nn.Module):
    """针对晶界分割优化的U²-Net - 增强版本with Dense连接"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_GRAIN, self).__init__()
        
        # 编码器 - 通道数扩大一倍
        self.stage1 = RSU5G(in_ch, 32, 64)      # 原来16,32 -> 现在32,64
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU4G(64, 32, 128)        # 原来32,16,64 -> 现在64,32,128
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(128, 64, 256)       # 原来64,32,128 -> 现在128,64,256
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(256, 128, 512)     # 原来128,64,256 -> 现在256,128,512
        
        # Dense连接的特征融合层 - 用于降维和特征增强
        self.dense_conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dense_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dense_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 解码器 - 采用dense连接方式
        # stage3d: 输入为stage4上采样(512) + stage3(256) = 768
        self.stage3d = RSU4G(768, 64, 256)      # 512+256=768
        
        # stage2d: 输入为stage3d上采样(256) + stage2(128) + dense3_up(128) = 512
        self.stage2d = RSU4G(512, 32, 128)      # 256+128+128=512 (包含dense连接)
        
        # stage1d: 输入为stage2d上采样(128) + stage1(64) + dense2_up(64) = 256
        self.stage1d = RSU5G(256, 32, 64)       # 128+64+64=256 (包含dense连接)
        
        # 转置卷积上采样层 - 用于解码器
        self.upsample4to3 = TransposeUpsampler(512, 512, scale_factor=2)  # stage4 -> stage3
        self.upsample3to2 = TransposeUpsampler(256, 256, scale_factor=2)  # stage3d -> stage2
        self.upsample2to1 = TransposeUpsampler(128, 128, scale_factor=2)  # stage2d -> stage1
        
        # Dense连接的转置卷积上采样层
        self.dense3_upsample = TransposeUpsampler(128, 128, scale_factor=2)  # dense3 -> stage2 size
        self.dense2_upsample = TransposeUpsampler(64, 64, scale_factor=2)    # dense2 -> stage1 size
        
        # 侧输出的转置卷积上采样层
        self.side2_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=2)   # side2 -> side1 size
        self.side3_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=4)   # side3 -> side1 size
        self.side4_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=8)   # side4 -> side1 size
        
        # 侧输出层 - 通道数相应增加
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        # 最终融合层
        self.outconv = nn.Conv2d(4*out_ch, out_ch, 1)
        
        # 全局边缘增强
        self.global_edge_enhance = EdgeEnhanceModule(out_ch)
        
        # Dense连接的特征增强模块
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(out_ch, out_ch*2, 3, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        hx = x
        
        # 编码器
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        
        # Dense连接特征处理
        dense1 = self.dense_conv1(hx1)  # 64->32
        dense2 = self.dense_conv2(hx2)  # 128->64
        dense3 = self.dense_conv3(hx3)  # 256->128
        
        # 解码器 - 使用转置卷积上采样
        hx4up = self.upsample4to3(hx4, target_size=hx3.shape[2:])
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))  # 512+256=768
        
        hx3dup = self.upsample3to2(hx3d, target_size=hx2.shape[2:])
        dense3_up = self.dense3_upsample(dense3, target_size=hx2.shape[2:])  # 将dense3上采样到hx2尺寸
        hx2d = self.stage2d(torch.cat((hx3dup, hx2, dense3_up), 1))  # 256+128+128=512
        
        hx2dup = self.upsample2to1(hx2d, target_size=hx1.shape[2:])
        dense2_up = self.dense2_upsample(dense2, target_size=hx1.shape[2:])  # 将dense2上采样到hx1尺寸
        hx1d = self.stage1d(torch.cat((hx2dup, hx1, dense2_up), 1))  # 128+64+64=256
        
        # 侧输出 - 使用转置卷积上采样
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = self.side2_upsample(d2, target_size=d1.shape[2:])
        
        d3 = self.side3(hx3d)
        d3 = self.side3_upsample(d3, target_size=d1.shape[2:])
        
        d4 = self.side4(hx4)
        d4 = self.side4_upsample(d4, target_size=d1.shape[2:])
        
        # 特征融合
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        
        # 全局边缘增强
        d0 = self.global_edge_enhance(d0)
        
        # 特征增强
        d0 = self.feature_enhance(d0) + d0  # 残差连接
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4)

### U²-Net-Grain 轻量版 ###
class U2NETP_GRAIN(nn.Module):
    """轻量版晶界分割网络 - 增强版本with Dense连接"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP_GRAIN, self).__init__()
        
        # 轻量版也适当增加通道数
        self.stage1 = RSU5G(in_ch, 16, 32)      # 原来8,16 -> 现在16,32
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU4G(32, 16, 64)         # 原来16,8,32 -> 现在32,16,64
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(64, 32, 128)        # 原来32,16,64 -> 现在64,32,128
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(128, 64, 128)      # 原来64,32,64 -> 现在128,64,128
        
        # Dense连接的特征融合层 - 轻量版
        self.dense_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.dense_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dense_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 解码器 - 采用dense连接
        self.stage3d = RSU4G(256, 32, 128)      # 128+128=256
        self.stage2d = RSU4G(224, 16, 64)       # 128+64+64=224 (包含dense连接)
        self.stage1d = RSU5G(128, 16, 32)       # 64+32+32=128 (包含dense连接)
        
        # 转置卷积上采样层 - 轻量版
        self.upsample4to3 = TransposeUpsampler(128, 128, scale_factor=2)  # stage4 -> stage3
        self.upsample3to2 = TransposeUpsampler(128, 128, scale_factor=2)  # stage3d -> stage2
        self.upsample2to1 = TransposeUpsampler(64, 64, scale_factor=2)    # stage2d -> stage1
        
        # Dense连接的转置卷积上采样层 - 轻量版
        self.dense3_upsample = TransposeUpsampler(64, 64, scale_factor=2)   # dense3 -> stage2 size
        self.dense2_upsample = TransposeUpsampler(32, 32, scale_factor=2)   # dense2 -> stage1 size
        
        # 侧输出的转置卷积上采样层 - 轻量版
        self.side2_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=2)   # side2 -> side1 size
        self.side3_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=4)   # side3 -> side1 size
        self.side4_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=8)   # side4 -> side1 size
        
        # 侧输出层
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(4*out_ch, out_ch, 1)
        self.global_edge_enhance = EdgeEnhanceModule(out_ch)
        
        # 轻量版特征增强模块
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        hx = x
        
        # 编码器
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        
        # Dense连接特征处理
        dense1 = self.dense_conv1(hx1)  # 32->16
        dense2 = self.dense_conv2(hx2)  # 64->32
        dense3 = self.dense_conv3(hx3)  # 128->64
        
        # 解码器 - 使用转置卷积上采样
        hx4up = self.upsample4to3(hx4, target_size=hx3.shape[2:])
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))  # 128+128=256
        
        hx3dup = self.upsample3to2(hx3d, target_size=hx2.shape[2:])
        dense3_up = self.dense3_upsample(dense3, target_size=hx2.shape[2:])  # 将dense3上采样到hx2尺寸
        hx2d = self.stage2d(torch.cat((hx3dup, hx2, dense3_up), 1))  # 128+64+64=224
        
        hx2dup = self.upsample2to1(hx2d, target_size=hx1.shape[2:])
        dense2_up = self.dense2_upsample(dense2, target_size=hx1.shape[2:])  # 将dense2上采样到hx1尺寸
        hx1d = self.stage1d(torch.cat((hx2dup, hx1, dense2_up), 1))  # 64+32+32=128
        
        # 侧输出 - 使用转置卷积上采样
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = self.side2_upsample(d2, target_size=d1.shape[2:])
        d3 = self.side3(hx3d)
        d3 = self.side3_upsample(d3, target_size=d1.shape[2:])
        d4 = self.side4(hx4)
        d4 = self.side4_upsample(d4, target_size=d1.shape[2:])
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        d0 = self.global_edge_enhance(d0)
        
        # 特征增强
        d0 = self.feature_enhance(d0) + d0  # 残差连接
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4) 