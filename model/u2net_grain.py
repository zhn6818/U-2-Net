import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """上采样函数，使src与tar具有相同的空间尺寸"""
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src

def _transpose_upsample(src, tar, in_ch, out_ch):
    """使用转置卷积进行上采样"""
    scale_factor = tar.shape[2] // src.shape[2]
    if scale_factor == 2:
        return nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)(src)
    else:
        return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

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
        
        # 解码器
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
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
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
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
    """针对晶粒度晶界分割优化的U²-Net"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_GRAIN, self).__init__()
        
        # 编码器 - 减少下采样层数，保留更多细节
        self.stage1 = RSU5G(in_ch, 16, 32)  # 减少通道数
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU4G(32, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(128, 64, 256)  # 最深层使用空洞卷积
        
        # 解码器
        self.stage3d = RSU4G(384, 32, 128)  # 128+256=384
        self.stage2d = RSU4G(192, 16, 64)   # 64+128=192
        self.stage1d = RSU5G(96, 16, 32)    # 32+64=96
        
        # 侧输出层
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        
        # 最终融合层
        self.outconv = nn.Conv2d(4*out_ch, out_ch, 1)
        
        # 全局边缘增强
        self.global_edge_enhance = EdgeEnhanceModule(out_ch)

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
        
        # 解码器
        hx4up = _upsample_like(hx4, hx3)
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # 侧输出
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        
        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, d1)
        
        # 特征融合
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        
        # 全局边缘增强
        d0 = self.global_edge_enhance(d0)
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4)

### U²-Net-Grain 轻量版 ###
class U2NETP_GRAIN(nn.Module):
    """轻量版晶界分割网络"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP_GRAIN, self).__init__()
        
        self.stage1 = RSU5G(in_ch, 8, 16)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU4G(16, 8, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(32, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(64, 32, 64)
        
        # 解码器
        self.stage3d = RSU4G(128, 16, 64)
        self.stage2d = RSU4G(96, 8, 32)
        self.stage1d = RSU5G(48, 8, 16)
        
        self.side1 = nn.Conv2d(16, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(4*out_ch, out_ch, 1)
        self.global_edge_enhance = EdgeEnhanceModule(out_ch)

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
        
        # 解码器
        hx4up = _upsample_like(hx4, hx3)
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # 侧输出
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, d1)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))
        d0 = self.global_edge_enhance(d0)
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4) 