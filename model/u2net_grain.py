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
        # 增加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x
            
        # 计算边缘响应
        # 确保输入没有非常极端的值
        x_mean = x.mean(dim=1, keepdim=True).clamp(-10, 10)
        edge_x = F.conv2d(x_mean, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_mean, self.sobel_y, padding=1)
        
        # 使用更稳定的方式计算边缘幅度，避免平方根导致的不稳定性
        edge_magnitude = torch.sqrt(torch.clamp(edge_x**2 + edge_y**2, min=1e-6, max=50))
        
        # 边缘增强
        edge_enhanced = self.relu(self.bn(self.edge_conv(x)))
        # 限制边缘权重的值范围，避免太大或太小
        edge_weight = torch.sigmoid(edge_magnitude).clamp(0.2, 0.8)
        
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
        
        # 添加通道压缩层 - 将concat后的3倍通道压缩回原始out_ch
        self.channel_compress = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

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
        
        # 改进：应用注意力机制和边缘增强前先concat
        # 使用通道注意力机制增强关键特征
        output_cbam = self.cbam(output)
        
        # 使用边缘增强模块强化边缘信息
        output_edge = self.edge_enhance(output)
        
        # 将原始输出、注意力增强输出和边缘增强输出concat
        # 这里需要增加一个额外的卷积层来压缩通道
        output_concat = torch.cat((output, output_cbam, output_edge), dim=1)
        
        # 压缩通道数，假设out_ch是目标通道数
        # 这里需要在类初始化时添加这个压缩层
        output_final = self.channel_compress(output_concat)
        
        # 额外添加一个残差连接，将压缩后的结果与原始输出相加
        output_final = output_final + output
        
        return output_final

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
        
        # 注意力模块
        self.cbam = CBAM(out_ch)
        
        # 边缘增强模块
        self.edge_enhance = EdgeEnhanceModule(out_ch)
        
        # 添加通道压缩层 - 将concat后的3倍通道压缩回原始out_ch
        self.channel_compress = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

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
        
        # 残差连接
        output = hx1d + hxin
        
        # 改进：应用注意力机制和边缘增强前先concat
        # 使用通道注意力机制增强关键特征
        output_cbam = self.cbam(output)
        
        # 使用边缘增强模块强化边缘信息
        output_edge = self.edge_enhance(output)
        
        # 将原始输出、注意力增强输出和边缘增强输出concat
        # 这里需要增加一个额外的卷积层来压缩通道
        output_concat = torch.cat((output, output_cbam, output_edge), dim=1)
        
        # 压缩通道数，假设out_ch是目标通道数
        # 这里需要在类初始化时添加这个压缩层
        output_final = self.channel_compress(output_concat)
        
        # 额外添加一个残差连接，将压缩后的结果与原始输出相加
        output_final = output_final + output
        
        return output_final

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
        
        # 注意力模块
        self.cbam = CBAM(out_ch)
        
        # 边缘增强模块
        self.edge_enhance = EdgeEnhanceModule(out_ch)
        
        # 添加通道压缩层 - 将concat后的3倍通道压缩回原始out_ch
        self.channel_compress = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

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
        
        # 残差连接
        output = hx1d + hxin
        
        # 改进：应用注意力机制和边缘增强前先concat
        # 使用通道注意力机制增强关键特征
        output_cbam = self.cbam(output)
        
        # 使用边缘增强模块强化边缘信息
        output_edge = self.edge_enhance(output)
        
        # 将原始输出、注意力增强输出和边缘增强输出concat
        # 这里需要增加一个额外的卷积层来压缩通道
        output_concat = torch.cat((output, output_cbam, output_edge), dim=1)
        
        # 压缩通道数，假设out_ch是目标通道数
        # 这里需要在类初始化时添加这个压缩层
        output_final = self.channel_compress(output_concat)
        
        # 额外添加一个残差连接，将压缩后的结果与原始输出相加
        output_final = output_final + output
        
        return output_final

### 添加RSU6G模块 ###
class RSU6G(nn.Module):
    """针对晶界分割优化的RSU6模块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6G, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        # 编码器部分
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        # 瓶颈层使用空洞卷积
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        # 解码器部分
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        
        # 转置卷积上采样层
        self.upsample5d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample4d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample3d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        self.upsample2d = TransposeUpsampler(mid_ch, mid_ch, scale_factor=2)
        
        # 注意力模块
        self.cbam = CBAM(out_ch)
        
        # 边缘增强模块
        self.edge_enhance = EdgeEnhanceModule(out_ch)
        
        # 添加通道压缩层 - 将concat后的3倍通道压缩回原始out_ch
        self.channel_compress = nn.Sequential(
            nn.Conv2d(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        # 编码器
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        
        # 瓶颈层
        hx6 = self.rebnconv6(hx5)
        
        # 解码器 - 使用转置卷积上采样
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = self.upsample5d(hx5d, target_size=hx4.shape[2:])
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upsample4d(hx4d, target_size=hx3.shape[2:])
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upsample3d(hx3d, target_size=hx2.shape[2:])
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upsample2d(hx2d, target_size=hx1.shape[2:])
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        # 残差连接
        output = hx1d + hxin
        
        # 改进：应用注意力机制和边缘增强前先concat
        # 使用通道注意力机制增强关键特征
        output_cbam = self.cbam(output)
        
        # 使用边缘增强模块强化边缘信息
        output_edge = self.edge_enhance(output)
        
        # 将原始输出、注意力增强输出和边缘增强输出concat
        # 这里需要增加一个额外的卷积层来压缩通道
        output_concat = torch.cat((output, output_cbam, output_edge), dim=1)
        
        # 压缩通道数，假设out_ch是目标通道数
        # 这里需要在类初始化时添加这个压缩层
        output_final = self.channel_compress(output_concat)
        
        # 额外添加一个残差连接，将压缩后的结果与原始输出相加
        output_final = output_final + output
        
        return output_final

##### U²-Net-Grain: 针对晶界分割优化的网络 ####
class U2NET_GRAIN(nn.Module):
    """针对晶界分割优化的U²-Net - 增强版本with Dense连接"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_GRAIN, self).__init__()
        
        # 编码器 - 通道数扩大一倍
        self.stage1 = RSU6G(in_ch, 32, 64)      # 修改为RSU6G
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU5G(64, 32, 128)        # 修改为RSU5G
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(128, 64, 256)       # 修改为RSU4G
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(256, 128, 512)     # 修改为RSU4FG
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4FG(512, 256, 512)     # 保持RSU4FG
        
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
        # stage4d: 输入为stage5上采样(512) + stage4(512) = 1024
        self.stage4d = RSU4FG(1024, 128, 256)
        
        # stage3d: 输入为stage4d上采样(256) + stage3(256) = 512
        self.stage3d = RSU4G(512, 64, 128)
        
        # stage2d: 输入为stage3d上采样(128) + stage2(128) + dense3_up(128) = 384
        self.stage2d = RSU5G(384, 32, 64)       # 修改为RSU5G
        
        # stage1d: 输入为stage2d上采样(64) + stage1(64) + dense2_up(64) = 192
        self.stage1d = RSU6G(192, 32, 64)       # 修改为RSU6G
        
        # 转置卷积上采样层 - 用于解码器
        self.upsample5to4 = TransposeUpsampler(512, 512, scale_factor=2)
        self.upsample4to3 = TransposeUpsampler(256, 256, scale_factor=2)
        self.upsample3to2 = TransposeUpsampler(128, 128, scale_factor=2)
        self.upsample2to1 = TransposeUpsampler(64, 64, scale_factor=2)
        
        # Dense连接的转置卷积上采样层
        self.dense3_upsample = TransposeUpsampler(128, 128, scale_factor=2)  # dense3 -> stage2 size
        self.dense2_upsample = TransposeUpsampler(64, 64, scale_factor=2)    # dense2 -> stage1 size
        
        # 侧输出的转置卷积上采样层
        self.side2_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=2)   # side2 -> side1 size
        self.side3_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=4)   # side3 -> side1 size
        self.side4_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=8)   # side4 -> side1 size
        self.side5_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=16)  # side5 -> side1 size
        
        # 侧输出层 - 通道数相应增加
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        # 最终融合层
        self.outconv = nn.Conv2d(5*out_ch, out_ch, 1)       # 5*out_ch
        
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
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        
        # Dense连接特征处理
        dense1 = self.dense_conv1(hx1)  # 64->32
        dense2 = self.dense_conv2(hx2)  # 128->64
        dense3 = self.dense_conv3(hx3)  # 256->128
        
        # 解码器 - 使用转置卷积上采样
        hx5up = self.upsample5to4(hx5, target_size=hx4.shape[2:])
        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))    # 512+512=1024
        
        hx4dup = self.upsample4to3(hx4d, target_size=hx3.shape[2:])
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))    # 256+256=512
        
        hx3dup = self.upsample3to2(hx3d, target_size=hx2.shape[2:])
        dense3_up = self.dense3_upsample(dense3, target_size=hx2.shape[2:])  # 将dense3上采样到hx2尺寸
        hx2d = self.stage2d(torch.cat((hx3dup, hx2, dense3_up), 1))  # 128+128+128=384
        
        hx2dup = self.upsample2to1(hx2d, target_size=hx1.shape[2:])
        dense2_up = self.dense2_upsample(dense2, target_size=hx1.shape[2:])  # 将dense2上采样到hx1尺寸
        hx1d = self.stage1d(torch.cat((hx2dup, hx1, dense2_up), 1))  # 64+64+64=192
        
        # 侧输出 - 使用转置卷积上采样
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = self.side2_upsample(d2, target_size=d1.shape[2:])
        
        d3 = self.side3(hx3d)
        d3 = self.side3_upsample(d3, target_size=d1.shape[2:])
        
        d4 = self.side4(hx4d)
        d4 = self.side4_upsample(d4, target_size=d1.shape[2:])
        
        d5 = self.side5(hx5)
        d5 = self.side5_upsample(d5, target_size=d1.shape[2:])
        
        # 特征融合
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
        
        # 全局边缘增强
        d0 = self.global_edge_enhance(d0)
        
        # 特征增强
        d0 = self.feature_enhance(d0) + d0  # 残差连接
        
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)

### U²-Net-Grain 轻量版 ###
class U2NETP_GRAIN(nn.Module):
    """轻量版晶界分割网络 - 增强版本with Dense连接"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP_GRAIN, self).__init__()
        
        # 轻量版也适当增加通道数，并修改对应模块
        self.stage1 = RSU6G(in_ch, 16, 32)      # 修改为RSU6G
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU5G(32, 16, 64)         # 修改为RSU5G
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU4G(64, 32, 128)        # 修改为RSU4G
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4FG(128, 64, 128)      # 修改为RSU4FG
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4FG(128, 64, 128)      # 保持RSU4FG
        
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
        # stage4d: 输入为stage5上采样(128) + stage4(128) = 256
        self.stage4d = RSU4FG(256, 32, 64)
        
        # stage3d: 输入为stage4d上采样(64) + stage3(128) = 192
        self.stage3d = RSU4G(192, 32, 64)
        
        # stage2d: 输入为stage3d上采样(64) + stage2(64) + dense3_up(32) = 160
        self.stage2d = RSU5G(160, 16, 32)       # 修改为RSU5G
        
        # stage1d: 输入为stage2d上采样(32) + stage1(32) + dense2_up(32) = 96
        self.stage1d = RSU6G(96, 16, 32)        # 修改为RSU6G
        
        # 转置卷积上采样层 - 轻量版
        self.upsample5to4 = TransposeUpsampler(128, 128, scale_factor=2)
        self.upsample4to3 = TransposeUpsampler(64, 64, scale_factor=2)
        self.upsample3to2 = TransposeUpsampler(64, 64, scale_factor=2)
        self.upsample2to1 = TransposeUpsampler(32, 32, scale_factor=2)
        
        # Dense连接的转置卷积上采样层 - 轻量版
        self.dense3_upsample = TransposeUpsampler(64, 32, scale_factor=2)   # dense3 -> stage2 size
        self.dense2_upsample = TransposeUpsampler(32, 32, scale_factor=2)   # dense2 -> stage1 size
        
        # 侧输出的转置卷积上采样层 - 轻量版
        self.side2_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=2)   # side2 -> side1 size
        self.side3_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=4)   # side3 -> side1 size
        self.side4_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=8)   # side4 -> side1 size
        self.side5_upsample = TransposeUpsampler(out_ch, out_ch, scale_factor=16)  # side5 -> side1 size
        
        # 侧输出层
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(128, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(5*out_ch, out_ch, 1)       # 5*out_ch
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
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        
        # Dense连接特征处理
        dense1 = self.dense_conv1(hx1)  # 32->16
        dense2 = self.dense_conv2(hx2)  # 64->32
        dense3 = self.dense_conv3(hx3)  # 128->64
        
        # 解码器 - 使用转置卷积上采样
        hx5up = self.upsample5to4(hx5, target_size=hx4.shape[2:])
        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))    # 128+128=256
        
        hx4dup = self.upsample4to3(hx4d, target_size=hx3.shape[2:])
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))    # 64+128=192
        
        hx3dup = self.upsample3to2(hx3d, target_size=hx2.shape[2:])
        dense3_up = self.dense3_upsample(dense3, target_size=hx2.shape[2:])  # 将dense3上采样到hx2尺寸
        hx2d = self.stage2d(torch.cat((hx3dup, hx2, dense3_up), 1))  # 64+64+32=160
        
        hx2dup = self.upsample2to1(hx2d, target_size=hx1.shape[2:])
        dense2_up = self.dense2_upsample(dense2, target_size=hx1.shape[2:])  # 将dense2上采样到hx1尺寸
        hx1d = self.stage1d(torch.cat((hx2dup, hx1, dense2_up), 1))  # 32+32+32=96
        
        # 侧输出 - 使用转置卷积上采样
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = self.side2_upsample(d2, target_size=d1.shape[2:])
        
        d3 = self.side3(hx3d)
        d3 = self.side3_upsample(d3, target_size=d1.shape[2:])
        
        d4 = self.side4(hx4d)
        d4 = self.side4_upsample(d4, target_size=d1.shape[2:])
        
        d5 = self.side5(hx5)
        d5 = self.side5_upsample(d5, target_size=d1.shape[2:])
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
        d0 = self.global_edge_enhance(d0)
        
        # 特征增强
        d0 = self.feature_enhance(d0) + d0  # 残差连接
        
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5) 