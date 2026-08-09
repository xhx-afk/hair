#version  3
""""
### . 提升训练速度
- 向量化操作 ：使用 F.unfold 替代双层循环，实现GPU并行计算
- 降低分辨率 ：将计算分辨率从128x128降至96x96，减少计算量
- 增加计算间隔 ：将 compute_interval 从2调整为5，减少计算频率
### 2. 改善边界损失
- 优化边界掩码计算 ：增加膨胀参数，添加备用边界计算
- 确保边界区域存在 ：当头发与人脸没有交集时，使用头发边缘作为边界
### 3. 平衡FID分数
- 降低自相似性权重 ：将 lambda_self_sim 从0.1调整为0.05
- 减少自相似性约束 ：通过降低权重和增加计算间隔，减少对多样性的影响
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightSelfSimilarityMixer(nn.Module):
    """
    轻量级自相似性混合模块
    针对单张V100优化，降低计算开销
    只关注头发区域的自相似性
    """
    
    def __init__(self, window_size=5, compute_interval=5):
        super().__init__()
        self.window_size = window_size
        self.compute_interval = compute_interval
        self.step_counter = 0
        
        # 轻量级特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def compute_self_similarity(self, image, hair_mask):
        """
        在降低的分辨率上计算自相似性
        
        Args:
            image: 输入图像 [B, 3, H, W]
            hair_mask: 头发区域mask [B, 1, H, W]
            
        Returns:
            similarity_map: 自相似性图 [B, H, W, window_size, window_size]
        """
        B, C, H, W = image.shape
        
        # 确保特征提取器在正确的设备上
        if self.feature_extractor[0].weight.device != image.device:
            self.feature_extractor.to(image.device)
        
        # 降低分辨率到96x96以减少计算
        target_size = 96
        if H > target_size:
            small_image = F.interpolate(image, size=(target_size, target_size), mode='bilinear', align_corners=False)
            small_mask = F.interpolate(hair_mask, size=(target_size, target_size), mode='nearest')
            scale_factor = H / target_size
        else:
            small_image = image
            small_mask = hair_mask
            scale_factor = 1.0
        
        # 提取特征
        features = self.feature_extractor(small_image)  # [B, 32, H', W']
        hair_features = features * small_mask
        
        # 计算局部自相似性 - 使用向量化操作
        pad = self.window_size // 2
        padded_features = F.pad(hair_features, (pad, pad, pad, pad), mode='reflect')
        
        B, C, H_s, W_s = hair_features.shape
        
        # 使用unfold提取所有窗口
        windows = F.unfold(padded_features, kernel_size=self.window_size, padding=0)
        # 重塑为 [B, C, window_size, window_size, H_s, W_s]
        windows = windows.view(B, C, self.window_size, self.window_size, H_s, W_s)
        
        # 提取中心特征 [B, C, 1, 1, H_s, W_s]
        center = hair_features.unsqueeze(2).unsqueeze(3)
        # 扩展维度以匹配windows的形状
        center = center.expand(-1, -1, self.window_size, self.window_size, -1, -1)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(center, windows, dim=1)
        # similarity的形状是 [B, window_size, window_size, H_s, W_s]
        # 重塑为 [B, H_s, W_s, window_size, window_size]
        similarity_map = similarity.permute(0, 3, 4, 1, 2)
        
        # 应用头发掩码
        similarity_map = similarity_map * small_mask.squeeze(1).unsqueeze(3).unsqueeze(4)
        
        # 如果降低了分辨率，需要上采样回原始尺寸
        if scale_factor > 1:
            B, H_s, W_s, W1, W2 = similarity_map.shape
            # 将similarity_map重塑为 [B, W1, W2, H_s, W_s] 以便上采样空间维度
            similarity_map = similarity_map.permute(0, 3, 4, 1, 2)
            # 重塑为 [B, W1*W2, H_s, W_s]
            similarity_map = similarity_map.reshape(B, W1 * W2, H_s, W_s)
            # 上采样空间维度
            similarity_map = F.interpolate(
                similarity_map,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            # 重塑为 [B, W1, W2, H, W]
            similarity_map = similarity_map.view(B, W1, W2, H, W)
            # 重塑回 [B, H, W, W1, W2]
            similarity_map = similarity_map.permute(0, 2, 3, 1, 4)
        
        return similarity_map
    
    def self_similarity_loss(self, generated_image, color_reference, hair_mask, step):
        """
        计算头发区域的自相似性损失
        
        Args:
            generated_image: 生成的图像 [B, 3, H, W]
            color_reference: 颜色参考图像 [B, 3, H, W]
            hair_mask: 头发区域mask [B, 1, H, W]
            step: 当前训练步数
            
        Returns:
            loss: 自相似性损失
        """
        # 每隔指定步数计算一次
        if step % self.compute_interval != 0:
            return torch.tensor(0.0, device=generated_image.device)
        
        # 检查头发区域大小
        hair_area = hair_mask.sum() / (hair_mask.shape[0] * hair_mask.shape[2] * hair_mask.shape[3])
        if hair_area < 0.05:  # 头发区域太小，跳过
            return torch.tensor(0.0, device=generated_image.device)
        
        # 计算生成图像的自相似性
        gen_similarity = self.compute_self_similarity(generated_image, hair_mask)
        
        # 计算颜色参考图像的自相似性
        color_similarity = self.compute_self_similarity(color_reference, hair_mask)
        
        # 计算损失
        loss = F.mse_loss(gen_similarity, color_similarity)
        
        return loss


class SelfSimilarityBlendingLoss(nn.Module):
    """
    结合自相似性的颜色混合损失
    修正版本：确保损失计算与原始训练目标一致
    """
    
    def __init__(self, lambda_self_sim=0.05, lambda_color=1.0, lambda_boundary=0.05):
        super().__init__()
        self.lambda_self_sim = lambda_self_sim
        self.lambda_color = lambda_color
        self.lambda_boundary = lambda_boundary
        
        self.self_similarity_mixer = LightweightSelfSimilarityMixer(window_size=5, compute_interval=5)
        
    def forward(self, generated_image, color_reference, hair_mask, face_mask, step=0):
        """
        计算综合损失
        
        Args:
            generated_image: 生成的图像 [B, 3, H, W]
            color_reference: 颜色参考图像 [B, 3, H, W]
            hair_mask: 头发区域mask [B, 1, H, W]
            face_mask: 人脸区域mask [B, 1, H, W]
            step: 当前训练步数
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量的字典
        """
        # 1. 颜色一致性损失（仅在头发区域）
        color_loss = F.mse_loss(
            generated_image * hair_mask, 
            color_reference * hair_mask
        )
        
        # 2. 自相似性损失（仅用于头发区域）
        self_sim_loss = self.self_similarity_mixer.self_similarity_loss(
            generated_image, color_reference, hair_mask, step
        )
        
        # 3. 边界平滑损失
        boundary_mask = self.compute_boundary_mask(hair_mask, face_mask)
        
        # 计算梯度
        gen_grad_x = torch.abs(generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1])
        gen_grad_y = torch.abs(generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :])
        
        color_grad_x = torch.abs(color_reference[:, :, :, 1:] - color_reference[:, :, :, :-1])
        color_grad_y = torch.abs(color_reference[:, :, 1:, :] - color_reference[:, :, :-1, :])
        
        # 在边界区域计算梯度差异
        boundary_loss_x = F.mse_loss(
            gen_grad_x * boundary_mask[:, :, :, 1:], 
            color_grad_x * boundary_mask[:, :, :, 1:]
        ) if boundary_mask[:, :, :, 1:].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
        
        boundary_loss_y = F.mse_loss(
            gen_grad_y * boundary_mask[:, :, 1:, :], 
            color_grad_y * boundary_mask[:, :, 1:, :]
        ) if boundary_mask[:, :, 1:, :].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
        
        boundary_loss = boundary_loss_x + boundary_loss_y
        
        # 4. 总损失
        total_loss = (self.lambda_self_sim * self_sim_loss + 
                     self.lambda_color * color_loss + 
                     self.lambda_boundary * boundary_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'self_similarity_loss': self_sim_loss.item(),
            'color_loss': color_loss.item(),
            'boundary_loss': boundary_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_boundary_mask(self, hair_mask, face_mask, dilation=5):
        """
        计算头发与人脸的边界mask
        
        Args:
            hair_mask: 头发mask [B, 1, H, W]
            face_mask: 人脸mask [B, 1, H, W]
            dilation: 边界膨胀大小
            
        Returns:
            boundary_mask: 边界mask [B, 1, H, W]
        """
        # 膨胀头发mask
        kernel_size = dilation * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                           device=hair_mask.device)
        dilated_hair = F.conv2d(hair_mask.float(), kernel, padding=dilation) > 0
        
        # 方法1：边界是膨胀后的头发与人脸的交集
        boundary = (dilated_hair.float() * face_mask).clamp(0, 1)
        
        # 方法2：如果交集为空，使用头发边缘作为边界
        # 获取头发边缘（膨胀后减去原始）
        hair_edge = (dilated_hair.float() - hair_mask).clamp(0, 1)
        
        # 合并两种边界：如果交集为空，用头发边缘
        boundary = torch.where(boundary.sum(dim=(2, 3), keepdim=True) > 0, 
                              boundary, 
                              hair_edge)
        
        return boundary




##version 2
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
#class LightweightSelfSimilarityMixer(nn.Module):
#    """
#    轻量级自相似性混合模块
#    针对单张V100优化，降低计算开销
#    只关注头发区域的自相似性
#    """
#    
#    def __init__(self, window_size=5, compute_interval=2):
#        super().__init__()
#        self.window_size = window_size
#        self.compute_interval = compute_interval
#        self.step_counter = 0
#        
#        # 轻量级特征提取器
#        self.feature_extractor = nn.Sequential(
#            nn.Conv2d(3, 32, kernel_size=3, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, kernel_size=3, padding=1),
#            nn.ReLU()
#        )
#        
#    def compute_self_similarity(self, image, hair_mask):
#        """
#        在降低的分辨率上计算自相似性
#        
#        Args:
#            image: 输入图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            
#        Returns:
#            similarity_map: 自相似性图 [B, H, W, window_size, window_size]
#        """
#        B, C, H, W = image.shape
#        
#        # 确保特征提取器在正确的设备上
#        if self.feature_extractor[0].weight.device != image.device:
#            self.feature_extractor.to(image.device)
#        
#        # 降低分辨率到128x128以减少计算
#        if H > 128:
#            small_image = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
#            small_mask = F.interpolate(hair_mask, size=(128, 128), mode='nearest')
#            scale_factor = H / 128.0
#        else:
#            small_image = image
#            small_mask = hair_mask
#            scale_factor = 1.0
#        
#        # 提取特征
#        features = self.feature_extractor(small_image)  # [B, 32, H', W']
#        hair_features = features * small_mask
#        
#        # 计算局部自相似性
#        pad = self.window_size // 2
#        padded_features = F.pad(hair_features, (pad, pad, pad, pad), mode='reflect')
#        
#        B, C, H_s, W_s = hair_features.shape
#        similarity_map = torch.zeros(B, H_s, W_s, self.window_size, self.window_size, 
#                                    device=image.device, dtype=torch.float32)
#        
#        # 只在头发区域计算
#        for b in range(B):
#            hair_indices = torch.where(small_mask[b, 0] > 0.5)
#            for i, j in zip(hair_indices[0], hair_indices[1]):
#                center_feat = hair_features[b, :, i, j].unsqueeze(1).unsqueeze(2)  # [C, 1, 1]
#                i_start, i_end = i, i + self.window_size
#                j_start, j_end = j, j + self.window_size
#                neighborhood = padded_features[b, :, i_start:i_end, j_start:j_end]  # [C, window, window]
#                
#                # 计算余弦相似度
#                similarity = F.cosine_similarity(center_feat, neighborhood, dim=0)
#                similarity_map[b, i, j] = similarity
#        
#        # 如果降低了分辨率，需要上采样回原始尺寸
#        if scale_factor > 1:
#            # 重塑为4D张量进行上采样
#            B, H_s, W_s, W1, W2 = similarity_map.shape
#            similarity_map = similarity_map.view(B, H_s * W_s, W1, W2)
#            similarity_map = F.interpolate(similarity_map, size=(W1, W2), mode='bilinear', align_corners=False)
#            similarity_map = similarity_map.view(B, H_s, W_s, W1, W2)
#            # 再次上采样空间维度
#            similarity_map = F.interpolate(
#                similarity_map.permute(0, 3, 4, 1, 2).reshape(B, W1 * W2, H_s, W_s),
#                size=(H, W),
#                mode='bilinear',
#                align_corners=False
#            )
#            similarity_map = similarity_map.view(B, W1, W2, H, W).permute(0, 3, 4, 1, 2)
#        
#        return similarity_map
#    
#    def self_similarity_loss(self, generated_image, color_reference, hair_mask, step):
#        """
#        计算头发区域的自相似性损失
#        
#        Args:
#            generated_image: 生成的图像 [B, 3, H, W]
#            color_reference: 颜色参考图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            step: 当前训练步数
#            
#        Returns:
#            loss: 自相似性损失
#        """
#        # 每隔指定步数计算一次
#        if step % self.compute_interval != 0:
#            return torch.tensor(0.0, device=generated_image.device)
#        
#        # 检查头发区域大小
#        hair_area = hair_mask.sum() / (hair_mask.shape[0] * hair_mask.shape[2] * hair_mask.shape[3])
#        if hair_area < 0.05:  # 头发区域太小，跳过
#            return torch.tensor(0.0, device=generated_image.device)
#        
#        # 计算生成图像的自相似性
#        gen_similarity = self.compute_self_similarity(generated_image, hair_mask)
#        
#        # 计算颜色参考图像的自相似性
#        color_similarity = self.compute_self_similarity(color_reference, hair_mask)
#        
#        # 计算损失
#        loss = F.mse_loss(gen_similarity, color_similarity)
#        
#        return loss
#
#
#class SelfSimilarityBlendingLoss(nn.Module):
#    """
#    结合自相似性的颜色混合损失
#    修正版本：确保损失计算与原始训练目标一致
#    """
#    
#    def __init__(self, lambda_self_sim=0.1, lambda_color=1.0, lambda_boundary=0.05):
#        super().__init__()
#        self.lambda_self_sim = lambda_self_sim
#        self.lambda_color = lambda_color
#        self.lambda_boundary = lambda_boundary
#        
#        self.self_similarity_mixer = LightweightSelfSimilarityMixer(window_size=5, compute_interval=2)
#        
#    def forward(self, generated_image, color_reference, hair_mask, face_mask, step=0):
#        """
#        计算综合损失
#        
#        Args:
#            generated_image: 生成的图像 [B, 3, H, W]
#            color_reference: 颜色参考图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            face_mask: 人脸区域mask [B, 1, H, W]
#            step: 当前训练步数
#            
#        Returns:
#            total_loss: 总损失
#            loss_dict: 各损失分量的字典
#        """
#        # 1. 颜色一致性损失（仅在头发区域）
#        color_loss = F.mse_loss(
#            generated_image * hair_mask, 
#            color_reference * hair_mask
#        )
#        
#        # 2. 自相似性损失（仅用于头发区域）
#        self_sim_loss = self.self_similarity_mixer.self_similarity_loss(
#            generated_image, color_reference, hair_mask, step
#        )
#        
#        # 3. 边界平滑损失
#        boundary_mask = self.compute_boundary_mask(hair_mask, face_mask)
#        
#        # 计算梯度
#        gen_grad_x = torch.abs(generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1])
#        gen_grad_y = torch.abs(generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :])
#        
#        color_grad_x = torch.abs(color_reference[:, :, :, 1:] - color_reference[:, :, :, :-1])
#        color_grad_y = torch.abs(color_reference[:, :, 1:, :] - color_reference[:, :, :-1, :])
#        
#        # 在边界区域计算梯度差异
#        boundary_loss_x = F.mse_loss(
#            gen_grad_x * boundary_mask[:, :, :, 1:], 
#            color_grad_x * boundary_mask[:, :, :, 1:]
#        ) if boundary_mask[:, :, :, 1:].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
#        
#        boundary_loss_y = F.mse_loss(
#            gen_grad_y * boundary_mask[:, :, 1:, :], 
#            color_grad_y * boundary_mask[:, :, 1:, :]
#        ) if boundary_mask[:, :, 1:, :].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
#        
#        boundary_loss = boundary_loss_x + boundary_loss_y
#        
#        # 4. 总损失
#        total_loss = (self.lambda_self_sim * self_sim_loss + 
#                     self.lambda_color * color_loss + 
#                     self.lambda_boundary * boundary_loss)
#        
#        loss_dict = {
#            'total_loss': total_loss.item(),
#            'self_similarity_loss': self_sim_loss.item(),
#            'color_loss': color_loss.item(),
#            'boundary_loss': boundary_loss.item()
#        }
#        
#        return total_loss, loss_dict
#    
#    def compute_boundary_mask(self, hair_mask, face_mask, dilation=3):
#        """
#        计算头发与人脸的边界mask
#        
#        Args:
#            hair_mask: 头发mask [B, 1, H, W]
#            face_mask: 人脸mask [B, 1, H, W]
#            dilation: 边界膨胀大小
#            
#        Returns:
#            boundary_mask: 边界mask [B, 1, H, W]
#        """
#        # 膨胀头发mask
#        kernel = torch.ones(1, 1, dilation*2+1, dilation*2+1, 
#                           device=hair_mask.device)
#        dilated_hair = F.conv2d(hair_mask, kernel, padding=dilation) > 0
#        
#        # 边界是膨胀后的头发与人脸的交集
#        boundary = (dilated_hair.float() * face_mask).clamp(0, 1)
#        
#        return boundary


#version 1
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
#class LightweightSelfSimilarityMixer(nn.Module):
#    """
#    轻量级自相似性混合模块
#    针对单张V100优化，降低计算开销
#    """
#    
#    def __init__(self, window_size=5, compute_interval=2):
#        super().__init__()
#        self.window_size = window_size
#        self.compute_interval = compute_interval
#        self.step_counter = 0
#        
#        # 轻量级特征提取器
#        self.feature_extractor = nn.Sequential(
#            nn.Conv2d(3, 32, kernel_size=3, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(32, 32, kernel_size=3, padding=1),
#            nn.ReLU()
#        )
#        
#    def compute_self_similarity(self, image, hair_mask):
#        """
#        在降低的分辨率上计算自相似性
#        
#        Args:
#            image: 输入图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            
#        Returns:
#            similarity_map: 自相似性图 [B, H, W, window_size, window_size]
#        """
#        B, C, H, W = image.shape
#        
#        # 确保特征提取器在正确的设备上
#        if self.feature_extractor[0].weight.device != image.device:
#            self.feature_extractor.to(image.device)
#        
#        # 降低分辨率到128x128以减少计算
#        if H > 128:
#            small_image = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
#            small_mask = F.interpolate(hair_mask, size=(128, 128), mode='nearest')
#            scale_factor = H / 128.0
#        else:
#            small_image = image
#            small_mask = hair_mask
#            scale_factor = 1.0
#        
#        # 提取特征
#        features = self.feature_extractor(small_image)  # [B, 32, H', W']
#        hair_features = features * small_mask
#        
#        # 计算局部自相似性
#        pad = self.window_size // 2
#        padded_features = F.pad(hair_features, (pad, pad, pad, pad), mode='reflect')
#        
#        B, C, H_s, W_s = hair_features.shape
#        similarity_map = torch.zeros(B, H_s, W_s, self.window_size, self.window_size, 
#                                    device=image.device, dtype=torch.float32)
#        
#        # 只在头发区域计算
#        for b in range(B):
#            hair_indices = torch.where(small_mask[b, 0] > 0.5)
#            for i, j in zip(hair_indices[0], hair_indices[1]):
#                center_feat = hair_features[b, :, i, j].unsqueeze(1).unsqueeze(2)  # [C, 1, 1]
#                i_start, i_end = i, i + self.window_size
#                j_start, j_end = j, j + self.window_size
#                neighborhood = padded_features[b, :, i_start:i_end, j_start:j_end]  # [C, window, window]
#                
#                # 计算余弦相似度
#                similarity = F.cosine_similarity(center_feat, neighborhood, dim=0)
#                similarity_map[b, i, j] = similarity
#        
#        # 如果降低了分辨率，需要上采样回原始尺寸
#        if scale_factor > 1:
#            # 重塑为4D张量进行上采样
#            B, H_s, W_s, W1, W2 = similarity_map.shape
#            similarity_map = similarity_map.view(B, H_s * W_s, W1, W2)
#            similarity_map = F.interpolate(similarity_map, size=(W1, W2), mode='bilinear', align_corners=False)
#            similarity_map = similarity_map.view(B, H_s, W_s, W1, W2)
#            # 再次上采样空间维度
#            similarity_map = F.interpolate(
#                similarity_map.permute(0, 3, 4, 1, 2).reshape(B, W1 * W2, H_s, W_s),
#                size=(H, W),
#                mode='bilinear',
#                align_corners=False
#            )
#            similarity_map = similarity_map.view(B, W1, W2, H, W).permute(0, 3, 4, 1, 2)
#        
#        return similarity_map
#    
#    def self_similarity_loss(self, generated_image, target_image, hair_mask, step):
#        """
#        计算自相似性损失
#        
#        Args:
#            generated_image: 生成的图像 [B, 3, H, W]
#            target_image: 目标图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            step: 当前训练步数
#            
#        Returns:
#            loss: 自相似性损失
#        """
#        # 每隔指定步数计算一次
#        if step % self.compute_interval != 0:
#            return torch.tensor(0.0, device=generated_image.device)
#        
#        # 检查头发区域大小
#        hair_area = hair_mask.sum() / (hair_mask.shape[0] * hair_mask.shape[2] * hair_mask.shape[3])
#        if hair_area < 0.05:  # 头发区域太小，跳过
#            return torch.tensor(0.0, device=generated_image.device)
#        
#        # 计算自相似性
#        gen_similarity = self.compute_self_similarity(generated_image, hair_mask)
#        target_similarity = self.compute_self_similarity(target_image, hair_mask)
#        
#        # 计算损失
#        loss = F.mse_loss(gen_similarity, target_similarity)
#        
#        return loss
#
#
#class SelfSimilarityBlendingLoss(nn.Module):
#    """
#    结合自相似性的颜色混合损失
#    """
#    
#    def __init__(self, lambda_self_sim=0.2, lambda_color=1.0, lambda_boundary=0.1):
#        super().__init__()
#        self.lambda_self_sim = lambda_self_sim
#        self.lambda_color = lambda_color
#        self.lambda_boundary = lambda_boundary
#        
#        self.self_similarity_mixer = LightweightSelfSimilarityMixer(window_size=5, compute_interval=2)
#        
#    def forward(self, generated_image, target_image, hair_mask, face_mask, step=0):
#        """
#        计算综合损失
#        
#        Args:
#            generated_image: 生成的图像 [B, 3, H, W]
#            target_image: 目标图像 [B, 3, H, W]
#            hair_mask: 头发区域mask [B, 1, H, W]
#            face_mask: 人脸区域mask [B, 1, H, W]
#            step: 当前训练步数
#            
#        Returns:
#            total_loss: 总损失
#            loss_dict: 各损失分量的字典
#        """
#        # 1. 颜色一致性损失（仅在头发区域）
#        color_loss = F.mse_loss(
#            generated_image * hair_mask, 
#            target_image * hair_mask
#        )
#        
#        # 2. 自相似性损失（轻量级）
#        self_sim_loss = self.self_similarity_mixer.self_similarity_loss(
#            generated_image, target_image, hair_mask, step
#        )
#        
#        # 3. 边界平滑损失
#        boundary_mask = self.compute_boundary_mask(hair_mask, face_mask)
#        
#        # 计算梯度
#        gen_grad_x = torch.abs(generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1])
#        gen_grad_y = torch.abs(generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :])
#        
#        target_grad_x = torch.abs(target_image[:, :, :, 1:] - target_image[:, :, :, :-1])
#        target_grad_y = torch.abs(target_image[:, :, 1:, :] - target_image[:, :, :-1, :])
#        
#        # 在边界区域计算梯度差异
#        boundary_loss_x = F.mse_loss(
#            gen_grad_x * boundary_mask[:, :, :, 1:], 
#            target_grad_x * boundary_mask[:, :, :, 1:]
#        ) if boundary_mask[:, :, :, 1:].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
#        
#        boundary_loss_y = F.mse_loss(
#            gen_grad_y * boundary_mask[:, :, 1:, :], 
#            target_grad_y * boundary_mask[:, :, 1:, :]
#        ) if boundary_mask[:, :, 1:, :].sum() > 0 else torch.tensor(0.0, device=generated_image.device)
#        
#        boundary_loss = boundary_loss_x + boundary_loss_y
#        
#        # 4. 总损失
#        total_loss = (self.lambda_self_sim * self_sim_loss + 
#                     self.lambda_color * color_loss + 
#                     self.lambda_boundary * boundary_loss)
#        
#        loss_dict = {
#            'total_loss': total_loss.item(),
#            'self_similarity_loss': self_sim_loss.item(),
#            'color_loss': color_loss.item(),
#            'boundary_loss': boundary_loss.item()
#        }
#        
#        return total_loss, loss_dict
#    
#    def compute_boundary_mask(self, hair_mask, face_mask, dilation=3):
#        """
#        计算头发与人脸的边界mask
#        
#        Args:
#            hair_mask: 头发mask [B, 1, H, W]
#            face_mask: 人脸mask [B, 1, H, W]
#            dilation: 边界膨胀大小
#            
#        Returns:
#            boundary_mask: 边界mask [B, 1, H, W]
#        """
#        # 膨胀头发mask
#        kernel = torch.ones(1, 1, dilation*2+1, dilation*2+1, 
#                           device=hair_mask.device)
#        dilated_hair = F.conv2d(hair_mask, kernel, padding=dilation) > 0
#        
#        # 边界是膨胀后的头发与人脸的交集
#        boundary = (dilated_hair.float() * face_mask).clamp(0, 1)
#        
#        return boundary