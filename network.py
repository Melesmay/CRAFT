import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys  

'''
class LinearClassifierMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=128, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),                    # 非线性仍然用 GELU
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   # ← 方案 A
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):          # x 形状 [B] 或 [B,1]
        if x.ndim == 1:
            x = x.unsqueeze(-1)    # [B,1]
        return self.mlp(x)

'''
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=2):
        super(LinearClassifier, self).__init__()
        #_, feat_dim = model_dicts[name]
        #feat_dim = 2048
        
        # self.conv1 = nn.Conv2d(4, 32, 3)
        # self.conv2 = nn.Conv2d(32, 64, 3)
        # self.bn = nn.BatchNorm2d(64)

        # self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64, 2)

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)))

    def logits(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        #return self.logits(features)

def random_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean = 0., std = 0.01)
            if m.bias is not None:
                nn.init.normal_(m.weight, mean = 0., std = 0.01)

                
class LPIPSClassifierMLP(nn.Module):
    def __init__(self, hidden=32, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 2)           # 输出 logits
        )

    def forward(self, d):                 # d shape [B] or [B,1]
        if d.dim() == 1:
            d = d.unsqueeze(1)
        return self.net(d)

'''
class LPIPSClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2): # in_channels 之前可能是 1
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32x32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 16x16

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 8x8
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 将 8x8 降为 1x1
        self.classifier = nn.Linear(512 * 1 * 1, num_classes) # 输入通道数 * 1 * 1

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
'''
'''        
class LPIPSClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2): # Default in_channels might be 256 or 512 depending on LPIPS VGG layer
        super(LPIPSClassifierCNN, self).__init__()
        # Since input is expected to be [B, C, 1, 1], standard pooling layers will fail.
        # We should treat this as a feature vector [B, C] after flattening the 1x1,
        # and then use fully connected layers.
        
        # The 'conv_block' was causing the issue. Let's make it a sequence that
        # effectively processes the channels and then flattens.
        
        # If your LPIPS feature has a fixed channel dimension (e.g., 256 for LPIPS VGG)
        # this in_channels should match that.
        
        # Example structure:
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1), # Use 1x1 conv to process channels if desired
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            # No pooling layers here, as the spatial dimensions are 1x1
        )
        
        # After conv_block, the output will still be [B, 128, 1, 1]. Flatten it.
        self.classifier_head = nn.Sequential(
            nn.Flatten(), # Flattens [B, 128, 1, 1] to [B, 128]
            nn.Linear(128, 64), # Hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(64, num_classes) # Output layer
        )

    def forward(self, x):
        # x is expected to be [Batch_size, C, 1, 1]
        x = self.conv_block(x) # Output: [Batch_size, 128, 1, 1]
        x = self.classifier_head(x) # Output: [Batch_size, num_classes]
        return x
'''
'''
class LPIPSClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(LPIPSClassifierCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # [B, 1, H, W] -> [B, 16, H, W]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 16, H/2, W/2]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 32, 1, 1]
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 1, 1)
        elif x.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            x = x.unsqueeze(1)

        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten [B, 32, 1, 1] -> [B, 32]
        return self.fc(x)
'''
# class LPIPSClassifier(nn.Module):
#     def __init__(self, in_dim=1, hidden_dim=64, num_classes=2):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x):
#         if len(x.shape) > 2:
#             x = x.view(x.size(0), -1)  # Flatten if needed
#         return self.fc(x)

class ClassCenterBuffer:
    def __init__(self, latent_dim, num_classes, device):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device
        self.centers = torch.zeros(num_classes, latent_dim).to(device)
        self.counts = torch.zeros(num_classes).to(device)

    def update(self, latents, labels):
        latents_flat = latents.view(latents.size(0), -1)
        
        # for i in range(self.num_classes):
        #     mask = (labels == i)
        #     if mask.sum() > 0:
        #         class_latents = latents_flat[mask]
        #         self.centers[i] = (self.centers[i] * self.counts[i] + class_latents.sum(dim=0)) / (self.counts[i] + mask.sum())
        #         self.counts[i] += mask.sum()
        
        mask = (labels == 0)
        if mask.sum() > 0:
            class_latents = latents_flat[mask]
            self.centers[0] = (self.centers[0] * self.counts[0] + class_latents.sum(dim=0)) / (self.counts[0] + mask.sum())
            self.counts[0] += mask.sum()

    def get_opposite_direction(self, latents, labels):
        latents_flat = latents.view(latents.size(0), -1)
        directions = []
        for i in range(latents.size(0)):
            label = labels[i]
            center = self.centers[label]
            dir_vec = latents_flat[i] - center
            directions.append(dir_vec)
        return torch.stack(directions).view_as(latents)

    def get_opposite_direction_no_label(self, latents, alpha=0.5):
        B, C, H, W = latents.shape
        latents_flat = latents.view(B, -1)  # [B, D]
        centers_flat = self.centers[0]  # [B, D]

        direction = latents_flat - centers_flat  # [B, D]
        direction = F.normalize(direction, dim=1) * alpha

        return direction.view(B, C, H, W)

    def get_opposite_direction_from_class0_center(self, latents, alpha=0.5):
        """
        Calculates a perturbation direction for all samples, pushing them away
        from the center of class 0, regardless of their actual label.
        The alpha parameter directly scales the normalized direction.
        """
        B, C, H, W = latents.shape
        latents_flat = latents.view(B, -1)  # [B, D]
        
        # Ensure center[0] is not all zeros if it hasn't been updated yet
        if self.counts[0] == 0:
            print("Warning: Class 0 center not yet updated. Perturbation direction will be from origin.")
            # If center is zero, direction will be from origin, which might still be useful.
            centers_flat = torch.zeros_like(self.centers[0]) # Use zero if no center is available
        else:
            centers_flat = self.centers[0] # [D]
        
        # Direction vector = current latent - class 0 center
        # Subtracting this `direction` from `latents` will push *away* from `centers_flat`.
        direction = latents_flat - centers_flat.unsqueeze(0) # [B, D] - [1, D] broadcast

        # Normalize the direction vector for each sample and then scale by alpha
        norm = torch.norm(direction, dim=1, keepdim=True) # [B, 1]
        
        # Avoid division by zero for samples where norm is very small
        # For those, the direction remains zero.
        direction_normalized_scaled = torch.zeros_like(direction)
        non_zero_norm_mask = (norm > 1e-8).squeeze(1) # [B]
        
        if non_zero_norm_mask.any():
            direction_normalized_scaled[non_zero_norm_mask] = \
                direction[non_zero_norm_mask] / norm[non_zero_norm_mask] * alpha

        return direction_normalized_scaled.view(B, C, H, W)
'''
class ContrastiveReconstructionLoss(nn.Module):
    def __init__(self, base_loss='mse', margin=0.3):
        super().__init__()
        if base_loss == 'mse':
            self.metric = nn.MSELoss(reduction='none')
        elif base_loss == 'lpips':
            import lpips
            self.metric = lpips.LPIPS(net='vgg')
        else:
            raise NotImplementedError
        self.margin = margin 

    def forward(self, input_img, recon_img, label):  # label: 0 (real), 1 (fake)
        diff = self.metric(input_img, recon_img)  # [B, C, H, W] or [B, 1]
        if len(diff.shape) > 2:
            diff = diff.mean(dim=[1, 2, 3])
        
        real_loss = diff[label == 0]
        fake_loss = diff[label == 1]

        loss_real = real_loss.mean() if real_loss.numel() > 0 else 0
        loss_fake = torch.relu(self.margin - fake_loss).mean() if fake_loss.numel() > 0 else 0

        return loss_real + loss_fake
'''

class ContrastiveReconstructionLoss(nn.Module):
    def __init__(self, base_loss='mse', margin=0.4, delta=0.3, lambda_over=0.5):
        super().__init__()
        if base_loss == 'mse':
            self.metric = nn.MSELoss(reduction='none')
        elif base_loss == 'lpips':
            import lpips
            self.metric = lpips.LPIPS(net='vgg')
        else:
            raise NotImplementedError

        self.margin = margin
        self.delta = delta
        self.lambda_over = lambda_over

    def forward(self, input_img, recon_img, label):       # label: 0(real) 1(fake)
        diff = self.metric(input_img, recon_img)          # [B,C,H,W] or [B,1]
        if diff.dim() > 2:
            diff = diff.mean(dim=[1, 2, 3])               # -> [B]

        loss_real = diff[label == 0].mean() if (label == 0).any() else 0.0

        fake_diff = diff[label == 1]
        if fake_diff.numel() == 0:
            loss_fake = 0.0
        else:
            low_term  = torch.relu(self.margin - fake_diff)                  # < margin
            high_term = torch.relu(fake_diff - (self.margin + self.delta))   # > margin+δ
            loss_fake = low_term.mean() + self.lambda_over * high_term.mean()

        return loss_real + loss_fake
