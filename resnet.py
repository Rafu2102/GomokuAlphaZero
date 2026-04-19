import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 全局注意力模組"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 降維成 [B, C, 1, 1] 的全局特徵
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 兩層全連接層產生通道注意力權重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # [B, C]
        y = self.fc(y).view(b, c, 1, 1) # [B, C, 1, 1]
        # 權重乘回原特徵圖，進行注意力過濾
        return x * y.expand_as(x)

class ResNetSEBlock(nn.Module):
    """嵌有 SE Block 的基本殘差模組"""
    def __init__(self, channels):
        super(ResNetSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 將全局注意力置於殘差相加之前
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 通過鷹眼過濾器
        out = self.se(out)
        
        out += residual
        out = F.relu(out)
        return out

class PolicyValueNet(nn.Module):
    """五子棋 AlphaZero v7: 10層 ResNet-SE 三頭神經網路 (Policy + Value + Aux)"""
    def __init__(self, board_width=15, board_height=15, num_channels=128, num_res_blocks=10):
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # 1. 視覺萃取層 (4ch 純棋盤 → 128 通道特徵)
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 2. 骨幹大腦：10 層 ResNet + SE
        self.res_blocks = nn.ModuleList([
            ResNetSEBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 3. 策略頭 (Policy Head)
        # 用 1x1 卷積將厚度降至 2 通道，再展平成 225 維的機率分佈
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_width * board_height, board_width * board_height)
        
        # 4. 價值頭 (Value Head)
        # 避免資訊過度截斷，降至 2 通道並擴充中繼層
        self.value_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 5. 輔助預測頭 (Auxiliary Threat Prediction Head)
        # 訓練時預測盤面威脅熱力圖，迫使 backbone 學會辨識棋型
        # 推論時不呼叫 (return_aux=False)，零額外成本
        # 僅 129 個新參數 (128 weights + 1 bias)
        self.aux_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=True)

    def forward(self, x, return_aux=False):
        """
        前向推論。
        Args:
            x: [B, 4, 15, 15] 棋盤觀測
            return_aux: True=訓練模式(回傳 policy, value, aux)
                        False=推論模式(只回傳 policy, value，跳過 aux head)
        """
        # x shape: [B, 4, 15, 15]
        
        # 輸入層處理
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 經過骨幹大腦
        for block in self.res_blocks:
            x = block(x)
            
        # --- 策略分岔 ---
        p = F.relu(self.policy_bn(self.policy_conv(x))) 
        p = p.flatten(1)  # [B, 450]
        p = self.policy_fc(p)  # [B, 225]
        policy_out = F.log_softmax(p, dim=1) 
        
        # --- 價值分岔 ---
        v = F.relu(self.value_bn(self.value_conv(x))) 
        v = v.flatten(1)  # [B, 450]
        v = F.relu(self.value_fc1(v))  # [B, 256]
        v = torch.tanh(self.value_fc2(v))  # [B, 1]
        
        if not return_aux:
            return policy_out, v  # MCTS 推論路徑：零額外成本
        
        # --- 輔助分岔（僅訓練時啟用）---
        # 從 backbone 特徵直接預測威脅熱力圖 [B, 15, 15]
        aux = torch.sigmoid(self.aux_conv(x)).squeeze(1)  # [B, 15, 15]
        return policy_out, v, aux

if __name__ == '__main__':
    # 效能與規格驗證腳本
    net = PolicyValueNet()
    dummy_input = torch.randn(32, 4, 15, 15)  # 4ch 純棋盤
    
    print("=== 推論模式測試 (return_aux=False) ===")
    policy, value = net(dummy_input)
    print(f"策略頭輸出形狀 (Log Probs): {policy.shape}  -> 預期: [32, 225]")
    print(f"價值頭輸出形狀 (Win Rate):   {value.shape}  -> 預期: [32, 1]")
    assert policy.shape == (32, 225)
    assert value.shape == (32, 1)
    
    print("\n=== 訓練模式測試 (return_aux=True) ===")
    policy, value, aux = net(dummy_input, return_aux=True)
    print(f"輔助頭輸出形狀 (Threats):    {aux.shape}  -> 預期: [32, 15, 15]")
    assert aux.shape == (32, 15, 15)
    assert (aux >= 0).all() and (aux <= 1).all(), "Aux output should be in [0, 1]"
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n模型總參數數量: {total_params:,}")
    print("resnet.py 所有測試通過！")
