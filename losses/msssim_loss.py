import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
#from piqa import MS_SSIM


# 定义MS-SSIM损失
class MSSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, win_size=11, channel=3):
        super(MSSSIMLoss, self).__init__()
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            channel=channel,
        )

    def forward(self, img1, img2):
        # MS-SSIM返回的是相似度值，1 - MS-SSIM作为损失
        return 1 - self.ms_ssim(img1, img2)

# 示例用法
if __name__ == "__main__":
    # 创建随机图像数据 (batch_size, channel, height, width)
    img1 = torch.rand(4, 3, 256, 256)  # 预测图像
    img2 = torch.rand(4, 3, 256, 256)  # 真实图像

    # 初始化MS-SSIM损失
    loss_fn = MSSSIMLoss(data_range=1.0, channel=3)

    # 计算损失
    loss = loss_fn(img1, img2)
    print("MS-SSIM Loss:", loss.item())