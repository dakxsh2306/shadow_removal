from PIL import Image
import os

# padding input images to (1024, 768)
dir = '//ShadowDataset/ntire25_sh_rem_test_inp/LQ/'
dir_save = '//ShadowDataset/ntire25_sh_rem_test_inp/LQ_padding/'

for i in os.listdir(dir):

    # 打开图像
    image = Image.open(dir+i).convert('RGB')  # 替换为你的图像路径
    image_width, image_height = image.size  # 原始尺寸 1000x750

    # 目标尺寸
    target_width, target_height = 1024, 768

    # 计算需要填充的像素数
    pad_width = target_width - image_width  # 1024 - 1000 = 24
    pad_height = target_height - image_height  # 768 - 750 = 18

    # 创建一个目标尺寸的空白图像（填充值为 0）
    padded_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    # 将原始图像粘贴到填充图像的左上角
    padded_image.paste(image, (0, 0))

    # 保存填充后的图像
    padded_image.save(dir_save+i)
