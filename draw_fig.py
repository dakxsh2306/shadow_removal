import matplotlib.pyplot as plt
import os
from PIL import Image

# 定义图像名称和文件夹路径
image_names = ['0000.png', '0013.png', '0039.png', '0068.png']  # 图像名称
folder_paths = ['ShadowDataset/ntire25_sh_rem_test_inp/LQ', 'results/crop_to_1000_750','results/final_test']  # 不同方法的文件夹路径

labels = ['Input', 'ShadowRefiner', 'Ours']
output_image_path = 'comparison_result.png'  # 输出图像路径

# 读取所有图像
images = []
for folder in folder_paths:
    folder_images = []
    for name in image_names:
        image_path = os.path.join(folder, name)
        img = Image.open(image_path)
        folder_images.append(img)
    images.append(folder_images)

# 创建画布
n_rows = len(image_names) + 1  # 行数：图像数量 + 1（用于方法名称）
n_cols = len(folder_paths)      # 列数：文件夹数量
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))

# 如果只有一行或一列，axes 的维度会变化，需要统一为二维数组
if n_rows == 1:
    axes = axes.reshape(1, -1)
if n_cols == 1:
    axes = axes.reshape(-1, 1)

# 设置子图之间的间隔
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # wspace 控制列间距，hspace 控制行间距

# 隐藏多余的子图
for ax in axes[-1]:
    ax.axis('off')

# 拼接图像
for i in range(len(image_names)):
    for j in range(len(folder_paths)):
        ax = axes[i, j]
        ax.imshow(images[j][i])  # 注意：images[j][i] 是第 j 个文件夹的第 i 张图像
        ax.axis('off')  # 隐藏坐标轴

# 在最后一行添加方法名称
for j, folder in enumerate(labels):
    ax = axes[-1, j]
    ax.text(0.5, 0.95, folder, fontsize=18, ha='center', va='top')
    ax.axis('off')

# 调整布局
plt.tight_layout()

# 保存结果图像
plt.savefig(output_image_path)

# 显示结果图像
plt.show()