from PIL import Image
import os

# merge 2 patchs
folder1 = 'results/crop1/'  # 替换为第一个文件夹路径
folder2 = 'results/crop2/'  # 替换为第二个文件夹路径
output_folder = 'results/cat'

# 获取两个文件夹中的文件列表
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

for file1, file2 in zip(files1, files2):
    # 确保文件名一致
    if file1 != file2:
        print(f"警告：文件名不一致 - {file1} 和 {file2}")
        continue

    # 读取图像
    image1 = Image.open(os.path.join(folder1, file1)).convert('RGB')
    image2 = Image.open(os.path.join(folder2, file2)).convert('RGB')

    # 确保图像高度一致
    if image1.size[1] != image2.size[1]:
        raise ValueError(f"图像高度不一致 - {file1}")

    # 计算拼接后的图像尺寸
    new_width = image1.size[0] + image2.size[0]
    new_height = max(image1.size[1], image2.size[1])

    # 创建空白图像
    new_image = Image.new('RGB', (new_width, new_height))

    # 将两张图像粘贴到空白图像中
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.size[0], 0))

    # 保存拼接后的图像
    output_path = os.path.join(output_folder, file1)
    new_image.save(output_path)
    print(f"已保存：{output_path}")

print("所有图像拼接完成！")

