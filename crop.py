from PIL import Image
import os

# crop padding (1024, 768) images to target (1000, 750)
folder = 'results/padding_output/'  # padding input to network output
output_folder = 'results/crop_to_1000_750/'  # target (1000, 750) output

for i in os.listdir(folder):
    image = Image.open(folder + i).convert('RGB')
    # image_width, image_height = image.size  # 原始尺寸 1000x750

    # 目标尺寸
    # target_width, target_height = 1024, 768

    new_img = image.crop((0, 0, 1000, 750))

    new_img.save(output_folder + i)
    print(i)
