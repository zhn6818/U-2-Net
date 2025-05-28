import os
from PIL import Image
 
def convert_png_to_jpg(directory):
    # 遍历指定目录
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # 检查文件是否以.png结尾
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开图像文件
            with Image.open(file_path) as img:
                # 构建新的文件名（去掉.png，加上.jpg）
                new_filename = filename.replace('.png', '.jpg')
                new_file_path = os.path.join(directory, new_filename)
                # 保存为JPG格式
                img.save(new_file_path, 'JPEG')
                print(f"Converted {filename} to {new_filename}")
 
# 指定包含PNG图像的文件夹路径
directory_path = '/data1/zhn/JLD/train_768/img/'
convert_png_to_jpg(directory_path)