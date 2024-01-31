from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--pic-dir', nargs='+', required=True, help='please give picture dir')
    parser.add_argument('--title', nargs='+')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    # 图片目录列表
    directories = [os.path.join('visual', args.dataset, args.pic_dir[i]) for i in range(len(args.pic_dir))]

    # 创建一个新的目录来保存结果
    output_dir = "concat_result_" + args.dataset
    imglist = os.listdir(directories[0])

    # 配置字体路径
    font_path = "./tools/TheanoDidot-Bold.ttf"
    # 遍历每张图片
    for filename in tqdm(imglist):
        # 加载图片
        images = [Image.open(os.path.join(dir, filename)) for dir in directories]
        # 计算拼接后的图片尺寸
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths) * 3
        total_height = sum(sorted(heights, reverse=True)[:2])

        # 创建一个新的空白图片用于拼接
        result = Image.new('RGB', (total_width, total_height))

        # 拼接图片
        x_offset = 0
        y_offset = 0
        for j, img in enumerate(images):
            if j == 3:  # 第二行的开始
                x_offset = 0
                y_offset = max(heights)  # 第一行的最大高度
            result.paste(img, (x_offset, y_offset))
            
            font_size = img.size[0]//10
            font = ImageFont.truetype(font_path, font_size)
            # 在左上角添加自定义文本
            draw = ImageDraw.Draw(result)
            text = "({})".format(args.title[j])  # 根据需要修改文本)
            draw.text((x_offset, y_offset), text, fill=(255, 255, 255), font=font)

            x_offset += img.size[0]
            if j == 2:  # 第一行结束
                x_offset = max(widths)  # 从第二列开始

        # 保存拼接后的图片
        os.makedirs(os.path.join('visual', args.dataset, output_dir), exist_ok=True)
        result.save(os.path.join('visual', args.dataset, output_dir, f"{filename}"))

    print("concatted all picture！")
