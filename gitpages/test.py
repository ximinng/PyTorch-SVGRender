import glob
import os
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_gif(output_path)

def find_and_convert_mp4_to_gif(directory):
    # 递归地查找所有的MP4文件
    mp4_files = glob.glob(directory + '/**/*.mp4', recursive=True)

    # 遍历所有找到的MP4文件并进行转换
    for mp4_file in mp4_files:
        gif_file = mp4_file.rsplit('.', 1)[0] + '.gif'
        convert_mp4_to_gif(mp4_file, gif_file)
        print(f"Converted {mp4_file} to {gif_file}")

# 调用函数，传入你想要搜索的顶级目录
find_and_convert_mp4_to_gif('.')
