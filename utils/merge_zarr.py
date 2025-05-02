import zarr
import numpy as np
import os
import glob
from typing import List
from replay_buffer import ReplayBuffer
from visualize_demo import visualize_demo

def merge_zarr_files(input_paths: List[str], output_path: str):
    """
    合并多个zarr文件的数据
    
    Args:
        input_paths: 输入zarr文件路径列表
        output_path: 输出zarr文件路径
    """
    # 检查输入文件是否存在
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到输入文件: {path}")
    
    # 创建输出缓冲区
    output_buffer = ReplayBuffer.create_empty_zarr()
    
    # 读取所有输入文件的数据
    for path in input_paths:
        print(f"正在读取: {path}")
        # 使用ReplayBuffer加载数据
        input_buffer = ReplayBuffer.create_from_path(path)
        
        # 遍历每个回合的数据并添加到输出缓冲区
        for i in range(input_buffer.n_episodes):
            episode_data = input_buffer.get_episode(i)
            # 添加回合数据到输出缓冲区
            output_buffer.add_episode(
                data=episode_data,
                chunks={
                    'action': (32, 2),
                    'state': (32, 4),
                    'img': (32, 96, 96, 3)
                },
                compressors={
                    'action': 'default',
                    'state': 'default',
                    'img': 'disk'  # 对图像数据使用更高压缩率
                }
            )
    
    # 保存合并后的数据
    output_buffer.save_to_path(output_path)
    
    print(f"\n合并完成！")
    print(f"合并后的回合数: {output_buffer.n_episodes}")
    print(f"合并后的总步数: {output_buffer.n_steps}")

def main(type:str):
    """
    主函数，用于处理命令行参数并执行合并
    """
    # 获取当前目录下的所有zarr文件
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs/",type)
    zarr_files = glob.glob(os.path.join(input_dir, "*.zarr"))
    
    if len(zarr_files) == 0:
        print("没有找到任何zarr文件！")
        return
    
    # 创建输出文件名
    output_path = os.path.join(input_dir, "pusht_"+type+".zarr")
    
    # 执行合并
    print(f"找到以下文件:")
    for f in zarr_files:
        print(f"- {os.path.basename(f)}")
    print(f"\n开始合并...")
    
    merge_zarr_files(zarr_files, output_path)
    visualize_demo(output_path)

if __name__ == "__main__":
    type = "obstacle" #obstacle
    main(type)