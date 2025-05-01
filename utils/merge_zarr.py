import zarr
import numpy as np
import os
import glob
from typing import List

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
    
    # 创建输出文件
    root = zarr.open(output_path, mode='w')
    
    # 用于存储所有数据
    all_actions = []
    all_states = []
    all_images = []
    
    # 读取所有输入文件的数据
    for path in input_paths:
        print(f"正在读取: {path}")
        input_data = zarr.open(path, mode='r')
        
        # 读取数据
        actions = input_data['action'][:]
        states = input_data['state'][:]
        images = input_data['img'][:]
        
        # 添加到列表中
        all_actions.append(actions)
        all_states.append(states)
        all_images.append(images)
    
    # 合并数据
    merged_actions = np.concatenate(all_actions, axis=0)
    merged_states = np.concatenate(all_states, axis=0)
    merged_images = np.concatenate(all_images, axis=0)
    
    # 保存合并后的数据
    root.create_dataset('action', data=merged_actions, chunks=True, dtype=np.float32)
    root.create_dataset('state', data=merged_states, chunks=True, dtype=np.float32)
    root.create_dataset('img', data=merged_images, chunks=True, dtype=np.uint8)
    
    print(f"\n合并完成！")
    print(f"合并后的数据大小:")
    print(f"- 动作数据: {merged_actions.shape}")
    print(f"- 状态数据: {merged_states.shape}")
    print(f"- 图像数据: {merged_images.shape}")

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

if __name__ == "__main__":
    type = "base" #obstacle
    main(type)