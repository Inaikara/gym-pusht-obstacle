import zarr
import cv2
import numpy as np
import os

def visualize_demo(zarr_path: str):
    # 创建outputs文件夹（如果不存在）
    output_dir = os.path.join(os.path.dirname(zarr_path), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(zarr_path))[0] + '.mp4')
    
    # 打开zarr文件
    root = zarr.open(zarr_path, mode='r')
    
    # 获取数据集
    actions = root['action'][:]* (96/512)
    states = root['state'][:]
    images = root['img'][:]
    
    # 获取视频参数
    num_frames, height, width, channels = images.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
        output_filename,
        fourcc,
        10.0,
        (width, height),
        isColor=True
    )
    
    print(f"\n开始处理演示数据")
    
    # 写入每一帧
    for i, frame in enumerate(images):
        # OpenCV使用BGR格式，需要从RGB转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 创建轨迹图层
        trajectory_layer = np.zeros_like(frame_bgr)
        
        # 绘制从开始到当前时刻的所有轨迹点
        for j in range(i + 1):
            if j > i - 16:  # 只显示最近的16个轨迹点
                # 获取历史轨迹点坐标并取整
                x, y = np.round(actions[j]).astype(int)
                cv2.circle(trajectory_layer, (x, y), 2, (255, 0, 0), -1)
        
        # 将轨迹叠加到原图上
        frame_with_trajectory = cv2.addWeighted(frame_bgr, 0.3, trajectory_layer, 1, 0)
        
        # 写入帧
        out.write(frame_with_trajectory)
    
    # 释放视频写入器
    out.release()
    print(f"\n\n处理完成，视频已保存为 {output_filename}")

if __name__ == "__main__":
    visualize_demo("outputs/pushT_image_obs_0501145900.zarr")