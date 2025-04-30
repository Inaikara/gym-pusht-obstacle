import h5py
import cv2
import numpy as np
import os

def visualize_demo(h5_filename: str):
    
    # 创建outputs文件夹（如果不存在）
    output_dir = os.path.join(os.path.dirname(h5_filename), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(h5_filename))[0] + '.mp4')
    
    with h5py.File(h5_filename, 'r') as f:
        # 获取所有demo的名称
        demo_names = list(f['data'].keys())
        print(f"找到以下demo: {demo_names}")
        
        # 获取视频参数（假设所有demo的图像尺寸相同）
        first_demo = f['data'][demo_names[0]]['obs']['pixels'][:]
        num_frames, height, width, channels = first_demo.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(
            output_filename,
            fourcc,
            10.0,
            (width, height),
            isColor=True
        )
        
        # 处理每个demo
        for demo_name in demo_names:
            print(f"\n开始处理 {demo_name}")
            
            # 读取图像序列和动作轨迹
            pixels = f['data'][demo_name]['obs']['pixels'][:]
            actions = f['data'][demo_name]['actions'][:] * (96/512)
            
            # 写入每一帧
            for i, frame in enumerate(pixels):
                # OpenCV使用BGR格式，需要从RGB转换
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 创建轨迹图层
                trajectory_layer = np.zeros_like(frame_bgr)
                
                # 绘制从开始到当前时刻的所有轨迹点
                for j in range(i + 1):
                    if j > i - 16:
                        # 获取历史轨迹点坐标并取整
                        x, y = np.round(actions[j]).astype(int)
                        cv2.circle(trajectory_layer, (x, y), 2, (255, 0, 0), -1)
                
                # 将轨迹叠加到原图上
                frame_with_trajectory = cv2.addWeighted(frame_bgr, 0.3, trajectory_layer, 1, 0)
                
                # 添加demo标识
                cv2.putText(frame_with_trajectory, 
                          demo_name, 
                          (5, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, 
                          (255, 255, 255), 
                          1)
                
                # 写入帧
                out.write(frame_with_trajectory)
        
        # 释放视频写入器
        out.release()
        print(f"\n\n所有demo处理完成，视频已保存为 {output_filename}")

if __name__ == "__main__":
    visualize_demo(h5_filename = "outputs/pushT_image_0430191254.hdf5")