import h5py
import cv2
import numpy as np

def main():
    # 打开HDF5文件
    demo_name = "demo_0"
    h5_filename = "./pushT_image_obs_0430180114.hdf5"
    with h5py.File(h5_filename, 'r') as f:
        # 读取图像序列
        pixels = f['data'][demo_name]['obs']['pixels'][:]
        # 读取动作轨迹并进行缩放
        actions = f['data'][demo_name]['actions'][:] * (96/512)  # 将坐标从512x512缩放到96x96
        # 获取视频参数
        num_frames, height, width, channels = pixels.shape
        
        # 创建视频写入器，使用不同的编码器和参数
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用 H.264 编码器
        out = cv2.VideoWriter(
            'demo_0.mp4', 
            fourcc, 
            10.0, 
            (width, height),
            isColor=True
        )
        
        # 写入每一帧
        for i, frame in enumerate(pixels):
            # OpenCV使用BGR格式，需要从RGB转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 创建轨迹图层
            trajectory_layer = np.zeros_like(frame_bgr)
            
            # 绘制从开始到当前时刻的所有轨迹点
            for j in range(i + 1):
                if j < len(actions):
                    # 获取历史轨迹点坐标并取整
                    x, y = np.round(actions[j]).astype(int)
                    # 确保坐标在图像范围内
                    if 0 <= x < width and 0 <= y < height:
                        # 绘制轨迹点，当前点用大红点，历史点用小红点
                        if j == i:
                            pass
                        else:
                            cv2.circle(trajectory_layer, (x, y), 2, (100, 100, 255), -1)  # 历史点
            
            # 将轨迹叠加到原图上
            frame_with_trajectory = cv2.addWeighted(frame_bgr, 0.3  , trajectory_layer, 0.9, 0)
            
            # 写入帧
            out.write(frame_with_trajectory)
            
            # 显示进度
            if i % 10 == 0:
                print(f"\r处理进度: {i+1}/{num_frames}", end="")
        
        # 释放视频写入器
        out.release()
        print("\n")
        print(f"视频已保存为 demo_0.mp4")
        print(f"总帧数: {num_frames}")
        print(f"分辨率: {width}x{height}")
        print(f"轨迹点数: {len(actions)}")

if __name__ == "__main__":
    main()