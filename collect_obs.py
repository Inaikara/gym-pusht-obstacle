import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
import h5py
import os
from datetime import datetime

def main():
    """
    通过鼠标控制 PushT 环境的演示脚本
    
    操作说明:
    - 将鼠标移到蓝色圆圈附近开始控制
    - 通过鼠标移动来推动 T 形方块到绿色目标区域
    - 按 'Q' 退出程序
    - 按 'R' 重新开始
    """

    env = gym.make("gym_pusht/PushT-Obstacle", render_mode="human",cover_obstacle = True)

    observation, info = env.reset()
    
    # 初始化时钟以控制帧率
    clock = pygame.time.Clock()
    running = True
    teleop = False
    
    # 创建HDF5文件用于保存所有演示数据
    timestamp = datetime.now().strftime('%m%d%H%M%S')
    h5_filename = f"pushT_image_obs_{timestamp}.hdf5"
    h5_file = h5py.File(h5_filename, 'w')
    
    # 创建data组
    data_group = h5_file.create_group("data")
    
    # 用于存储每个回合的数据
    demo_count = 0
    
    # 当前回合的数据
    actions = []
    observations = {
        'pixels': [],
        'obstacle_pos': []
    }
    
    def save_demo():
        nonlocal actions, observations, demo_count, data_group
        
        if len(actions) == 0:
            return
            
        # 创建新的demo组
        demo_group = data_group.create_group(f"demo_{demo_count}")
        
        # 保存actions
        actions_array = np.array(actions)
        demo_group.create_dataset("actions", data=actions_array)
        
        # 保存obs
        obs_group = demo_group.create_group("obs")
        for key in observations:
            if observations[key]:  # 确保有数据
                obs_group.create_dataset(key, data=np.array(observations[key]))
        
        print(f"保存了demo_{demo_count}")
        
        # 重置数据并增加demo计数
        actions.clear()
        for key in observations:
            observations[key].clear()
        demo_count += 1
    
    try:
        while running:
            # 处理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # 按 Q 退出
                        running = False
                        break
                    elif event.key == pygame.K_r:  # 按 R 重置环境
                        # 保存当前回合数据
                        save_demo()
                        # 重置环境
                        observation, info = env.reset()
                        teleop = False
            
            if not running:
                break
                
            # 获取鼠标位置
            mouse_pos = pygame.mouse.get_pos()
            distance = np.linalg.norm(mouse_pos - info['pos_agent'])
            
            if teleop or distance < 30:
                teleop = True
                action = mouse_pos
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                
                # 保存数据
                observations['pixels'].append(observation.get('pixels', np.zeros((96, 96, 3), dtype=np.uint8)))
                observations['obstacle_pos'].append(info.get('pos_obstacle', np.zeros(2)))
                actions.append(info.get('pos_agent', np.zeros(2)))
                
                # 渲染环境
                image = env.render()
                
                # 如果回合结束则重置环境
                if terminated or truncated:
                    if terminated:
                        save_demo()
                    observation, info = env.reset()
                    teleop = False
                    
            # 控制帧率（10Hz） 
            clock.tick(10)
    finally:
        # 确保在任何情况下都关闭HDF5文件
        h5_file.close()
        env.close()
        print(f"共保存了{demo_count}个回合的数据到文件 {h5_filename}")

if __name__ == "__main__":
    main()