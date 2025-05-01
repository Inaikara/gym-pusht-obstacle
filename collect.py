import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
import zarr
import os
from utils.visualize_demo import visualize_demo
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
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="human")
    observation, info = env.reset()
    
    # 初始化时钟以控制帧率
    clock = pygame.time.Clock()
    running = True
    teleop = False
    
    # 创建outputs文件夹（如果不存在）
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/base")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建zarr文件用于保存所有演示数据
    timestamp = datetime.now().strftime('%m%d%H%M%S')
    zarr_path = os.path.join(output_dir, f"pushT_base_{timestamp}.zarr")
    root = zarr.open(zarr_path, mode='w')
    
    # 用于存储每个回合的数据
    demo_count = 0
    all_actions = []
    all_states = []
    all_images = []
    
    def save_demo():
        nonlocal demo_count, all_actions, all_states, all_images
        
        if len(all_actions) == 0:
            return
            
        # 如果是第一个demo，创建数据集
        if demo_count == 0:
            root.create_dataset('action', data=np.array(all_actions), chunks=True, dtype=np.float32)
            root.create_dataset('state', data=np.array(all_states), chunks=True, dtype=np.float32)
            root.create_dataset('img', data=np.array(all_images), chunks=True, dtype=np.uint8)
        else:
            # 追加数据到已有数据集
            root['action'].append(np.array(all_actions))
            root['state'].append(np.array(all_states))
            root['img'].append(np.array(all_images))
        
        print(f"保存了demo_{demo_count}")
        
        # 重置数据并增加demo计数
        all_actions.clear()
        all_states.clear()
        all_images.clear()
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
                        observation, info = env.reset()
                        teleop = False
                        all_actions.clear()
                        all_states.clear()
                        all_images.clear()
            
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
                
                # 保存观测
                all_images.append(observation.get('pixels', np.zeros((96, 96, 3), dtype=np.uint8)))
                # 保存agent_pos作为state
                state = info.get('pos_agent', np.zeros(2))
                all_states.append(state)
                all_actions.append(action)
                
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
        # zarr doesn't need explicit closing
        env.close()
        visualize_demo(zarr_path)
        print(f"共保存了{demo_count}个回合的数据到文件 {zarr_path}")

if __name__ == "__main__":
    main()