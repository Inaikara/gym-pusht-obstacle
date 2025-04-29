import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
from gym_pusht.envs.pusht_obstacle import Vec2d

def main():
    """
    通过鼠标控制 PushT 环境的演示脚本
    
    操作说明:
    - 将鼠标移到蓝色圆圈附近开始控制
    - 通过鼠标移动来推动 T 形方块到绿色目标区域
    - 按 'Q' 退出程序
    - 按 'R' 重新开始
    """
    
    # 创建环境
    env = gym.make("gym_pusht/PushT-v0", render_mode="human")
    observation, info = env.reset()
    
    # 初始化时钟以控制帧率
    clock = pygame.time.Clock()
    running = True
    teleop = False   # 替换 not_start 变量为 teleop
    
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
        
        if not running:
            break
            
        # 获取鼠标位置
        mouse_pos = pygame.mouse.get_pos()
        # 直接使用鼠标坐标作为动作
        mouse_position = Vec2d(*mouse_pos)
        
        # 计算鼠标位置与智能体位置的距离
        agent_pos = Vec2d(*observation[:2])
        if teleop or (mouse_position - agent_pos).length < 30:
            teleop = True
            action = np.array([mouse_position.x, mouse_position.y], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)        
            image = env.render()  
            
            # 如果回合结束则重置环境
            if terminated or truncated:
                observation, info = env.reset()
                teleop = False
                
        # 控制帧率（10Hz） 
        clock.tick(10)
    
    env.close()

if __name__ == "__main__":
    main()