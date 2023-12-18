from stable_baselines3 import DQN,PPO
from f1_openai_bridge import f110_gym
import rclpy
import threading
import numpy as np
import os
import csv

model_dir = "model_dir"
log_dir = "log_dir"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def ros_spin(env):
    rclpy.spin(env.f110_gym_node)

def train(gym_env):
    print("Training")
    path_to_model = 'path to file'
    pretrained_model = PPO.load(path_to_model)
    model = PPO('MlpPolicy',device='cuda',env=gym_env,verbose=1,tensorboard_log=log_dir)
    model.set_parameters(pretrained_model.get_parameters())
    model.learning_rate = 1e-3
    TIMESTEPS = 25000
    iters = 0
    while True:
        iters+=1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{'PP0'}_{TIMESTEPS*iters}")

def test(env):
    print("Testing")
    path_to_model = 'path to file'
    model = PPO.load(path_to_model,env = env)
    obs = env.reset()
    done = False
    extra_steps = 5
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _= env.step(action)
        if done:
            if extra_steps> 0:
                done = False
                extra_steps -=1
                obs = env.reset()
            else:
                obs = env.reset()
                break


def main():
    env = f110_gym()
    # Start the ROS spinning in a separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(env,))
    ros_thread.start()
    #train(env)
    test(env)
    # Wait for the ROS thread to finish
    ros_thread.join()

if __name__ == '__main__':
    main()