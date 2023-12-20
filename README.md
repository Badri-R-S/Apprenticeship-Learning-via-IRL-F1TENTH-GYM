# Apprenticeship Learning via IRL using F1TENTH Gym ROS
Implementation of Apprenticeship Learning via IRL on the F1TENTH GYM environment, that runs on ROS2 Foxy. The environment consists of the ego vehicle that provides information such as LIDAR scan data, velocities and positions via Rviz2.

## Environment Installation
Follow the instructions provided in: https://github.com/f1tenth/f1tenth_gym_ros.

## Setup and Methododology
- The F1TENTH Gym utilizes a containerized ROS communication bridge, transforming it into a ROS2 simulation compatible with systems running WSL and Ubuntu 20.04. The simulation leverages ROS2 Foxy and provides access to critical information through ROS topics, including the robot's position, velocity, and LIDAR scan data distances.
- To define an effective reward function for the RL
agent, the team considered the robot's linear velocity
Vx, the minimum LIDAR scan distance, LIDmin, and
angular velocity wz. The reward function is expressed
as follows: 𝑅 = 𝛼.𝑉𝑥 + 𝛽.LID𝑚in + 𝛾.𝑤𝑧 
- Here, α,β,γ are randomly initialized weight parameters. The minimum LIDAR distance is
calculated over 10 scans ranging from 0 to 270 degrees. Initial values for the weights were set at 0.5, 0.1, and -0.02, respectively. 
- The training process commenced with Behavioral Cloning, utilizing expert data consisting of 26,000 rows of (state, actions) pairs.
- A model was trained using this data to generate trajectories, to maximize the average reward on a given track. The average reward for both expert and model-generated trajectories was calculated, and an iterative application of gradient ascent on the reward function aimed to reduce the discrepancy.
- Transitioning to RL training, Proximal Policy Optimization (PPO) was chosen. To expedite
convergence, the team used the model generated from Behavioral Cloning as the baseline, progressively updating its parameters during PPO training. 
- This iterative combination of Behavioral Cloning and RL allowed the model to converge more efficiently. Additional expert trajectories were incorporated in stages to train the model effectively, resulting in a refined RL agent.
- The training process spanned nearly 15 million
iterations, incorporating 160,000 state-action pairs
from expert data. This hybrid approach, merging
Behavioral Cloning and RL in an iterative manner,
proved instrumental in achieving an RL agent within a
reasonable timeframe.

## Results
- After the 15 million iterations of training, the final stages of training iterations looked as follows:
<p align="center">
  <img src = "https://github.com/Badri-R-S/Apprenticeship-Learning-via-IRL-F1TENTH-GYM/blob/master/Results/Final_training_iterations.mp4"
    </p>
- The policy was tested on the same map and the result is as follows:

- The most important result is the generaization of the policy for different maps. This is shown as below:
