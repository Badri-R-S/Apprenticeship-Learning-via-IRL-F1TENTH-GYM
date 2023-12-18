import pandas as pd
import numpy as np
from f1_openai_bridge import f110_gym

csv_file_path_ai = 'path to file'
csv_file_path = 'path to file'
df_ai = pd.read_csv(csv_file_path_ai)
df = pd.read_csv(csv_file_path)

print("Environment initialization")
env = f110_gym()

v_x_ai = df_ai['v_linear_x'].values
v_x = df['v_linear_x'].values
v_z_ai = df_ai['v_angular_z'].values
v_z = df['v_angular_z'].values
lidar_readings_ai = df_ai.iloc[:,5:15].values
lidar_readings = df.iloc[:,5:15].values
#print(lidar_readings)

speed_reward_ai = env.alpha * v_x_ai
speed_reward = env.alpha * v_x

angular_penalty = -env.gamma * v_z
angular_penalty_ai = -env.gamma * v_z_ai

min_lidar_readings_ai = np.min(lidar_readings_ai, axis=1)
min_lidar_readings = np.min(lidar_readings,axis=1)

#print(min_lidar_readings)
sum_lidar_readings_ai = np.sum(lidar_readings_ai, axis=1)
sum_lidar_readings = np.sum(lidar_readings,axis=1)


if np.any(min_lidar_readings_ai < 0.5):
    collision_penalty_ai = -1000
    reward_ai = collision_penalty_ai
if np.any(min_lidar_readings_ai < 1):
    collision_penalty_ai = -100.0 * (min_lidar_readings_ai / sum_lidar_readings_ai)
    reward_ai = speed_reward_ai + angular_penalty_ai + collision_penalty_ai
if np.any(min_lidar_readings_ai > 1):
    collision_penalty_ai = env.beta * sum_lidar_readings_ai
    reward_ai = speed_reward_ai + angular_penalty_ai + collision_penalty_ai
#print(reward_ai)
avg_reward_ai = np.mean(reward_ai)
print("ai_avg_reward:",avg_reward_ai)

if np.any(min_lidar_readings < 0.5):
    collision_penalty = -1000
    reward = collision_penalty
if np.any(min_lidar_readings < 1):
    collision_penalty = -100.0 * (min_lidar_readings/ sum_lidar_readings)
    reward = speed_reward + angular_penalty + collision_penalty
if np.any(min_lidar_readings > 1):
    collision_penalty = env.beta * sum_lidar_readings
    reward = speed_reward + angular_penalty + collision_penalty
avg_reward = np.mean(reward)
print("avg_reward_expert:",avg_reward)
print("Difference in average reward:", avg_reward - avg_reward_ai)
v_x_ai_avg = np.mean(v_x) - np.mean(v_x_ai)
print("Gradient with respect to alpha:",v_x_ai_avg)
v_z_ai_avg = np.mean(v_z) - np.mean(v_z_ai)
print("Gradient with respect to gamma:",v_z_ai_avg)
min_lidar_readings_ai_avg =np.mean(min_lidar_readings) -  np.mean(min_lidar_readings_ai)
print("Gradient with respect to Beta:", min_lidar_readings_ai_avg)
print("alpha:",env.alpha)
updated_alpha = env.alpha + (v_x_ai_avg*0.01)
print("Beta:",env.beta)
updated_beta = env.beta + min_lidar_readings_ai_avg*0.01
print("Gamma:",env.gamma)
updated_gamma = env.gamma + v_z_ai_avg*0.01

print("Updated_alpha:",updated_alpha)
print("Updated_beta:",updated_beta)
print("Updated_gamma",updated_gamma)
