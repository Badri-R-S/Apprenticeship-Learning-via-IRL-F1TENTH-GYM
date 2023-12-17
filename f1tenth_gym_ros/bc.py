import gym
from gym import spaces
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from f1_openai_bridge import f110_gym
from torch.utils.data.dataset import Dataset, random_split


model_dir = "model_dir"
log_dir = "log_dir"

class ExpertDataSet(Dataset):
   def __init__(self, expert_observations, expert_actions):
      self.observations = expert_observations
      self.actions = expert_actions
   def __getitem__(self, index):
      return (self.observations[index], self.actions[index])
   def __len__(self):
      return len(self.observations)
   

def pretrain_agent(student,gym_env,train_expert_dataset, test_expert_dataset,batch_size=64,epochs=1000,
scheduler_gamma=0.7,learning_rate=1.0,log_interval=100,
   no_cuda=True,seed=1,test_batch_size=64):
    use_cuda = True
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    if isinstance(gym_env.action_space, spaces.Box):
       criterion = nn.MSELoss()
    else:
       criterion = nn.CrossEntropyLoss()
    # Extract initial policy
    #student = A2C('MlpPolicy', env=gym_env, verbose=1, device='cuda', tensorboard_log=log_dir)
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
         data, target = data.to(device), target.to(device)
         optimizer.zero_grad()
         if isinstance(gym_env.action_space, spaces.Box):
            # A2C/PPO policy outputs actions, values, log_prob
            # SAC/TD3 policy outputs actions only
            if isinstance(student, (A2C, PPO)):
               action, _, _ = model(data)
            else:
               #SAC/TD3
               action = model(data)
            action_prediction = action.double()
         else:
            # Retrieve the logits for A2C/PPO when using discrete actions
            dist = model.get_distribution(data)
            action_prediction = dist.distribution.logits
            target = target.long()
         loss = criterion(action_prediction, target)
         loss.backward()
         optimizer.step()

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if isinstance(gym_env.action_space, gym.spaces.Box):
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
               # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()
                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
    
    # Implant the trained policy network back into the RL student agent
    return model





def main():
    gym_env = f110_gym()

    #ppo_expert = PPO('MlpPolicy', env=gym_env, verbose=1,device='cuda', tensorboard_log=log_dir)
    #ppo_expert.learn(total_timesteps=3e6)
    #ppo_expert.save(f"{model_dir}/{'PP0_expert'}")

    #mean_reward = evaluate_policy(ppo_expert,env=gym_env,n_eval_episodes=10)
    #print(f"Mean reward = {mean_reward}")

    pretrained_model = PPO.load('/home/badri/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/models/PP0_2350000.zip')
    a2c_student = PPO('MlpPolicy',env=gym_env, verbose=1,device='cuda', tensorboard_log=log_dir)
    a2c_student.set_parameters(pretrained_model.get_parameters())
    num_interactions = int(4e4)

    if isinstance(gym_env.action_space, spaces.Box):
        expert_observations = np.empty((num_interactions,)   +gym_env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + (gym_env.action_space.shape[0],))
    else:
        expert_observations = np.empty((num_interactions,) + gym_env.observation_space.shape)
        expert_actions = np.empty((num_interactions,) + gym_env.action_space.shape)

    obs = gym_env.reset()
    
    #for i in tqdm(range(num_interactions)):
    #    action, _ = ppo_expert.predict(obs, deterministic=True)
    #    expert_observations[i] = obs
    #    expert_actions[i] = action
    #    obs, reward, done, info = gym_env.step(action)
    #    if done:
    #        obs = gym_env.reset()

    #np.savez_compressed(
    #"expert_data",
    #expert_actions=expert_actions,
    #expert_observations=expert_observations,
    #)
    
    expert_actions = np.load('/home/badri/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/expert_actions.npy')
    expert_observations = np.load('/home/badri/sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/expert_observations.npy')

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
    )
    mean_reward= evaluate_policy(a2c_student, gym_env, n_eval_episodes=10)
    #Should be random.
    print(f"Mean reward = {mean_reward}")

    model = pretrain_agent( 
    a2c_student,
    gym_env,
    train_expert_dataset,
    test_expert_dataset,
    epochs=3,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    batch_size=64,
    test_batch_size=1000,
    )
    #Updating student network after getting a trained model
    gym_env.reset()
    a2c_student.policy = model
    a2c_student.save(f"{model_dir}/{'PPO_student_8'}")

    mean_reward = evaluate_policy(a2c_student, gym_env,n_eval_episodes=10)
    print(f"Mean reward = {mean_reward}")


if __name__ == '__main__':
    main()