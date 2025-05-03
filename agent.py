from datetime import datetime, timedelta
import random
import stat
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

import numpy as np
import torch
from torch import nn
import os
import argparse

import matplotlib
import matplotlib.pyplot as plt

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#print("Current working directory:", os.getcwd())
script_dir = os.path.dirname(__file__)  # directory containing agent.py
yaml_path = os.path.join(script_dir, 'hyperparameters.yml')


class Agent:

    def __init__(self, hyperparameter_set):
        with open(yaml_path, "r") as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']

        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")


    def run(self, is_training = True, render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f'{start_time.strftime(DATE_FORMAT)}: training starting...'
            print(log_message)
            with open(self.LOG_FILE, "w") as f:
                f.write(log_message + '\n')

        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env =gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []
        
        policy_dqn = DQN(num_states, num_actions, hidden_dim =self.fc1_nodes).to(device)


        if is_training:
            epsilon =self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            
            target_dqn = DQN(num_states, num_actions,self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            epsilon_history = []
            step =0

            best_reward = -99999999

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0

            while (not terminated and episode_reward < self.stop_on_reward):
                #env.render()
                #action = env.action_space.sample()
                #state, reward, terminated, _, info = env.step(action):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1,2,3...]) ==> tensor([[1,2,3...]])
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                    # with training, we will get better action, no need to randomize the action

                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step+=1
                state = new_state
            
            reward_per_episode.append(episode_reward)



            if is_training:
                if episode_reward > best_reward:
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: new reward: {episode_reward:0.1f}({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model...'
                    print(log_message)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_message + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graphs(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.batch_size:
                    #self.train(memory, policy_dqn, target_dqn)
                    mini_batch = memory.sample(self.batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                    epsilon_history.append(epsilon)

                    if step  > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step = 0
    
    def save_graphs(self, rewards_per_episode, epsilon_history):
        fig =plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        ## replace this part for a more efficient pytorch implementation
        #for state, action, new_state, reward, terminated in mini_batch:
        #    if terminated:
        #        target = reward
        #    else:
        #        with torch.no_grad():
        #            target_q = reward + self.gamma * target_dqn(new_state.unsqueeze(dim=0)).squeeze().max()


        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device) ## True/False to 1/0

        with torch.no_grad():
            target_q = rewards + self.gamma * (1 - terminations) * target_dqn(new_states).max(dim=1)[0]
            #target_q = target_q.unsqueeze(dim=1)
            #target_q = target_q * torch.eye(2)[actions].to(device) # one hot encoding of actions


        current_q = policy_dqn(states).gather(1, actions.unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    #agent = Agent("cartpole1")
    #agent.run(is_training=True, render=True)
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
