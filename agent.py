import datetime
import random
import stat
import flappy_bird_gymnasium
import gymnasium
from sympy.plotting.backends.matplotlibbackend import matplotlib
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch
from torch import nn
import os


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

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
        self.env_make_params = hyperparameters['env_make_params']

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

            while not terminated:
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

                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))     
                    step +=1      

                state = new_state

            reward_per_episode.append(episode_reward)
            print(f"Episode {episode} finished with reward {episode_reward}")
            epsilon = max(self.epsilon_final, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if len(memory) > self.batch_size:
                #3self.train(memory, policy_dqn, target_dqn)
                mini_batch = memory.sample(self.batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step  > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step = 0
    
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
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
