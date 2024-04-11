import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torch.distributions import Categorical, Beta, Normal

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TrueHumanResponse():
    def __init__(self, valence_csv_path=None, arousal_csv_path=None,
                 num_subjects=18, seed=0):
        self.valence_csv_path = valence_csv_path
        self.arousal_csv_path = arousal_csv_path
        self.num_subjects = num_subjects
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def load_response(self, file_path, index):
        data = np.loadtxt(file_path, delimiter=",")
        return data[index, :9], data[index, 9], data[index, 10]
    
    def sample_task(self):
        random_index = np.random.randint(0, self.num_subjects)
        task = {"subject_id": random_index}
        task["val_coeffs"], task["val_mean"], task["val_std"] = self.load_response(self.valence_csv_path, random_index)
        task["aro_coeffs"], task["aro_mean"], task["aro_std"] = self.load_response(self.arousal_csv_path, random_index)
        return task
    
    def load_from_task(self, task):
        self._task      = task
        self.sub_id     = task["subject_id"]
        self.val_coeffs = task["val_coeffs"]
        self.val_mean   = task["val_mean"]
        self.val_std    = task["val_std"]
        self.aro_coeffs = task["aro_coeffs"]
        self.aro_mean   = task["aro_mean"]
        self.aro_std    = task["aro_std"]

    def sample_data(self, task, size=32):
        """
        self.sample_task returns the coefficients of one person's human response model.
        """
        # task = self.sample_task()
        self.load_from_task(task)

        move_speed_boundary = (27.8, 143.8)
        arm_speed_boundary  = (23.8, 109.1)
        choice_values       = torch.tensor([-1, 1])

        x_tensors_list = []
        y_tensors_list = []
        for _ in range(size):
            move_speed = torch.FloatTensor(1).uniform_(move_speed_boundary[0], move_speed_boundary[1])
            arm_speed  = torch.FloatTensor(1).uniform_(arm_speed_boundary[0], arm_speed_boundary[1])
            random_choices = choice_values[torch.randint(0, 2, (3,))]
            random_tensor = torch.cat((move_speed, arm_speed, random_choices), 0)

            current_task_M = torch.tensor([
                1, random_tensor[0] ** 2, random_tensor[0], random_tensor[1] ** 2, random_tensor[1], random_tensor[2],
                random_tensor[3], random_tensor[4], 1
            ])

            valence = torch.matmul(current_task_M, torch.from_numpy(self.val_coeffs).float()).unsqueeze(0)
            arousal = torch.matmul(current_task_M, torch.from_numpy(self.aro_coeffs).float()).unsqueeze(0)
            response = torch.cat((valence, arousal), 0)
            
            x_tensors_list.append(random_tensor)
            y_tensors_list.append(response)

        x_stacked_tensor = torch.stack(x_tensors_list) # shape (size, 5)
        y_stacked_tensor = torch.stack(y_tensors_list) # shape (size, 2)
        return x_stacked_tensor, y_stacked_tensor


class HumanResponseNet(nn.Module):
    """
    Predict human response (valance and arousal) based on the HRC robot state (5 dimension)
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=2, dropout_rate=0):  # dropout too aggressive for online learning
        super(HumanResponseNet, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers
        self.linear1  = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2  = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear3  = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = torch.tanh(x) * 10
        return x

    def arg_forward(self, x, weights):
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        x = torch.tanh(x) * 10
        return x
    

class HumanResponseMAML():
    def __init__(self, net, alpha, beta, K, num_meta_tasks, device=None):
        self.net     = net.to(device)
        self.device  = device
        self.weights = list(self.net.parameters())
        self.alpha   = alpha
        self.beta    = beta
        self.K       = K
        self.TrueHumanResponse = TrueHumanResponse(
            valence_csv_path="valence_merge.csv", arousal_csv_path="arousal_merge.csv",
            num_subjects=12, seed=0
        )

        self.criterion      = nn.MSELoss()
        self.meta_optimizer = optim.Adam(self.weights,self.beta)
        self.meta_losses = []
        self.plot_every  = 10
        self.print_every = 500
        self.num_meta_tasks=num_meta_tasks

    def inner_loop(self, task):
        temp_weights = [w.clone() for w in self.weights]
        x, y = self.TrueHumanResponse.sample_data(task=task, size=self.K) # sampling D
        x, y = x.to(self.device), y.to(self.device)

        output = self.net.arg_forward(x, temp_weights)
        loss   = self.criterion(output,y)/self.K
        grads  = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w-self.alpha*g for w,g in zip(temp_weights,grads)] # temporary update of weights

        x, y = self.TrueHumanResponse.sample_data(task=task, size=self.K) # sampling D'
        x, y = x.to(self.device), y.to(self.device)

        output = self.net.arg_forward(x, temp_weights)
        meta_loss = self.criterion(output, y)/self.K

        return meta_loss
    
    def outer_loop(self, num_epochs):
        total_loss = 0
        for epoch in range(1, num_epochs+1):
            meta_loss_sum=0

            for i in range(self.num_meta_tasks):
                task = self.TrueHumanResponse.sample_task()
                meta_loss = self.inner_loop(task)
                meta_loss_sum += meta_loss

            meta_grads = torch.autograd.grad(meta_loss_sum, self.weights)
            ## important step
            for w,g in zip(self.weights, meta_grads):
                w.grad = g

            # self.meta_optimizer.zero_grad()  # optimizer gradient initialization
            # meta_loss_sum.backward()         # gradient automated computation

            self.meta_optimizer.step()
            
            total_loss += meta_loss_sum.item()/self.num_meta_tasks

            if epoch % self.print_every == 0:
                print("{}/{}. loss: {}".format(epoch, num_epochs, total_loss / self.plot_every))
            if epoch % self.plot_every==0:
                self.meta_losses.append(total_loss/self.plot_every)
                total_loss = 0
            if (epoch%100)==0:
                print(f"Epoch {str(epoch)} completed")


def train_supervised(subject_id=0, sample_size=32, num_epochs=100, batch_size=32, net_supervised=None, pretrained_pth_path=None, device=None):
    True_HumanResponse = TrueHumanResponse(valence_csv_path="valence_merge.csv",
                                        arousal_csv_path="arousal_merge.csv",
                                        num_subjects=18, seed=0)

    task = {"subject_id": subject_id}
    task["val_coeffs"], task["val_mean"], task["val_std"] = True_HumanResponse.load_response(True_HumanResponse.valence_csv_path, subject_id)
    task["aro_coeffs"], task["aro_mean"], task["aro_std"] = True_HumanResponse.load_response(True_HumanResponse.arousal_csv_path, subject_id)

    move_speed_boundary = (27.8, 143.8)
    arm_speed_boundary  = (23.8, 109.1)
    choice_values       = torch.tensor([-1, 1])

    x_tensors_list = []
    y_tensors_list = []
    for _ in range(sample_size):
        move_speed = torch.FloatTensor(1).uniform_(move_speed_boundary[0], move_speed_boundary[1])
        arm_speed  = torch.FloatTensor(1).uniform_(arm_speed_boundary[0], arm_speed_boundary[1])
        random_choices = choice_values[torch.randint(0, 2, (3,))]
        random_tensor = torch.cat((move_speed, arm_speed, random_choices), 0)

        current_task_M = torch.tensor([
            1, random_tensor[0] ** 2, random_tensor[0], random_tensor[1] ** 2, random_tensor[1], random_tensor[2],
            random_tensor[3], random_tensor[4], 1
        ])

        valence = torch.matmul(current_task_M, torch.from_numpy(task["val_coeffs"]).float()).unsqueeze(0)
        arousal = torch.matmul(current_task_M, torch.from_numpy(task["aro_coeffs"]).float()).unsqueeze(0)
        response = torch.cat((valence, arousal), 0)
        
        x_tensors_list.append(random_tensor)
        y_tensors_list.append(response)

    x_stacked_tensor = torch.stack(x_tensors_list).to(device) # shape (size, 5)
    y_stacked_tensor = torch.stack(y_tensors_list).to(device) # shape (size, 2)

    net_supervised = net_supervised.to(device)
    if pretrained_pth_path is not None:
        net_supervised.load_state_dict(torch.load(pretrained_pth_path))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(net_supervised.parameters()), 1e-3)
    
    for epoch in range(num_epochs):
        total_batches = x_stacked_tensor.size(0) // batch_size
        epoch_loss = 0.0
        
        for i in range(total_batches):
            start = i * batch_size
            end   = start + batch_size
            x_batch = x_stacked_tensor[start:end]
            y_batch = y_stacked_tensor[start:end]
            
            net_supervised.train()
            optimizer.zero_grad()
            outputs = net_supervised(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_epoch_loss = epoch_loss / total_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    
    return net_supervised


def scale_action(prob, min_action, max_action):
    """
    From a sampled value (probability within the range of 0 to 1),
    convert the value into a real continuous value (e.g., robot movement speed). 
    """
    scaled_action = min_action + prob*(max_action-min_action)
    return scaled_action

def inverse_scale_action(scaled_action, min_action, max_action):
    """
    Inverse version of the function, scale_action
    """
    prob = (scaled_action - min_action) / (max_action - min_action)
    return prob

def enforce_sample(sample):
    """
    Each mu is in the range of (0, 1).
    Given this mu, when sampling, the sampled value can be larger than 1 and smaller than 0.
    To make the sampled value still be in the range of (0, 1), designed this function. 
    """
    return torch.clamp(sample, 0, 1)

def enforce_prob(prob, prob_epsilon=1e-5):
    return torch.clamp(prob, prob_epsilon, 1-prob_epsilon)

def enforce_alpha_beta(ab, ab_epsilon=1e-5):
    ab = torch.where(torch.isnan(ab), torch.tensor(ab_epsilon, device=ab.device), ab)
    ab = torch.where(torch.isnan(ab), torch.tensor(ab_epsilon, device=ab.device), ab)
    return ab


class ActorCritic(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=5, std=0.1):
        """
        Args:
            state_dim >> [current productivity, current valence, current arousal]
            NOTE: productivity range scale (0, >271) is larger than valence/arousal range scale (-10, 10).
        """
        super(ActorCritic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim) # mu1, mu2, bin1, bin2, bin3
        self.critic = nn.Linear(hidden_dim, 1)
        self.std = std
        self.move_speed_boundary = (27.8, 143.8)
        self.arm_speed_boundary  = (23.8, 109.1)
    
    def forward(self, state):
        """
        Return:
            By using torch.sigmoid, enforce the range of output (i.e., mus and binaries) to be within (0, 1)
            self.critic >> Estimated state value function
        """
        state_embd = self.layers(state)
        return torch.sigmoid(self.actor(state_embd)), self.critic(state_embd)
    
    def sample(self, state):
        probs, _ = self.forward(state)
        mu1, mu2 = probs[0], probs[1]
        bin_probs = probs[2:]
        
        dist1 = Normal(mu1, self.std)
        action1 = dist1.sample()
        action1 = enforce_sample(action1)
        log_prob1 = dist1.log_prob(action1) #TODO: Check the order between enforce_sample and log_prob

        dist2 = Normal(mu2, self.std)
        action2 = dist2.sample()
        action2 = enforce_sample(action2)
        log_prob2 = dist2.log_prob(action2)

        bin_actions = (bin_probs > 0.5).float()
        bin_actions[bin_actions == 0] = -1 # [0, 1] swith to [-1, 1] TODO: Check binary action values.
        return dist1, dist2, action1, action2, bin_actions, log_prob1, log_prob2, bin_probs

    def sample_log_prob(self, state, action):
        """
        From multiple probability distributions,
        we want to get a single probability.
        Since each probablity is independent, we multiplied them.
        When identifying the final single probability, consider the below formula.
            log(p1*p2) = log(p1) + log(p2)
        """
        probs, _ = self.forward(state)
        mu1, mu2 = probs[:, 0], probs[:,1]
        bin_probs = probs[:, 2:]

        dist1 = Normal(mu1, self.std)
        dist2 = Normal(mu2, self.std)

        move_speed = action[:,0]
        prob1 = inverse_scale_action(move_speed, self.move_speed_boundary[0], self.move_speed_boundary[1])
        
        log_prob1 = dist1.log_prob(prob1.unsqueeze(1))
        
        arm_speed  = action[:,1]
        prob2 = inverse_scale_action(arm_speed, self.arm_speed_boundary[0], self.arm_speed_boundary[1])
        log_prob2 = dist2.log_prob(prob2.unsqueeze(1))

        bin_actions = action[:,2:]
        
        log_bin_probs = torch.where(
            (bin_actions == 1) & (bin_probs > 0.5), bin_probs,
            torch.where(
                (bin_actions == 1) & (bin_probs <= 0.5), 1 - bin_probs,
                torch.where(
                    (bin_actions == -1) & (bin_probs > 0.5), 1 - bin_probs,
                    bin_probs
                )
            )
        )
        log_probs = torch.cat((log_prob1, log_prob2, log_bin_probs), dim=1).sum(dim=1, keepdim=True)
        return log_probs


class PPO(nn.Module):
    """
    Note that our PPO considered multiple continous action spaces.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, std=0.1, learning_rate=5e-5,
                 K_epochs=3, gamma=0.98, lmbda=0.95, eps_clip=0.1,
                 device=False):
        super(PPO, self).__init__()
        self.AC = ActorCritic(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, std=std)
        self.device = device
        if device:
            self.AC = self.AC.to(device)
            
        self.K_epochs  = K_epochs
        self.gamma     = gamma
        self.lmbda     = lmbda
        self.eps_clip  = eps_clip
        self.dist_std  = std
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.MSEloss   = nn.MSELoss()
        
        self.data = []
        
    def get_v(self, x):
        _, v = self.AC(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s = torch.stack(s_lst, dim=0)
        a = torch.stack(a_lst, dim=0)
        r = torch.tensor(r_lst)
        s_prime = torch.stack(s_prime_lst, dim=0)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        """
        Generalized advantage estimator
            A_t^{GAE(gamma, lambda)}
                = \sum_{l=0}^{INF} { (gamma*lmbda)^l * delta_(t+l)^V }
            delta >>> GAE(gamma, 0): A_t = r_t + gamma*V(s_(t+1)) - V(s_t)
        
        Probability ratio (called ratio)
            r_t(theta) = policy(a|s) / policy_old(a|s)

        Surrogate objective function
            L_CLIP = E [min(ratio*advantage , CLIP(ratio, 1-eps, 1+eps) ]
            L = L_CLIP - L_VF (+ Entropy_bonus)
        """
        s, a, r, s_prime, done_mask, log_prob_a = self.make_batch()

        for i in range(self.K_epochs):
            td_target = r.to(self.device) + self.gamma * self.get_v(s_prime.to(device)) * done_mask.to(self.device)
            delta = td_target - self.get_v(s.to(self.device))
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float) # torch tensor shape >>> [T_horizon, 1]

            log_probs = self.AC.sample_log_prob(s.to(device), a.to(device))
            ratio = torch.exp(log_probs-log_prob_a.to(device))

            surr1 = ratio * advantage.to(self.device)
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage.to(self.device)
            
            #NOTE: maximize L_CLIP and minimize L_VF
            loss = -torch.min(surr1, surr2) + self.MSEloss(td_target.detach(), self.get_v(s.to(self.device)))
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def get_state(robot_state, hrnet, move_distance=305, arm_distance=120):
    """
    Args:
        move_distance unit : cm
        arm_distance  unit : cm >>> 2*60
    """
    move_speed, arm_swing_speed, proximity, level_of_autonomy, leader_of_collaboration = robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4]

    travel_time   = 2*(arm_distance/arm_swing_speed + move_distance/move_speed)

    if proximity > 0:
        travel_time += 2
    if level_of_autonomy > 0:
        travel_time += 3
    if leader_of_collaboration > 0:
        travel_time += 5

    travel_time /= 60               # unit: min
    productivity = 60 / travel_time # unit: bricks per hour

    valence, arousal = hrnet(robot_state)
    return torch.tensor([productivity, valence, arousal])
            

def get_reward(current_state, penalty_standard=(271, 0, 0), penalty=200):
    """
    Args:
        current_state is [productivity, valence, arousal] 
        penalty_standard is a standard for (productivity, valence, arousal)
    """
    theoretical_max_productivity, standard_valence, standard_arousal = penalty_standard

    distance = abs(current_state[0]-theoretical_max_productivity)
    reward = theoretical_max_productivity - distance

    if current_state[0] > theoretical_max_productivity:
        # reward -= distance
        reward -= penalty
    
    if current_state[1] < standard_valence or current_state[2] < standard_arousal:
        reward -= penalty
    
    #TODO: Add cognitive value associated term.
    return reward


class HRCENV:
    def __init__(self, hr_net=None,
                 move_distance=305, arm_distance=120,
                 penalty_standard=(271, 0, 0), penalty=200):
        self.hr_net = hr_net
        self.move_distance = move_distance
        self.arm_distance  = arm_distance
        self.penalty_standard = penalty_standard
        self.penalty = penalty

        self.move_speed_boundary = (27.8, 143.8)
        self.arm_speed_boundary  = (23.8, 109.1)
        self.choice_values       = torch.tensor([-1, 1])
        
        move_speed = torch.FloatTensor(1).uniform_(self.move_speed_boundary[0], self.move_speed_boundary[1])
        arm_speed  = torch.FloatTensor(1).uniform_(self.arm_speed_boundary[0], self.arm_speed_boundary[1])
        random_choices = self.choice_values[torch.randint(0, 2, (3,))]
        robot_state = torch.cat((move_speed, arm_speed, random_choices), 0) # random robot_state
        self.state = get_state(robot_state.to(device), self.hr_net, move_distance, arm_distance)
        
    def reset(self):
        move_speed = torch.FloatTensor(1).uniform_(self.move_speed_boundary[0], self.move_speed_boundary[1])
        arm_speed  = torch.FloatTensor(1).uniform_(self.arm_speed_boundary[0], self.arm_speed_boundary[1])
        random_choices = self.choice_values[torch.randint(0, 2, (3,))]
        robot_state = torch.cat((move_speed, arm_speed, random_choices), 0) # random robot_state
        self.state = get_state(robot_state.to(device), self.hr_net, self.move_distance, self.arm_distance)
        return self.state

    def step(self, action, terminate_condition=(271, 30, 0 ,0)):
        """
        Args:
            terminate_condition: (target productivity, epsilon for productivity, target_valence, target_arousal)
        """
        done = False
        self.state = get_state(action, self.hr_net, self.move_distance, self.arm_distance)
        reward = get_reward(self.state, self.penalty_standard, self.penalty)
        
        #TODO: Check a terminate_condition
        if self.state[0] <= terminate_condition[0] \
            and terminate_condition[0]-self.state[0] <= terminate_condition[1] \
            and self.state[1] > terminate_condition[2] \
            and self.state[2] > terminate_condition[3]:
            done = True
            
        ## Valence and Arousal only
        # if self.state[1] > terminate_condition[2] and self.state[2] > terminate_condition[3]:
        #     done = True
            
        return self.state, reward, done