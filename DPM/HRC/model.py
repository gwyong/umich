import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

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
    Predict human response (valance and arousal) based on the HRC robot state
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
            num_subjects=18, seed=0
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


def train_supervised(subject_id=0, sample_size=32, num_epochs=100, batch_size=32, net_supervised=None, device=None):
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=128, action_dim=72):
        """
        Args:
            state_dim  >> [robot movement speed, arm swing speed, proximity, level of autonomy, leader of collaboration]
            action_dim >> [+0-, +0-, on/off, on/off, on/off]
        """
        super(ActorCritic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        state_embd = self.layers(state)
        return self.actor(state_embd), self.critic(state_embd)
    

class ContinousActorCritic(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=5):
        """
        Args:
            state_dim >> [current productivity, current valence, current arousal]
            NOTE: productivity range scale (0, >271) is larger than valence/arousal range scale (-10, 10).
        """
        super(ContinousActorCritic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mu    = nn.Linear(hidden_dim, action_dim)
        self.actor_sigma = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Return:
            By using torch.tanh, enforce the range of output to be within (-1, 1)
            self.critic >> Estimated state value function
        """
        state_embd = self.layers(state)
        return torch.tanh(self.actor_mu(state_embd)), torch.sigmoid(self.actor_sigma(state_embd)), self.critic(state_embd)


class PPO(nn.Module):
    def __init__(self, AC_state_dim=5, AC_hidden_dim=128, AC_action_dim=72, learning_rate=1e-3,
                 K_epoch=3, gamma=0.98, lmbda=0.95, eps_clip=0.1,
                 device=False):
        super(PPO, self).__init__()
        self.data = []
        
        self.actorCritic = ActorCritic(state_dim=AC_state_dim, hidden_dim=AC_hidden_dim, action_dim=AC_action_dim)
        if device:
            self.actorCritic = self.actorCritic.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.K_epoch = K_epoch
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip

        self.MSEloss = nn.MSELoss()
        self.device = device

    def get_pi(self, x, softmax_dim=0):
        x, _ = self.actorCritic(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def get_v(self, x):
        _, v = self.actorCritic(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s = torch.stack(s_lst, dim=0)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.stack(s_prime_lst, dim=0)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r.to(self.device) + self.gamma * self.get_v(s_prime.to(device)) * done_mask.to(self.device)
            delta = td_target - self.get_v(s.to(self.device))
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi    = self.get_pi(s.to(self.device), softmax_dim=1)
            pi_a  = pi.gather(1,a.to(self.device))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.to(self.device)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage.to(self.device)
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage.to(self.device)
            # loss  = -torch.min(surr1, surr2) + F.smooth_l1_loss(td_target.detach(), self.get_v(s.to(self.device)))
            loss  = -torch.min(surr1, surr2) + self.MSEloss(td_target.detach(), self.get_v(s.to(self.device)))
            
            """
            If we want to use the entropy term to explore more, please use the below codes.
            """
            # dist = Categorical(pi)
            # entropy = dist.entropy()
            # entropy_bonus = -0.2 * entropy.mean()
            # loss -= entropy_bonus
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            """
            We also used a gradient clipping. If we don't want, just ignore it.
            """
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            #----------------------------------------------------------------#

            self.optimizer.step()


class ContinousPPO(nn.Module):
    def __init__(self, AC_state_dim=3, AC_hidden_dim=64, AC_action_dim=5, learning_rate=1e-4,
                 K_epoch=4, gamma=0.98, lmbda=0.95, eps_clip=0.1,
                 device=False):
        super(ContinousPPO, self).__init__()
        self.data = []
        
        self.AC_state_dim = AC_state_dim
        self.CAC = ContinousActorCritic(state_dim=AC_state_dim, hidden_dim=AC_hidden_dim, action_dim=AC_action_dim)
        if device:
            self.CAC = self.CAC.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.K_epoch  = K_epoch
        self.gamma    = gamma
        self.lmbda    = lmbda
        self.eps_clip = eps_clip
        self.MSEloss  = nn.MSELoss()
        
        self.device = device

    def get_pi(self, x):
        """
        Return:
            mu_and_binary_probs[:2] >> Only the first two elements are mus and the others will be used for binary switches.
        """
        
        mu_and_binary_probs, sigmas, _ = self.CAC(x)
        if x.shape[0] != self.AC_state_dim:
            mus = mu_and_binary_probs[:, :2]
            binary_probs = mu_and_binary_probs[:, 2:]
        else:
            mus = mu_and_binary_probs[:2]
            binary_probs = mu_and_binary_probs[2:]
        
        #FIXME: When training, `mus` go to extreme values (e.g., -1 or 1)
        return torch.distributions.Normal(mus, sigmas), binary_probs
    
    def get_v(self, x):
        _, _, v = self.CAC(x)
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
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r.to(self.device) + self.gamma * self.get_v(s_prime.to(device)) * done_mask.to(self.device)
            delta = td_target - self.get_v(s.to(self.device))
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            dist = self.get_pi(s.to(self.device))[0]
            log_probs = dist.log_prob(a[:,:2].to(self.device)).unsqueeze(-1)
            log_prob = log_probs[:,0] + log_probs[:,1] #NOTE: p(a1,a2) = p(a1)*p(a2) >> log(p(a1,a2)) = log(p(a1)) + log(p(a2))
            
            #FIXME: Please check prob_a. Is it probability or log_probability?
            ratio = torch.exp(log_prob.to(self.device) - torch.log(prob_a.to(self.device)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage.to(self.device)
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage.to(self.device)
            loss  = -torch.min(surr1, surr2) + self.MSEloss(td_target.detach(), self.get_v(s.to(self.device)))

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def get_reward(current_state, HRR_model=None):
    """
    Args:
        If current_state.shape[0] is 3, then it is [productivity, valence, arousal]. Otherwise, it is a 5 dim robot state. 
    """
    if current_state.shape[0] == 3:
        productivity, valence, arousal = current_state[0], current_state[1], current_state[2]

        theoretical_max_productivity = 271
        
        distance = abs(current_state[0]-theoretical_max_productivity)
        reward = theoretical_max_productivity - distance

        if current_state[0] > theoretical_max_productivity:
            reward -= distance
        
        if current_state[1] < 0 or current_state[2] < 0:
            reward -= 100

    elif current_state.shape[0] == 5:
        move_speed, arm_swing_speed, proximity, level_of_autonomy, leader_of_collaboration = current_state[0], current_state[1], current_state[2], current_state[3], current_state[4]

        max_productivity = 271
        move_distance = 305  # unit: cm
        arm_distance  = 2*60 # unit: cm
        travel_time   = 2*(arm_distance/arm_swing_speed + move_distance/move_speed)

        if proximity > 0:
            travel_time += 2
        if level_of_autonomy > 0:
            travel_time += 3
        if leader_of_collaboration > 0:
            travel_time += 5
    
        travel_time /= 60               # unit: min
        productivity = 60 / travel_time # unit: bricks per hour

        valence, arousal = HRR_model(current_state)

        reward = -abs(max_productivity-productivity)
        # reward = productivity + (valence + arousal)*50
        # if valence < 0 or arousal < 0 or productivity > 272:
        #     reward -= 10000
        
        move_speed_boundary = (27.8, 143.8)
        arm_speed_boundary  = (23.8, 109.1)
        
        if move_speed < move_speed_boundary[0] or move_speed > move_speed_boundary[1] or arm_swing_speed < arm_speed_boundary[0] or arm_swing_speed > arm_speed_boundary[1]:
            reward -= 10000
    
    return reward, productivity, valence, arousal
            

class HRCENV:
    def __init__(self, continous_action=False, hr_net=None, delta_speed=2.0, productivity_threshold=250.0, init_choice=torch.tensor([1, 1, 1])):

        self.state_dim  = 5   # robot state 5 dim tensor
        self.action_dim = 72  # action (increase/decrease/maintain, increase/decrease/maintain, on/off, on/off, on/off)
        self.hr_net = hr_net
        self.delta_speed= delta_speed
        self.productivity_threshold = productivity_threshold

        self.move_speed_boundary = (27.8, 143.8)
        self.arm_speed_boundary  = (23.8, 109.1)
        self.choice_values       = torch.tensor([-1, 1])
        self.max_sample_aciton   = 2

        self.init_move_speed = (self.move_speed_boundary[0] + self.move_speed_boundary[1])/2
        self.init_arm_speed  = (self.arm_speed_boundary[0] + self.arm_speed_boundary[1])/2
        self.init_choice = init_choice # binary mask
        
        """
        FIXME:
            Should the initial state be fixed or set randomly?
            # move_speed = torch.FloatTensor(1).uniform_(self.move_speed_boundary[0], self.move_speed_boundary[1])
            # arm_speed  = torch.FloatTensor(1).uniform_(self.arm_speed_boundary[0], self.arm_speed_boundary[1])
            # random_choices = self.choice_values[torch.randint(0, 2, (3,))]
            # self.state = torch.cat((move_speed, arm_speed, random_choices), 0) # random state as init state >>> torch.size([5])
        """

        self.state = torch.tensor([self.init_move_speed, self.init_arm_speed, self.init_choice[0], self.init_choice[1], self.init_choice[2]])
        self.action_effects = self._define_action_effects()
        
        self.continous_action = continous_action
        if continous_action:
            self.state_dim  = 3 # [productivity, valence, arousal] state 3 dim tensor
            self.action_dim = 5
            _, productivity, valence, arousal = get_reward(self.state.to(device), self.hr_net)
            self.state = torch.tensor([productivity, valence, arousal])

    def reset(self):
        """
        FIXME:
            Should the initial state be fixed or set randomly?
            # move_speed = torch.FloatTensor(1).uniform_(self.move_speed_boundary[0], self.move_speed_boundary[1])
            # arm_speed  = torch.FloatTensor(1).uniform_(self.arm_speed_boundary[0], self.arm_speed_boundary[1])
            # random_choices = self.choice_values[torch.randint(0, 2, (3,))]
            # self.state = torch.cat((move_speed, arm_speed, random_choices), 0) # random state as init state >>> torch.size([5])
        """
        self.state = torch.tensor([self.init_move_speed, self.init_arm_speed, self.init_choice[0], self.init_choice[0], self.init_choice[0]])
        if self.continous_action:
            _, productivity, valence, arousal = get_reward(self.state.to(device), self.hr_net)
            self.state = torch.tensor([productivity, valence, arousal])
        return self.state

    def step(self, action_index=None, normalized_probs=None, binary_probs=None):
        done = False

        if action_index is not None:
            action = self.action_effects[action_index]
            self.state[0] += action[0]
            self.state[1] += action[1]
            for i in range(2, 5):
                self.state[i] = action[i] 
            reward, productivity, valence, arousal = get_reward(self.state.to(device), self.hr_net)

            #FIXME: Condition to terminate the episode. How can we terminate the episode?
            # done = torch.rand(1).item() < 0.1  # e.g., randomly terminate with 10% prob
            
            if productivity > self.productivity_threshold and valence > 0 and arousal > 0:
                done = True
        
        else: # When we use normalized_probs and binary_probs!
            move_speed   = (normalized_probs[0] - (-self.max_sample_aciton))*(self.move_speed_boundary[1]-self.move_speed_boundary[0])/(self.max_sample_aciton*2)
            arm_speed    = (normalized_probs[1] - (-self.max_sample_aciton))*(self.arm_speed_boundary[1]-self.arm_speed_boundary[0])/(self.max_sample_aciton*2)
            binary_probs = torch.where(binary_probs < 0, torch.tensor(-1.0).to(device), torch.tensor(1.0).to(device))
            robot_state  = torch.cat((move_speed.repeat(1).to(device), arm_speed.repeat(1).to(device), binary_probs.to(device)), dim=0)
            action = robot_state

            _, productivity, valence, arousal = get_reward(robot_state, self.hr_net)
            self.state = torch.cat((productivity.repeat(1), valence.repeat(1), arousal.repeat(1)), dim=0)
            reward, _, _, _ = get_reward (self.state, self.hr_net)
            
            if productivity > self.productivity_threshold and valence > 0 and arousal > 0:
                done = True
                
        return self.state, reward, done, action

    def _define_action_effects(self):
        """
        If discrete, return 72 action combinations.
        """
        actions = []
        for speed_change in [-self.delta_speed, 0, self.delta_speed]:
            for arm_speed_change in [-self.delta_speed, 0, self.delta_speed]:
                for state_3 in [self.choice_values[0], self.choice_values[1]]:
                    for state_4 in [-self.choice_values[0], self.choice_values[1]]:
                        for state_5 in [-self.choice_values[0], self.choice_values[1]]:
                            action = torch.tensor([speed_change, arm_speed_change, state_3, state_4, state_5], dtype=torch.float)
                            actions.append(action)
        return actions

    def render(self):
        pass