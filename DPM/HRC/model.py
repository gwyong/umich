import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

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