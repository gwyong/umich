import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from model import TrueHumanResponse

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def visualize_MAML(net, seed, fixed_values=torch.tensor([-1.0, -1.0, 1.0]),
                   point_gap=20, net_name="Meta", pred_color="orange", true_color="green", device=None):
    """
    Args:
        fixed_value >> Fixed the last three binary robot states
    """
    True_HumanResponse = TrueHumanResponse(valence_csv_path="valence_merge.csv",
                                        arousal_csv_path="arousal_merge.csv",
                                        num_subjects=18, seed=1)
    task = True_HumanResponse.sample_task()

    move_speed_boundary = (27.8, 143.8)
    arm_speed_boundary  = (23.8, 109.1)

    move_speed_range = torch.linspace(move_speed_boundary[0], move_speed_boundary[1], point_gap) 
    arm_speed_range  = torch.linspace(arm_speed_boundary[0], arm_speed_boundary[1], point_gap)

    X, Y = torch.meshgrid(move_speed_range, arm_speed_range, indexing='ij')
    valence = torch.zeros(X.shape)
    arousal = torch.zeros(X.shape)
    true_valence = torch.zeros(X.shape)
    true_arousal = torch.zeros(X.shape)

    for i in range(X.size(0)):
        for j in range(Y.size(1)):
            input_tensor = torch.tensor([X[i, j], Y[i, j], *fixed_values], dtype=torch.float32).to(device)
            output = net(input_tensor)
            valence[i, j] = output[0]
            arousal[i, j] = output[1]

            current_task_M = torch.tensor([
                    1, input_tensor[0] ** 2, input_tensor[0], input_tensor[1] ** 2, input_tensor[1], input_tensor[2],
                    input_tensor[3], input_tensor[4], 1
                ])

            true_valence[i, j] = torch.matmul(current_task_M, torch.from_numpy(task["val_coeffs"]).float())
            true_arousal[i, j] = torch.matmul(current_task_M, torch.from_numpy(task["aro_coeffs"]).float())

    X_np, Y_np, valence_np, arousal_np = X.numpy(), Y.numpy(), valence.detach().numpy(), arousal.detach().numpy()
    true_valence_np, true_arousal_np = true_valence.numpy(), true_arousal.numpy()

    fig = plt.figure(figsize=(14, 6))

    # 3D plot for the first output (Valence)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_np, Y_np, valence_np, color=pred_color)
    ax1.plot_surface(X_np, Y_np, true_valence_np, color=true_color)
    ax1_title = "Subject ID " + str(task["subject_id"]) + "' Valence predicted by " + net_name
    ax1.set_title(ax1_title)
    ax1.set_xlabel('Robot movement speed')
    ax1.set_ylabel('Arm swing speed')
    ax1.set_zlabel('Valence value')

    # 3D plot for the second output (Arousal)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_np, Y_np, arousal_np, color=pred_color)
    ax2.plot_surface(X_np, Y_np, true_arousal_np, color=true_color)
    ax2_title = "Subject ID " + str(task["subject_id"]) + "' Arousal predicted by " + net_name
    ax2.set_title(ax2_title)
    ax2.set_xlabel('Robot movement speed')
    ax2.set_ylabel('Arm swing speed')
    ax2.set_zlabel('Arousal Value')