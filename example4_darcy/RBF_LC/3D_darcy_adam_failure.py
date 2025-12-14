import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from rbf_layer import RBFLayer, l_norm, rbf_gaussian
import math
import time
'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)


# 定义网络模型
def get_omega_points(num, x_left=0.0, y_left=0.0, x_right=1.0, y_right=1.0, z_left = 0.0, z_right = 1.0):
    
    # 生成 num 个 x 坐标，范围在 [x_left, x_right]
    x_coords = torch.rand(num, 1) * (x_right-x_left) + x_left
    # 生成 num 个 y 坐标，范围在 [y_left, y_right]
    y_coords = torch.rand(num, 1) * (y_right-y_left) + y_left
    # 拼接 x 和 y 坐标形状为 (num，形成, 2) 的张量
    z_coords = torch.rand(num, 1) * (z_right-z_left) + z_left
    points = torch.cat([x_coords, y_coords, z_coords], dim=1)
    return points

def get_sphere_surface_points(
    num: int, 
    center: torch.Tensor, 
    radius: float = 0.2
) -> torch.Tensor:
    
    # 2. 生成均匀分布的角度
    # 为了在球面上均匀分布，极角 theta 需要特殊处理
    # z = cos(theta)，我们从均匀分布中采样 z
    z = torch.linspace(1.0 - 1e-6, -1.0 + 1e-6, num, dtype=torch.float32, device=center.device)
    theta = torch.acos(z)  # 极角，范围 [0, π]

    # 方位角 phi 可以从均匀分布中采样
    phi = 2 * torch.pi * torch.rand(num, dtype=torch.float32, device=center.device) # 范围 [0, 2π)

    # 3. 利用球面坐标到笛卡尔坐标的转换公式计算点
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    x = radius * sin_theta * cos_phi
    y = radius * sin_theta * sin_phi
    z_coord = radius * cos_theta

    # 4. 将坐标与球心相加，并拼接成 (num, 3) 的输出张量
    sphere_points = torch.stack([x, y, z_coord], dim=1) + center

    return sphere_points
    

def get_cube_boundary_points(num: int) -> torch.Tensor:
    """
    生成三维 [0, 1]^3 立方体的边界点。
    
    参数:
        num (int): 生成的边界点总数。必须是6的倍数，以保证每个面的点数相等。
    
    返回:
        torch.Tensor: 边界点张量，形状为 (num, 3)，每一行对应一个 (x, y, z) 坐标。
    
    异常:
        ValueError: 若 num 不是6的倍数，则抛出异常。
    """
    # 1. 输入合法性校验
    if num % 6 != 0:
        raise ValueError(f"点数量 'num' 必须是6的倍数，以确保每个面有相同数量的点，当前为 {num}")

    # 计算每个面的点数量
    points_per_face = num // 6

    # 使用 torch.rand 生成 [0, 1) 范围内的随机坐标
    rand_coords_1 = torch.rand(points_per_face)
    rand_coords_2 = torch.rand(points_per_face)

    # --- 生成6个面的边界点 ---
    
    # 1. x = 0 面
    face_x0 = torch.stack([torch.zeros_like(rand_coords_1), rand_coords_1, rand_coords_2], dim=1)
    
    # 2. x = 1 面
    face_x1 = torch.stack([torch.ones_like(rand_coords_1), rand_coords_1, rand_coords_2], dim=1)
    
    # 3. y = 0 面
    face_y0 = torch.stack([rand_coords_1, torch.zeros_like(rand_coords_1), rand_coords_2], dim=1)
    
    # 4. y = 1 面
    face_y1 = torch.stack([rand_coords_1, torch.ones_like(rand_coords_1), rand_coords_2], dim=1)
    
    # 5. z = 0 面
    face_z0 = torch.stack([rand_coords_1, rand_coords_2, torch.zeros_like(rand_coords_1)], dim=1)
    
    # 6. z = 1 面
    face_z1 = torch.stack([rand_coords_1, rand_coords_2, torch.ones_like(rand_coords_1)], dim=1)

    # 拼接所有边界点
    all_points = torch.cat([face_x0, face_x1, face_y0, face_y1, face_z0, face_z1], dim=0)

    return all_points

def alpha_x(x_op, radius: float = 0.2) -> torch.Tensor:
    
    # 确保输入是一维的
    x = x_op[:,0].flatten()
    y = x_op[:,1].flatten()
    z = x_op[:,2].flatten()
    # 四个角落的中心坐标（转为 NumPy 数组用于计算）
    centers = np.array([
        [0.2, 0.2, 0.2],  
        [0.2, 0.2, 0.8],  
        [0.2, 0.8, 0.2],  
        [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.2],  
        [0.8, 0.2, 0.8],  
        [0.8, 0.8, 0.2],  
        [0.8, 0.8, 0.8]  
    ])
    # 将输入的 x, y 转为 NumPy 数组
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    # 初始化 alpha 值为 0.018
    alpha_np = np.ones_like(x_np) * 0.018
    # 遍历每个角落中心，判断点是否在角落区域内
    for center in centers:
        # 计算每个点到当前角落中心的距离
        dist = np.sqrt((x_np-center[0])**2 + (y_np-center[1])** 2 + (z_np-center[2])**2)
        # 若距离小于半径，将对应位置的 alpha 值设为 137
        alpha_np[dist < radius] = 137.0
    # 将 NumPy 数组转回 PyTorch Tensor，并保持设备和数据类型与输入一致
    alpha_tensor = torch.tensor(alpha_np, dtype=x.dtype, device=x.device)
    return alpha_tensor

def f_x(x):
    f = torch.ones_like(x[:,0]) * 10
    return f

def u_x_boundary(x):
    u_b = torch.zeros_like(x[:,0])
    return u_b

def func_loss_o_rbf1(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_1(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf2(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_2(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf3(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_3(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf4(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_4(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf5(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_5(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf6(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_6(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf7(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_7(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_rbf8(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = rbf_model_8(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dy),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_nn(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = model_nn(x_op)
    f_omega = f_omega.reshape(-1, 1)
    a = alpha_x(x_op.cpu()).to(device).reshape(-1, 1)
    grad_op = autograd.grad(outputs=output_op, inputs=x_op,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    du_dz = torch.unsqueeze(grad_op[:, 2],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op3 = autograd.grad(outputs=du_dz, inputs=x_op,
                            grad_outputs=torch.ones_like(du_dz),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    u_zz = torch.unsqueeze(grad_op3[:, 2],1)
    loss_op = a * (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_b_rbf1(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_1(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf2(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_2(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf3(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_3(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf4(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_4(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf5(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_5(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf6(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_6(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf7(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_7(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

def func_loss_b_rbf8(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = rbf_model_8(x_b)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

#计算边界上nn的loss
def func_loss_b_nn(x_b: torch.Tensor) -> torch.Tensor:
    g_omega = u_x_boundary(x_b).reshape(-1, 1)
    u = model_nn(x_b).reshape(-1, 1)
    loss_b = u - g_omega
    return torch.mean(loss_b**2)

'''-------------------------Define networks-------------------------'''
class NeuralNet_Shallow(torch.nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output

    def __init__(self, in_dim, h_dim, out_dim):
        super(NeuralNet_Shallow, self).__init__()
        self.ln1 = nn.Linear(in_dim, h_dim)
        self.act1 = nn.Sigmoid()
        # self.act1 = nn.Tanh()
        # self.act1 = nn.ReLU()

        self.ln2 = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        return out
        
class NeuralNet_Deep(torch.nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output
    ### depth: depth of the network
    def __init__(self, in_dim, h_dim, out_dim, depth):
        super(NeuralNet_Deep, self).__init__()
        self.depth = depth - 1
        self.list = nn.ModuleList()
        self.ln1 = nn.Linear(in_dim, h_dim)
        #self.act1 = nn.Sigmoid()
        self.act1 = nn.Tanh()
        # self.act1 = nn.ReLU()

        for i in range(self.depth):
            self.list.append(nn.Linear(h_dim, h_dim))

        self.lnd = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        for i in range(self.depth):
            out = self.list[i](out)
            out = self.act1(out)
        out = self.lnd(out)
        return out

# Network size of NN with RBF
n_input_rbf = 3
n_hidden_rbf = 80
n_output_rbf = 1
radius = 0.2
# number of training and test data points

def initialize_spherical_centers(num_centers: int, radius: float, center: torch.Tensor) -> torch.Tensor:
    """
    使用黄金螺旋算法，在指定球面上生成均匀分布的RBF中心点。

    参数:
        num_centers (int): 要生成的中心点数量。
        radius (float): RBF中心点所在球面的半径。
        center (torch.Tensor): RBF中心点所在球面的球心，形状为 (3,)。

    返回:
        torch.Tensor: 形状为 (num_centers, 3) 的张量，可直接用于初始化 nn.Parameter。
    """
    # 确保中心点是3D坐标
    if center.ndim != 1 or center.shape[0] != 3:
        raise ValueError(f"球心 'center' 必须是形状为 (3,) 的 torch.Tensor，当前形状为 {center.shape}")

    points = []
    # 黄金角，确保点的分布没有周期性
    phi = math.pi * (3.0 - math.sqrt(5.0)) 

    for i in range(num_centers):
        # 1. 在 [-1, 1] 范围内均匀采样 z 坐标
        y = 1.0 - (i / (num_centers - 1)) * 2.0  
        
        # 2. 计算在当前 z 高度上的圆的半径
        radius_at_y = math.sqrt(1.0 - y * y)
        
        # 3. 计算方位角，使用黄金角确保均匀分布
        theta = i * phi
        
        # 4. 将 (theta, y) 转换为笛卡尔坐标 (x, y, z)
        x = math.cos(theta) * radius_at_y
        z_coord = math.sin(theta) * radius_at_y
        
        points.append([x, y, z_coord])

    # 将列表转换为张量
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # 5. 将单位球面上的点缩放并平移到目标球面
    points_tensor = points_tensor * radius + center

    return points_tensor


# 定义网格密度
n = 40
x1 = torch.linspace(0, 1, n)
x2 = torch.linspace(0, 1, n)
x3 = torch.linspace(0, 1, n)

# 生成三维网格坐标
# 使用 indexing='ij' 明确指定索引方式，这是 PyTorch 的默认方式
X, Y, Z = torch.meshgrid(x1, x2, x3, indexing='ij')

# --- 正确的做法：直接展平，不要使用 .T ---
# X.flatten(), Y.flatten(), Z.flatten() 会保持相同的顺序，从而保证 (x,y,z) 坐标一一对应
test_points = torch.cat(
    (
        X.flatten()[:, None], 
        Y.flatten()[:, None], 
        Z.flatten()[:, None]
    ), 
    dim=1
).to(device)

test_points.requires_grad = True

# number of training and test data points
N_size = 10
eps_thresh = 0.05
alpha = 20
beta = 10
# lower left point and upper right point of the computational domain
box_lower = np.array([0, 0, 0])
box_upper = np.array([1.0,1.0, 1.0])
# Network size
n_input = 3
n_hidden = 20
n_output = 1
n_depth = 7  # only used in deep NN
num_omega = 10000
num_b = 600 #注意此处设置的为每个边取点数量
num_circle_b = 100
rate = 1e-3
epochs = 20000

# NN structure
if n_depth == 1:  # Shallow NN
	model_nn = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
else:  # Deep NN
	model_nn = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)
model_nn.load_state_dict(torch.load('3d_best_model_first_LM.mdl'))
model_nn.eval() 
# 将两个模型的参数转换为列表并拼接

time_start = time.time()
def getlocal4rbf(epsilon, N_size, box_lower, box_upper, eps_thresh):   
    # 确保 test_points 等在函数作用域内或作为参数传入
    # global test_points, f_x, func_loss_o_nn, device 

    fp = f_x(test_points)
    fp = fp.to(device)
    
    # 计算每个测试点的损失（残差）
    res = torch.abs(func_loss_o_nn(test_points, fp)).cpu().detach().numpy()
    res = res.reshape(-1)  

    # 计算网格步长
    h_x = (box_upper[0] -  box_lower[0]) / N_size
    h_y = (box_upper[1] -  box_lower[1]) / N_size
    h_z = (box_upper[2] -  box_lower[2]) / N_size

    count = 0
    # 创建三维索引数组，用于标记需要加密的网格
    ele_index = np.zeros([N_size, N_size, N_size], dtype='int')
    # 遍历三维网格的每个小立方体 (i, j, k)
    for i in range(N_size):
      for j in range(N_size):
          for k in range(N_size): # k 循环

            # 计算当前小立方体的边界
            x_left = box_lower[0] + j * h_x
            y_left = box_lower[1] + i * h_y
            z_left = box_lower[2] + k * h_z

            # 找到落在当前小立方体内的测试点
            index_local = np.arange(len(test_points))[
                (test_points[:,0].cpu() >= x_left) & (test_points[:,0].cpu() <= (x_left + h_x)) &
                (test_points[:,1].cpu() >= y_left) & (test_points[:,1].cpu() <= (y_left + h_y)) &
                (test_points[:,2].cpu() >= z_left) & (test_points[:,2].cpu() <= (z_left + h_z))
            ]
            
            res_local = res[index_local]

            # --- 关键修正：将判断和赋值放入 k 循环内 ---
            if res_local.size == 0:
                # 如果小立方体内没有任何测试点，根据策略决定是否标记
                ratio = 0  # 或者其他值，如 0.5
            else:
                # 计算误差超过阈值的点所占的比例
                ratio = res_local[res_local > epsilon].size / res_local.size
            
            # 如果误差大的点比例超过设定阈值，则标记该网格需要加密
            if ratio > eps_thresh:
               ele_index[i, j, k] = 1
               count += 1
             
    return ele_index, h_x, h_y, h_z, count

epsilon = 10.3
ele_index, h_x, h_y, h_z, count_num_le = getlocal4rbf(epsilon, N_size, box_lower, box_upper, eps_thresh)

def maxVolumeOfIsland3D(grid):
    if grid.ndim != 3:
        raise ValueError("输入的 grid 必须是三维数组")

    m, n, p = grid.shape  # m(x), n(y), p(z)
    list_of_bboxes = []

    def dfs(i, j, k, xmin, xmax, ymin, ymax, zmin, zmax):
        if 0 <= i < m and 0 <= j < n and 0 <= k < p and grid[i, j, k] == 1:
            grid[i, j, k] = 0  # 标记已访问

            curr_xmin = min(xmin, i)
            curr_xmax = max(xmax, i)
            curr_ymin = min(ymin, j)
            curr_ymax = max(ymax, j)
            curr_zmin = min(zmin, k)
            curr_zmax = max(zmax, k)  

            # 2. 递归探索6个方向（传入当前单元格的边界框）
            vol1, x1, X1, y1, Y1, z1, Z1 = dfs(i-1, j, k, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)
            vol2, x2, X2, y2, Y2, z2, Z2 = dfs(i+1, j, k, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)
            vol3, x3, X3, y3, Y3, z3, Z3 = dfs(i, j-1, k, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)
            vol4, x4, X4, y4, Y4, z4, Z4 = dfs(i, j+1, k, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)
            vol5, x5, X5, y5, Y5, z5, Z5 = dfs(i, j, k-1, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)
            vol6, x6, X6, y6, Y6, z6, Z6 = dfs(i, j, k+1, curr_xmin, curr_xmax, curr_ymin, curr_ymax, curr_zmin, curr_zmax)

            total_volume = 1 + vol1 + vol2 + vol3 + vol4 + vol5 + vol6
            final_xmin = min(curr_xmin, x1, x2, x3, x4, x5, x6)  # 加入 curr_xmin
            final_xmax = max(curr_xmax, X1, X2, X3, X4, X5, X6)  # 加入 curr_xmax
            final_ymin = min(curr_ymin, y1, y2, y3, y4, y5, y6)  # 加入 curr_ymin
            final_ymax = max(curr_ymax, Y1, Y2, Y3, Y4, Y5, Y6)  # 加入 curr_ymax
            final_zmin = min(curr_zmin, z1, z2, z3, z4, z5, z6)  # 加入 curr_zmin
            final_zmax = max(curr_zmax, Z1, Z2, Z3, Z4, Z5, Z6)  # 加入 curr_zmax

            return (total_volume, final_xmin, final_xmax, final_ymin, final_ymax, final_zmin, final_zmax)
        
        # 递归基：非陆地/越界，返回无效边界
        return (0, float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf'))

    # 主循环：遍历所有单元格
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if grid[i, j, k] == 1:
                    volume, xmin, xmax, ymin, ymax, zmin, zmax = dfs(i, j, k, i, i, j, j, k, k)
                    if volume > 1:
                        list_of_bboxes.append( (xmin, xmax, ymin, ymax, zmin, zmax) )
    
    return list_of_bboxes
matrix_tmp = ele_index.copy()
list_matrix = maxVolumeOfIsland3D(matrix_tmp)
print(list_matrix)
x1_left = box_lower[0] + (list_matrix[0][2])*h_x
y1_left = box_lower[1] + (list_matrix[0][0])*h_y
x1_right = box_lower[0] +(list_matrix[0][3]+1)*h_x
y1_right = box_lower[1] +(list_matrix[0][1]+1)*h_y
z1_left = box_lower[2] + (list_matrix[0][4])*(h_z)
z1_right = box_lower[2] +(list_matrix[0][5]+1)*(h_z)
x1_center = 0.5*(x1_left + x1_right)
y1_center = 0.5*(y1_left + y1_right)
z1_center = 0.5*(z1_left + z1_right)
radius1 = max((x1_right-x1_left)/2, (y1_right-y1_left)/2, (z1_right-z1_left)/2)

x2_left = box_lower[0] + (list_matrix[1][2])*h_x
y2_left = box_lower[1] + (list_matrix[1][0])*h_y
x2_right = box_lower[0] +(list_matrix[1][3]+1)*h_x
y2_right = box_lower[1] +(list_matrix[1][1]+1)*h_y
z2_left = box_lower[2] + (list_matrix[1][4])*(h_z)
z2_right = box_lower[2] +(list_matrix[1][5]+1)*(h_z)
x2_center = 0.5*(x2_left + x2_right)
y2_center = 0.5*(y2_left + y2_right)
z2_center = 0.5*(z2_left + z2_right)
radius2 = max((x2_right-x2_left)/2, (y2_right-y2_left)/2, (z2_right-z2_left)/2)

x3_left = box_lower[0] + (list_matrix[2][2])*h_x
y3_left = box_lower[1] + (list_matrix[2][0])*h_y
x3_right = box_lower[0] +(list_matrix[2][3]+1)*h_x
y3_right = box_lower[1] +(list_matrix[2][1]+1)*h_y
z3_left = box_lower[2] + (list_matrix[2][4])*(h_z)
z3_right = box_lower[2] +(list_matrix[2][5]+1)*(h_z)
x3_center = 0.5*(x3_left + x3_right)
y3_center = 0.5*(y3_left + y3_right)
z3_center = 0.5*(z3_left + z3_right)
radius3 = max((x3_right-x3_left)/2, (y3_right-y3_left)/2, (z3_right-z3_left)/2)

x4_left = box_lower[0] + (list_matrix[3][2])*h_x
y4_left = box_lower[1] + (list_matrix[3][0])*h_y
x4_right = box_lower[0] +(list_matrix[3][3]+1)*h_x
y4_right = box_lower[1] +(list_matrix[3][1]+1)*h_y
z4_left = box_lower[2] + (list_matrix[3][4])*(h_z)
z4_right = box_lower[2] +(list_matrix[3][5]+1)*(h_z)
x4_center = 0.5*(x4_left + x4_right)
y4_center = 0.5*(y4_left + y4_right)
z4_center = 0.5*(z4_left + z4_right)
radius4 = max((x4_right-x4_left)/2, (y4_right-y4_left)/2, (z4_right-z4_left)/2)

x5_left = box_lower[0] + (list_matrix[4][2])*h_x
y5_left = box_lower[1] + (list_matrix[4][0])*h_y
x5_right = box_lower[0] +(list_matrix[4][3]+1)*h_x
y5_right = box_lower[1] +(list_matrix[4][1]+1)*h_y
z5_left = box_lower[2] + (list_matrix[4][4])*(h_z)
z5_right = box_lower[2] +(list_matrix[4][5]+1)*(h_z)
x5_center = 0.5*(x5_left + x5_right)
y5_center = 0.5*(y5_left + y5_right)
z5_center = 0.5*(z5_left + z5_right)
radius5 = max((x5_right-x5_left)/2, (y5_right-y5_left)/2, (z5_right-z5_left)/2)

x6_left = box_lower[0] + (list_matrix[5][2])*h_x
y6_left = box_lower[1] + (list_matrix[5][0])*h_y
x6_right = box_lower[0] +(list_matrix[5][3]+1)*h_x
y6_right = box_lower[1] +(list_matrix[5][1]+1)*h_y
z6_left = box_lower[2] + (list_matrix[5][4])*(h_z)
z6_right = box_lower[2] +(list_matrix[5][5]+1)*(h_z)
x6_center = 0.5*(x6_left + x6_right)
y6_center = 0.5*(y6_left + y6_right)
z6_center = 0.5*(z6_left + z6_right)
radius6 = max((x6_right-x6_left)/2, (y6_right-y6_left)/2, (z6_right-z6_left)/2)

x7_left = box_lower[0] + (list_matrix[6][2])*h_x
y7_left = box_lower[1] + (list_matrix[6][0])*h_y
x7_right = box_lower[0] +(list_matrix[6][3]+1)*h_x
y7_right = box_lower[1] +(list_matrix[6][1]+1)*h_y
z7_left = box_lower[2] + (list_matrix[6][4])*(h_z)
z7_right = box_lower[2] +(list_matrix[6][5]+1)*(h_z)
x7_center = 0.5*(x7_left + x7_right)
y7_center = 0.5*(y7_left + y7_right)
z7_center = 0.5*(z7_left + z7_right)
radius7 = max((x7_right-x7_left)/2, (y7_right-y7_left)/2, (z7_right-z7_left)/2)

x8_left = box_lower[0] + (list_matrix[7][2])*h_x
y8_left = box_lower[1] + (list_matrix[7][0])*h_y
x8_right = box_lower[0] +(list_matrix[7][3]+1)*h_x
y8_right = box_lower[1] +(list_matrix[7][1]+1)*h_y
z8_left = box_lower[2] + (list_matrix[7][4])*(h_z)
z8_right = box_lower[2] +(list_matrix[7][5]+1)*(h_z)
x8_center = 0.5*(x8_left + x8_right)
y8_center = 0.5*(y8_left + y8_right)
z8_center = 0.5*(z8_left + z8_right)
radius8 = max((x8_right-x8_left)/2, (y8_right-y8_left)/2, (z8_right-z8_left)/2)

print(x1_center, y1_center, z1_center, radius1, x2_center, y2_center, z2_center, radius2, x3_center, y3_center, z3_center, radius3, x4_center, y4_center, z4_center, radius4, x5_center, y5_center, z5_center, radius5, x6_center, y6_center, z6_center, radius6, x7_center, y7_center, z7_center, radius7, x8_center, y8_center, z8_center, radius8)
centers = torch.tensor([
        [x1_center, y1_center, z1_center],  
        [x2_center, y2_center, z2_center],  
        [x3_center, y3_center, z3_center],  
        [x4_center, y4_center, z4_center],
        [x5_center, y5_center, z5_center],  
        [x6_center, y6_center, z6_center],  
        [x7_center, y7_center, z7_center],  
        [x8_center, y8_center, z8_center]  
    ])

center_init_1 =  initialize_spherical_centers(n_hidden_rbf, radius1, centers[0]) 
center_init_2 =  initialize_spherical_centers(n_hidden_rbf, radius2, centers[1]) 
center_init_3 =  initialize_spherical_centers(n_hidden_rbf, radius3, centers[2]) 
center_init_4 =  initialize_spherical_centers(n_hidden_rbf, radius4, centers[3])
center_init_5 =  initialize_spherical_centers(n_hidden_rbf, radius5, centers[4]) 
center_init_6 =  initialize_spherical_centers(n_hidden_rbf, radius6, centers[5]) 
center_init_7 =  initialize_spherical_centers(n_hidden_rbf, radius7, centers[6]) 
center_init_8 =  initialize_spherical_centers(n_hidden_rbf, radius8, centers[7])

rbf_model_1 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_1,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_2 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_2,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_3 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_3,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_4 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_4,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_5 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_5,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_6 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_6,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_7 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_7,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)

rbf_model_8 = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init_8,\
#                    initial_shape_parameter = initial_shape,\
#                    initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)
# NN structure
if n_depth == 1:  # Shallow NN
	model_nn = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
else:  # Deep NN
	model_nn = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)
# torch.load('best_3D_NN_model.mdl', model_nn.state_dict())
total_params_quick = sum(p.numel() for p in model_nn.parameters())
print(f"模型参数总数: {total_params_quick}")
model_nn.eval() 
# 将两个模型的参数转换为列表并拼接
combined_parameters = list(model_nn.parameters())+list(rbf_model_1.parameters()) + list(rbf_model_2.parameters()) + list(rbf_model_3.parameters()) + list(rbf_model_4.parameters()) + list(rbf_model_5.parameters()) + list(rbf_model_6.parameters()) + list(rbf_model_7.parameters()) + list(rbf_model_8.parameters())
optimizer = torch.optim.Adam(combined_parameters, lr = rate) 
best_loss, best_rel_l2, best_h2, best_L_infinity, best_epoch = 10000, 10000, 10000, 1000,0


for epoch in range(1, 1 + epochs):
    if epoch % 1000 == 0:
        rate = rate/2               
    xo = get_omega_points(num=num_omega).double().requires_grad_(True).to(device)
    centers = centers.to(device)
    mask_in_circle1 = torch.norm(xo - centers[0], dim=1) < radius
    mask_in_circle2 = torch.norm(xo - centers[1], dim=1) < radius
    mask_in_circle3 = torch.norm(xo - centers[2], dim=1) < radius
    mask_in_circle4 = torch.norm(xo - centers[3], dim=1) < radius
    mask_in_circle5 = torch.norm(xo - centers[4], dim=1) < radius
    mask_in_circle6 = torch.norm(xo - centers[5], dim=1) < radius
    mask_in_circle7 = torch.norm(xo - centers[6], dim=1) < radius
    mask_in_circle8 = torch.norm(xo - centers[7], dim=1) < radius
    xo_rbf1 =  xo[mask_in_circle1]
    xo_rbf2 =  xo[mask_in_circle2]
    xo_rbf3 =  xo[mask_in_circle3]
    xo_rbf4 =  xo[mask_in_circle4]
    xo_rbf5 =  xo[mask_in_circle5]
    xo_rbf6 =  xo[mask_in_circle6]
    xo_rbf7 =  xo[mask_in_circle7]
    xo_rbf8 =  xo[mask_in_circle8]
        
    mask_in_any_circle = mask_in_circle1 | mask_in_circle2 | mask_in_circle3 | mask_in_circle4 |mask_in_circle5 | mask_in_circle6 | mask_in_circle7 | mask_in_circle8
    # --- 3. 对联合掩码取反：不在任何一个圆内的点 ---
    mask_outside_all_circles = torch.logical_not(mask_in_any_circle)
    # --- 4. 使用最终掩码筛选点 ---
    xo_outside_all_circles = xo[mask_outside_all_circles]
    xb = get_cube_boundary_points(num=num_b).double().requires_grad_(True).to(device)
    xb_rbf1 = get_sphere_surface_points(num_circle_b, centers[0].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf2 = get_sphere_surface_points(num_circle_b, centers[1].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf3 = get_sphere_surface_points(num_circle_b, centers[2].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf4 = get_sphere_surface_points(num_circle_b, centers[3].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf5 = get_sphere_surface_points(num_circle_b, centers[4].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf6 = get_sphere_surface_points(num_circle_b, centers[5].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf7 = get_sphere_surface_points(num_circle_b, centers[6].cpu(), radius).double().requires_grad_(True).to(device)
    xb_rbf8 = get_sphere_surface_points(num_circle_b, centers[7].cpu(), radius).double().requires_grad_(True).to(device)
    
    f_xo_nn = f_x(xo_outside_all_circles).to(device)
    f_xo_rbf1 = f_x(xo_rbf1).to(device)
    f_xo_rbf2 = f_x(xo_rbf2).to(device)
    f_xo_rbf3 = f_x(xo_rbf3).to(device)
    f_xo_rbf4 = f_x(xo_rbf4).to(device)
    f_xo_rbf5 = f_x(xo_rbf5).to(device)
    f_xo_rbf6 = f_x(xo_rbf6).to(device)
    f_xo_rbf7 = f_x(xo_rbf7).to(device)
    f_xo_rbf8 = f_x(xo_rbf8).to(device)
    loss_xo_nn = torch.mean(torch.pow(func_loss_o_nn(xo_outside_all_circles, f_xo_nn), 2))
    loss_xo_rbf1 = torch.mean(torch.pow(func_loss_o_rbf1(xo_rbf1, f_xo_rbf1), 2))
    loss_xo_rbf2 = torch.mean(torch.pow(func_loss_o_rbf2(xo_rbf2, f_xo_rbf2), 2))
    loss_xo_rbf3 = torch.mean(torch.pow(func_loss_o_rbf3(xo_rbf3, f_xo_rbf3), 2))
    loss_xo_rbf4 = torch.mean(torch.pow(func_loss_o_rbf4(xo_rbf4, f_xo_rbf4), 2))
    loss_xo_rbf5 = torch.mean(torch.pow(func_loss_o_rbf5(xo_rbf5, f_xo_rbf5), 2))
    loss_xo_rbf6 = torch.mean(torch.pow(func_loss_o_rbf6(xo_rbf6, f_xo_rbf6), 2))
    loss_xo_rbf7 = torch.mean(torch.pow(func_loss_o_rbf7(xo_rbf7, f_xo_rbf7), 2))
    loss_xo_rbf8 = torch.mean(torch.pow(func_loss_o_rbf8(xo_rbf8, f_xo_rbf8), 2))
    loss_xb_rbf1 = func_loss_b_rbf1(xb_rbf1)
    loss_xb_rbf2 = func_loss_b_rbf2(xb_rbf2)
    loss_xb_rbf3 = func_loss_b_rbf3(xb_rbf3)
    loss_xb_rbf4 = func_loss_b_rbf4(xb_rbf4)
    loss_xb_rbf5 = func_loss_b_rbf5(xb_rbf5)
    loss_xb_rbf6 = func_loss_b_rbf6(xb_rbf6)
    loss_xb_rbf7 = func_loss_b_rbf7(xb_rbf7)
    loss_xb_rbf8 = func_loss_b_rbf8(xb_rbf8)
    loss_xb_nn = func_loss_b_nn(xb)
    loss = alpha*loss_xo_nn + alpha*loss_xb_nn + beta*loss_xo_rbf1 + loss_xb_rbf1 + beta*loss_xo_rbf2 + loss_xb_rbf2 + beta*loss_xo_rbf3 + loss_xb_rbf3 + beta*loss_xo_rbf4 + loss_xb_rbf4 + beta*loss_xo_rbf5 + loss_xb_rbf5 + beta*loss_xo_rbf6 + loss_xb_rbf6 + beta*loss_xo_rbf7 + loss_xb_rbf7 + beta*loss_xo_rbf8 + loss_xb_rbf8
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch:{} , loss: {} best_epoch: {}, best_loss:{}'.format(epoch, loss.item(), best_epoch, best_loss))
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        """
        torch.save(model_nn.state_dict(), 'best_3D_NN_model.mdl')
        torch.save(rbf_model_1.state_dict(), 'best_3D_RBF_model1.mdl')
        torch.save(rbf_model_2.state_dict(), 'best_3D_RBF_model2.mdl')
        torch.save(rbf_model_3.state_dict(), 'best_3D_RBF_model3.mdl')
        torch.save(rbf_model_4.state_dict(), 'best_3D_RBF_model4.mdl')
        torch.save(rbf_model_5.state_dict(), 'best_3D_RBF_model5.mdl')
        torch.save(rbf_model_6.state_dict(), 'best_3D_RBF_model6.mdl')
        torch.save(rbf_model_7.state_dict(), 'best_3D_RBF_model7.mdl')
        torch.save(rbf_model_8.state_dict(), 'best_3D_RBF_model8.mdl')
        # --- 1. 修正：正确定义所有掩码 ---
        mask_in_circle1 = torch.norm(test_points - centers[0], dim=1) < radius
        mask_in_circle2 = torch.norm(test_points - centers[1], dim=1) < radius
        mask_in_circle3 = torch.norm(test_points - centers[2], dim=1) < radius
        mask_in_circle4 = torch.norm(test_points - centers[3], dim=1) < radius
        mask_in_circle5 = torch.norm(test_points - centers[4], dim=1) < radius
        mask_in_circle6 = torch.norm(test_points - centers[5], dim=1) < radius
        mask_in_circle7 = torch.norm(test_points - centers[6], dim=1) < radius
        mask_in_circle8 = torch.norm(test_points - centers[7], dim=1) < radius
        
        mask_in_any_circle = mask_in_circle1 | mask_in_circle2 | mask_in_circle3 | mask_in_circle4 | mask_in_circle5 | mask_in_circle6 | mask_in_circle7 | mask_in_circle8
        
        # --- 关键修正：定义外部点的掩码 ---
        mask_outside_all_circles = ~mask_in_any_circle

        # --- 2. 筛选点 (确保 Z 已经设置了 requires_grad=True) ---
        xo_outside_all_circles = test_points[mask_outside_all_circles]
        xo_rbf1 = test_points[mask_in_circle1]
        xo_rbf2 = test_points[mask_in_circle2]
        xo_rbf3 = test_points[mask_in_circle3]
        xo_rbf4 = test_points[mask_in_circle4]
        xo_rbf5 = test_points[mask_in_circle5]
        xo_rbf6 = test_points[mask_in_circle6]
        xo_rbf7 = test_points[mask_in_circle7]
        xo_rbf8 = test_points[mask_in_circle8]

        # --- 3. 计算各区域的预测值 ---
        # 确保输入类型和设备正确
        P_nn = model_nn(xo_outside_all_circles.double())
        P_rbf1 = rbf_model_1(xo_rbf1.double())
        P_rbf2 = rbf_model_2(xo_rbf2.double())
        P_rbf3 = rbf_model_3(xo_rbf3.double())
        P_rbf4 = rbf_model_4(xo_rbf4.double())
        P_rbf5 = rbf_model_5(xo_rbf5.double())
        P_rbf6 = rbf_model_6(xo_rbf6.double())
        P_rbf7 = rbf_model_7(xo_rbf7.double())
        P_rbf8 = rbf_model_8(xo_rbf8.double())

        # --- 4. 修正：正确拼接预测值和坐标 ---
        pred_u = torch.cat((P_nn, P_rbf1, P_rbf2, P_rbf3, P_rbf4, P_rbf5, P_rbf6, P_rbf7, P_rbf8), dim=0)
        concat_Z = torch.cat((xo_outside_all_circles, xo_rbf1, xo_rbf2, xo_rbf3, xo_rbf4, xo_rbf5, xo_rbf6, xo_rbf7, xo_rbf8), dim=0)
        # 计算绝对误差
        errors = pred_u.flatten()

        # --- 6. 绘制散点图 (这部分逻辑是正确的) ---
        X = concat_Z[:, 0].cpu().detach().numpy()
        Y = concat_Z[:, 1].cpu().detach().numpy()
        Z = concat_Z[:, 2].cpu().detach().numpy()
        data_to_save = np.c_[X, Y, Z, errors.cpu().detach().numpy()]
        # 2. 定义文件名
        filename_txt = '3D_darcy_adam_prediction.txt'
        np.savetxt(filename_txt, data_to_save, fmt='%.6f', delimiter=' ', header='X Y Z pred_u')
        print(f"数据已成功保存到 '{filename_txt}'")
        """
time_end = time.time()
print('time cost', time_end - time_start, 's')
        


    
 