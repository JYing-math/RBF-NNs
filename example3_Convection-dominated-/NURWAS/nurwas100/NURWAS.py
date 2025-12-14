import torch
import time
import numpy as np
import os
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from pyDOE import lhs
import sys
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy import stats as sts
import math
from functorch import make_functional, vmap, grad, jacrev, hessian
import torch.autograd as autograd
import torch.optim.lr_scheduler as lr_scheduler

# select device and dtype
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
# parameters for one peak problem
lb = np.array([0, 0, 0])
ub = np.array([1, 1, 1])
epochs = 20000

# generate data
N_b, N_f = 300, 5000

# random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Ω:[0,1]×[0,1]x[0,1]上随机取点
def get_omega_points(num):
    x1 = torch.rand(num, 1)
    x2 = torch.rand(num, 1)
    x3 = torch.rand(num, 1)
    x = torch.cat((x1, x2,x3), dim=1)
    return x

# 边界上取点
def get_boundary_points(num):
    index1 = torch.rand(num,1)
    index2 = torch.rand(num, 1)
    xb1 = torch.cat((index1, index2, torch.ones_like(index1)), dim=1)
    xb2 = torch.cat((index1, index2, torch.full_like(index1, 0)), dim=1)
    xb3 = torch.cat((index1, torch.ones_like(index1), index2), dim=1)
    xb4 = torch.cat((index1, torch.full_like(index1, 0), index2), dim=1)
    xb5 = torch.cat((torch.ones_like(index1), index1, index2), dim=1)
    xb6 = torch.cat((torch.full_like(index2, 0), index1, index2), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)
    return xb


Re = 100

def u_x(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    u = ((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
    u = torch.unsqueeze(u, 1)
    return u

# generate initial training dataset
def generate_peak1_samples(N_b, N_f, lb, ub):
    # np.random.seed(1)
    # within domain [lb, ub]x[lb, ub]
    X_f = lb + (ub - lb) * lhs(3, N_f)
    X_b_train = get_boundary_points(N_b)
    u_b = u_x(X_b_train)
    return X_f, X_b_train.detach().cpu().numpy(), u_b.detach().cpu().numpy()

def sde_reflection(a):
    remainder = a % 1  # 向量化计算余数，保留原维度
    reflection = np.where(remainder < 0.5, remainder, 1 - remainder)  # 反射逻辑
    return reflection

sde_reflection_func = np.vectorize(sde_reflection)

class DNN(nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output
    ### depth: depth of the network
    def __init__(self, in_dim, h_dim, out_dim, depth):
        super(DNN, self).__init__()
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

def check_all_points_in_cube(points, eps=1e-6):
    """
    判断所有点是否全在 [0-eps, 1+eps]^3 立方体内部（eps处理浮点误差）
    
    Args:
        points: 输入点集，形状为 (N, 3)，支持 PyTorch 张量或 NumPy 数组
        eps: 容差，默认 1e-6（允许因浮点运算导致的微小超出）
    
    Returns:
        bool: 所有点都在范围内返回 True，否则返回 False
    """
    # 统一转换为 numpy 数组（兼容张量和数组输入）
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    elif isinstance(points, np.ndarray):
        points_np = points
    else:
        raise TypeError("输入必须是 PyTorch 张量或 NumPy 数组")
    
    # 检查形状是否为 (N, 3)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        print("NO (输入形状不是 (N, 3))")
        return False
    
    # 判断所有点的 x,y,z 坐标是否都在 [0-eps, 1+eps] 范围内
    lower_bound = -eps
    upper_bound = 1 + eps
    all_in_range = np.all(
        (points_np >= lower_bound) & (points_np <= upper_bound)
    )
    
    if not all_in_range:
        # 可选：打印超出范围的点（方便调试）
        out_of_range_mask = ~((points_np >= lower_bound) & (points_np <= upper_bound))
        out_of_range_points = points_np[out_of_range_mask.any(axis=1)]
        print(f"NO (共 {len(out_of_range_points)} 个点超出范围)")
        # 可选：打印前 5 个超出范围的点
        if len(out_of_range_points) > 0:
            print("超出范围的点示例：", out_of_range_points[:5])
        return False
    
    return True

class PinnOnePeak:
    """This script carry out unbounded pinn one peak pdes"""

    def __init__(self, X_b_train, u_b) -> None:
        self.loss_func = nn.MSELoss()
        self.iter = 0
        self.best_l2 = 1000
        self.net = DNN(3,30,1,3).to(device)
        # 1. 加载保存的参数元组
        loaded_params_tuple = torch.load('best_model.mdl', map_location=torch.device(device))

        # 2. 获取模型的 state_dict 作为模板
        state_dict_template = self.net.state_dict()

        # 3. 将元组中的参数与 state_dict 的键配对
        # loaded_params_tuple 中的参数顺序必须和 state_dict_template.values() 的顺序完全一致
        new_state_dict = {
            key: param for key, param in zip(state_dict_template.keys(), loaded_params_tuple)
        }

        # 4. 加载 state_dict 到模型中
        self.net.load_state_dict(new_state_dict)
        # train boundary data
        self.u_b = torch.tensor(u_b, dtype=DTYPE).to(device)
        self.x_b = torch.tensor(X_b_train[:, 0].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.y_b = torch.tensor(X_b_train[:, 1].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.z_b = torch.tensor(X_b_train[:, 2].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        x1 = torch.linspace(0, 1, 30)
        x2 = torch.linspace(0, 1, 30)
        x3 = torch.linspace(0, 1, 30)
        X1, X2,X3 = torch.meshgrid(x1, x2,x3)
        self.points = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None],X3.flatten()[:, None]),dim=1).to(device)
        self.true_u = u_x(self.points).to('cpu').detach().numpy()
        self.optim_adam = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optim_adam, 
            T_max= epochs, 
            eta_min=1e-5  # 最低学习率
        )
    def net_u(self, x):
        u = self.net(x)
        return u

    def f_x(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        du_dx = - Re * (torch.exp(Re * x1) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
        d2u_dx2 = -(Re ** 2 * torch.sin(x2) * torch.sin(x3) * torch.exp(Re * x1)) / (torch.exp(torch.tensor(Re)) - 1)
        d2u_dy2 = -((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
        d2u_dz2 = -((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3)) / (torch.exp(torch.tensor(Re)) - 1)
        f = -(d2u_dx2 + d2u_dy2 + d2u_dz2) + Re * du_dx
        f = torch.unsqueeze(f, 1)
        return f

    def net_f(self, x):
        output_op = self.net_u(x)
        f_omega = self.f_x(x).reshape(-1, 1)
        grad_op = autograd.grad(outputs=output_op, inputs=x,
                                grad_outputs=torch.ones_like(output_op),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = torch.unsqueeze(grad_op[:, 0],1)
        du_dy = torch.unsqueeze(grad_op[:, 1],1)
        du_dz = torch.unsqueeze(grad_op[:, 2],1)
        grad_op1 = autograd.grad(outputs=du_dx, inputs=x,
                                grad_outputs=torch.ones_like(du_dx),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_op2 = autograd.grad(outputs=du_dy, inputs=x,
                                grad_outputs=torch.ones_like(du_dx),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_op3 = autograd.grad(outputs=du_dz, inputs=x,
                                grad_outputs=torch.ones_like(du_dz),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xx = torch.unsqueeze(grad_op1[:, 0],1)
        u_yy = torch.unsqueeze(grad_op2[:, 1],1)
        u_zz = torch.unsqueeze(grad_op3[:, 2],1)
        loss_op =-(u_xx + u_yy + u_zz) - f_omega + Re * du_dx
        return loss_op

    def update(self, X_f_train):
        self.x_f = torch.tensor(X_f_train[:, 0].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f_train[:, 1].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.z_f = torch.tensor(X_f_train[:, 2].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)

    def train(self, X_f_train, adam_iters, sampling_method, lower_bound=np.array([0, 0, 0]),
              upper_bound=np.array([1, 1, 1])):
        self.update(X_f_train)
        self.net.train()
        self.error_fair, self.loss_f, self.loss_f_fair, self.loss_b, self.loss_f_IS = [], [], [], [], []
        self.grad_total = []

        # sde parameter
        num = len(X_f_train)
        eta = 1. / 10
        eta1 = 1. / 20

        # mini-batch parameter
        batch_sz = 5000
        n_batches = num // batch_sz

        for i in range(adam_iters):
            for j in range(n_batches):
                x_f_batch = self.x_f[j * batch_sz:(j * batch_sz + batch_sz), ]
                y_f_batch = self.y_f[j * batch_sz:(j * batch_sz + batch_sz), ]
                z_f_batch = self.z_f[j * batch_sz:(j * batch_sz + batch_sz), ]

                self.optim_adam.zero_grad()
                u_b_prediction = self.net_u(torch.cat([self.x_b, self.y_b, self.z_b], dim=1))
                f_prediction = self.net_f(torch.cat([x_f_batch, y_f_batch, z_f_batch], dim=1))

                # loss
                u_b_loss = self.loss_func(u_b_prediction, self.u_b)
                f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
                ls = f_loss + 100*u_b_loss
                ls.backward()
                self.optim_adam.step()

                if sampling_method == 'sde':
                    # 调整扰动系数（与二维逻辑一致，基于当前损失动态调整）
                    eta = 1. / 8 
                    
                    # 提取残差f（绝对值，加小量避免除零）
                    f = torch.abs(f_prediction) + 1e-8  # 张量上计算平方+偏移
                    f = f.detach().cpu().numpy().reshape(-1, 1)  # 转numpy后强制保留[batch_sz, 1]
                    mean_f = sts.hmean(f)  # 残差的调和平均，用于平衡扰动幅度
                    # --------------- 处理x维度 ---------------
                    # 原始x坐标 + 基于残差的随机扰动（残差越大，扰动越大，越可能在该区域密集采样）
                    tmp_x = (X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 0:1]  # 取当前batch的x列
                            + eta * mean_f / f * np.random.randn(batch_sz, 1)  # 随机扰动项（服从正态分布）
                            - lower_bound[0])  # 平移到[0, 2)区间（为反射做准备）
                    # 反射越界点，映射回有效域[lower_bound[0], upper_bound[0]]
                    x_f_batch = sde_reflection_func(tmp_x) + lower_bound[0]
                    
                    # --------------- 处理y维度 ---------------
                    tmp_y = (X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 1:2]  # 取当前batch的y列
                            + eta * mean_f / f * np.random.randn(batch_sz, 1)
                            - lower_bound[1])
                    y_f_batch = sde_reflection_func(tmp_y) + lower_bound[1]
                    
                    # --------------- 新增：处理z维度 ---------------
                    tmp_z = (X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 2:3]  # 取当前batch的z列（三维新增）
                            + eta * mean_f / f * np.random.randn(batch_sz, 1)  # 相同的扰动逻辑
                            - lower_bound[2])  # z维度下界平移
                    z_f_batch = sde_reflection_func(tmp_z) + lower_bound[2]  # z维度反射
                    
                    # 更新训练数据的x、y、z列（三维需同时更新z）
                    X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 0:1] = x_f_batch
                    X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 1:2] = y_f_batch
                    X_f_train[j * batch_sz:(j * batch_sz + batch_sz), 2:3] = z_f_batch  # 新增z列更新
                self.update(X_f_train)  # 更新网络输入的张量（需确保包含z维度）
            self.scheduler.step()
            # 打印当前的学习率（可选，但推荐）
            u_predict, f_predict = self.predict(self.points)
            f_predict = f_predict.reshape(self.true_u.shape)
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                        # 计算L2范数 (欧几里得范数)
                    l2_norm = torch.norm(param.grad).item()
            tmp1 = (np.linalg.norm(u_predict.squeeze() - self.true_u.flatten()) /
                        np.linalg.norm(self.true_u.flatten()))
            tmp2 = np.mean(f_predict ** 2)
            if tmp1 < self.best_l2:
                self.best_l2 = tmp1
            print('current epoch: %d, loss: %.5e, relative error: %.5e, best_l2: %.5e' % (i, ls.item(), tmp1, self.best_l2))
            f_predict = f_predict.reshape(self.true_u.shape)

            self.error_fair.append(tmp1)
            self.loss_f.append(f_loss.to('cpu').detach().numpy())
            self.loss_f_fair.append(tmp2)
            self.loss_b.append(u_b_loss.to('cpu').detach().numpy())
            self.grad_total.append(l2_norm)

    def predict(self, points):
        x = torch.tensor(points[:, 0:1], requires_grad=True).to(device)
        y = torch.tensor(points[:, 1:2], requires_grad=True).to(device)
        z = torch.tensor(points[:, 2:3], requires_grad=True).to(device)
        self.net.eval()
        u = self.net_u(torch.cat([x, y, z], dim=1))
        f = self.net_f(torch.cat([x, y, z], dim=1))
        u = u.to('cpu').detach().numpy()
        f = f.to('cpu').detach().numpy()
        return u, f

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    time_start = time.time()
    X_f_train, X_b_train, u_b = generate_peak1_samples(N_b, N_f, lb, ub)
    pinn = PinnOnePeak(X_b_train, u_b)
    pinn.train(X_f_train, epochs, 'sde')
    time_end = time.time()
    print("Total time cost: ", time_end - time_start)