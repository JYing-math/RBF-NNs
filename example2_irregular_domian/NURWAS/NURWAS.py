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
lb = np.array([-0.2, -0.2, -0.2])
ub = np.array([0.2, 0.2, 0.2])
epochs = 20000

# generate data
N_b, N_f = 400, 4000

'''-------------------------Data generator-------------------------'''
def get_omega_points(num, x_left=-1.0, y_left=-1.0, z_left=-1.0,
                        x_right=1.0, y_right=1.0, z_right=1.0):
    """
    在三维区域 Ω: [-1,1]^3 \ [0,1]×[0,1]×[-1,1] 内随机取点。
    即一个立方体挖去其右上角（x,y>0）的一个长方体后形成的区域。
    """
    points = []
    while len(points) < num:
        # 在整个大立方体内随机生成一个点
        x = torch.rand(1) * (x_right - x_left) + x_left
        y = torch.rand(1) * (y_right - y_left) + y_left
        z = torch.rand(1) * (z_right - z_left) + z_left

        # 判断点是否在挖去的长方体内 (0 <= x <= 1 and 0 <= y <= 1)
        # 如果不在，则保留该点
        if not (0 <= x <= 1 and 0 <= y <= 1):
            points.append([x, y, z])

    # 将列表转换为形状为 (num, 3) 的张量
    return torch.tensor(points).numpy()

def get_boundary_points(num, x_left=-1.0, y_left=-1.0, z_left=-1.0,
                           x_right=1.0, y_right=1.0, z_right=1.0):
    """
    生成三维区域 Ω: [-1,1]^3 \ [0,1]×[0,1]×[-1,1] 的边界点。
    该区域有8个边界面：原立方体的6个外表面（部分被挖去）和挖去长方体后产生的2个内表面。
    """
    # 1. 计算每个面的点数（均匀分配）
    points_per_face = num // 8
    extra_points = num % 8
    face_counts = [points_per_face + 1 if i < extra_points else points_per_face for i in range(8)]
    
    # 为了代码清晰，将8个面的点数分别命名
    f1, f2, f3, f4, f5, f6, f7, f8 = face_counts

    points_list = []
    
    # --- 外边界面 (6个) ---
    
    # 面1: x = x_left (左外表面, 完整)
    y = y_left + (y_right - y_left) * torch.rand(f1, 1)
    z = z_left + (z_right - z_left) * torch.rand(f1, 1)
    points_list.append(torch.cat((torch.full_like(y, x_left), y, z), dim=1))

    # 面2: y = y_left (下外表面, 完整)
    x = x_left + (x_right - x_left) * torch.rand(f2, 1)
    z = z_left + (z_right - z_left) * torch.rand(f2, 1)
    points_list.append(torch.cat((x, torch.full_like(x, y_left), z), dim=1))

    # 面3: z = z_left (后外表面, 部分被挖去)
    count = 0
    face3_points = []
    while count < f3:
        x = x_left + (x_right - x_left) * torch.rand(1, 1)
        y = y_left + (y_right - y_left) * torch.rand(1, 1)
        if not (0 <= x <= 1 and 0 <= y <= 1): # 排除挖去区域的部分
            face3_points.append(torch.cat((x, y, torch.full_like(x, z_left)), dim=1))
            count += 1
    points_list.append(torch.cat(face3_points, dim=0))

    # 面4: x = x_right (右外表面, 仅在 y < 0 的部分存在)
    y = y_left + (0 - y_left) * torch.rand(f4, 1) # y 仅在 [-1, 0) 范围内
    z = z_left + (z_right - z_left) * torch.rand(f4, 1)
    points_list.append(torch.cat((torch.full_like(y, x_right), y, z), dim=1))

    # 面5: y = y_right (上外表面, 仅在 x < 0 的部分存在)
    x = x_left + (0 - x_left) * torch.rand(f5, 1) # x 仅在 [-1, 0) 范围内
    z = z_left + (z_right - z_left) * torch.rand(f5, 1)
    points_list.append(torch.cat((x, torch.full_like(x, y_right), z), dim=1))

    # 面6: z = z_right (前外表面, 部分被挖去)
    count = 0
    face6_points = []
    while count < f6:
        x = x_left + (x_right - x_left) * torch.rand(1, 1)
        y = y_left + (y_right - y_left) * torch.rand(1, 1)
        if not (0 <= x <= 1 and 0 <= y <= 1): # 排除挖去区域的部分
            face6_points.append(torch.cat((x, y, torch.full_like(x, z_right)), dim=1))
            count += 1
    points_list.append(torch.cat(face6_points, dim=0))

    # --- 内边界面 (2个, 挖去长方体后产生) ---
    
    # 面7: x = 0 (内表面, y, z ∈ [-1, 1])
    y = 0 + (y_right - 0) * torch.rand(f7, 1)
    z = z_left + (z_right - z_left) * torch.rand(f7, 1)
    points_list.append(torch.cat((torch.full_like(y, 0.0), y, z), dim=1))

    # 面8: y = 0 (内表面, x, z ∈ [-1, 1])
    x = 0 + (x_right - 0) * torch.rand(f8, 1)
    z = z_left + (z_right - z_left) * torch.rand(f8, 1)
    points_list.append(torch.cat((x, torch.full_like(x, 0.0), z), dim=1))

    # 3. 合并所有面的点
    xb = torch.cat(points_list, dim=0)
    return xb

def generate_peak1_samples(N_b, N_f, lb, ub):
    # np.random.seed(1)
    # within domain [lb, ub]x[lb, ub]
    X_f = get_omega_points(N_f, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2)
    X_b_train = get_boundary_points(N_b, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2)
    u_b = u_x(X_b_train)
    return X_f, X_b_train.detach().cpu().numpy(), u_b.detach().cpu().numpy()
'''-------------------------Functions-------------------------'''

def u_x(x):
    """
    三维区域上的函数 u(x, y, z)。
    拓展自二维版本 u(x, y) = (x^2 + y^2)^(1/3)。
    """
    r_squared = torch.sum(torch.pow(x, 2), 1)
    ux = torch.pow(r_squared, 1/3)
    return ux.unsqueeze(1)

def g_x(x):
    """
    三维区域边界上的函数 g(x, y, z)。
    通常 g 是 u 在边界上的限制，即 g(x) = u(x) for x ∈ ∂Ω。
    """
    return u_x(x)

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

class PinnOnePeak:
    """This script carry out unbounded pinn one peak pdes"""

    def __init__(self, X_b_train, u_b) -> None:
        self.loss_func = nn.MSELoss()
        self.iter = 0
        self.best_l2 = 1000
        self.net = DNN(3,20,1,3).to(device)
        func_params = torch.load('best_model.mdl', map_location=device)
        self.net.load_state_dict(func_params) 
        # train boundary data
        self.u_b = torch.tensor(u_b, dtype=DTYPE).to(device)
        self.x_b = torch.tensor(X_b_train[:, 0].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.y_b = torch.tensor(X_b_train[:, 1].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.z_b = torch.tensor(X_b_train[:, 2].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        x1 = torch.linspace(-0.2, 0.2, 30)
        x2 = torch.linspace(-0.2, 0.2, 30)
        x3 = torch.linspace(-0.2, 0.2, 30)
        X1, X2, X3 = torch.meshgrid(x1, x2, x3, indexing='ij') # 建议加上 indexing='ij' 明确索引方式
        Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None],X3.flatten()[:, None]),dim=1)
        mask = ~((Z[:, 0] >= 0) & (Z[:, 0] <= 0.2) & (Z[:, 1] >= 0) & (Z[:, 1] <= 0.2))
        # 3. 应用掩码，筛选出有效点
        self.points = Z[mask].to(device)
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
        """
        三维Poisson方程中的源项 f(x, y, z)。
        它是通过对 u(x, y, z) 求拉普拉斯算子得到的，即 f = -Δu。
        """
        r_squared = torch.sum(torch.pow(x, 2), 1)
        r = torch.pow(r_squared, -2/3)
        # 为了数值稳定性，避免 r=0 时出现无穷大
        eps = 1e-8
        r = torch.clamp(r, min=eps)
        fx = -(10/9)*r
        return fx.unsqueeze(1)

    def net_f(self,  x_o):
        u = self.net_u(x_o)
        f_o = self.f_x(x_o)
        u_xyz = torch.autograd.grad(u, x_o, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        d2u_x_xyz = torch.autograd.grad(u_xyz[:,0].reshape(-1,1), x_o, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        d2u_y_xyz = torch.autograd.grad(u_xyz[:,1].reshape(-1,1), x_o, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        d2u_z_xyz = torch.autograd.grad(u_xyz[:,2].reshape(-1,1), x_o, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        u_x = u_xyz[:,0].reshape(-1,1)
        u_xx = d2u_x_xyz[:,0].reshape(-1,1)
        u_yy = d2u_y_xyz[:,1].reshape(-1,1)
        u_zz = d2u_z_xyz[:,2].reshape(-1,1)
        loss_o = -(u_xx + u_yy + u_zz) - f_o
        return loss_o


    def update(self, X_f_train):
        self.x_f = torch.tensor(X_f_train[:, 0].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f_train[:, 1].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.z_f = torch.tensor(X_f_train[:, 2].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)

    def train(self, X_f_train, adam_iters, sampling_method, lower_bound=np.array([-0.2, -0.2, -0.2]), # 注意：边界也应设为整个立方体的边界
            upper_bound=np.array([0.2, 0.2, 0.2])):
        self.update(X_f_train)
        self.net.train()
        self.error_fair, self.loss_f, self.loss_f_fair, self.loss_b, self.loss_f_IS = [], [], [], [], []
        self.grad_total = []

        # sde parameter
        num = len(X_f_train)
        eta = 1. / 10

        # mini-batch parameter
        batch_sz = min(4000, num) # 确保batch_sz不大于总点数
        n_batches = num // batch_sz

        for i in range(adam_iters):
            # 为了公平比较，每个epoch开始时打乱数据（如果需要）
            # indices = np.random.permutation(num)
            # X_f_train_shuffled = X_f_train[indices]
            
            for j in range(n_batches):
                # 计算批次的起始和结束索引
                start_idx = j * batch_sz
                end_idx = start_idx + batch_sz
                
                # 从原始数据中获取批次（如果打乱了就用打乱后的）
                x_f_batch = self.x_f[start_idx:end_idx, ]
                y_f_batch = self.y_f[start_idx:end_idx, ]
                z_f_batch = self.z_f[start_idx:end_idx, ]

                self.optim_adam.zero_grad()
                u_b_prediction = self.net_u(torch.cat([self.x_b, self.y_b, self.z_b], dim=1))
                f_prediction = self.net_f(torch.cat([x_f_batch, y_f_batch, z_f_batch], dim=1))

                # loss
                u_b_loss = self.loss_func(u_b_prediction, self.u_b)
                f_loss = self.loss_func(f_prediction, torch.zeros_like(f_prediction).to(device))
                ls = f_loss + 1000*u_b_loss
                ls.backward()
                self.optim_adam.step()

                if sampling_method == 'sde' and j == 0: # 可以选择每个epoch只更新一次采样点，以节省计算
                    # 调整扰动系数
                    eta = 1. / 2000 
                    
                    # 提取残差f（绝对值，加小量避免除零）
                    f = torch.abs(f_prediction)
                    f_np = f.detach().cpu().numpy().reshape(-1, 1)
                    
                    # 计算调和平均
                    mean_f = sts.hmean(f_np)
                    
                    # 获取当前批次的完整三维点
                    current_batch_points = X_f_train[start_idx:end_idx, :]
                    
                    # 生成随机扰动
                    random_noise = np.random.randn(batch_sz, 3)
                    
                    # 计算扰动幅度
                    perturbation = eta * mean_f / f_np * random_noise
                    
                    # 对当前批次的点进行扰动
                    perturbed_points = current_batch_points + perturbation
                    
                    # 将反射后的点更新回 X_f_train
                    X_f_train[start_idx:end_idx, :] = perturbed_points
                    
                    # 更新self.x_f, self.y_f, self.z_f以反映新的采样点
                    self.update(X_f_train)

            self.scheduler.step()
            
            # 评估和打印
            u_predict, f_predict = self.predict(self.points)
            f_predict = f_predict.reshape(self.true_u.shape)
            
            # 计算梯度范数
            l2_norm = 0.0
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    l2_norm += torch.norm(param.grad).item()**2
            l2_norm = math.sqrt(l2_norm)
            
            tmp1 = (np.linalg.norm(u_predict.squeeze() - self.true_u.flatten()) /
                        np.linalg.norm(self.true_u.flatten()))
            tmp2 = np.mean(f_predict ** 2)
            
            if tmp1 < self.best_l2 and i > 1000:
                self.best_l2 = tmp1 
                # 可以在这里保存最佳模型
                torch.save(self.net.state_dict(), 'best_model_nurwas.mdl')
                
            print('current epoch: %d, loss: %.5e, relative error: %.5e, best_l2: %.5e' % (i, ls.item(), tmp1, self.best_l2))

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
    time_start = time.time()
    X_f_train, X_b_train, u_b = generate_peak1_samples(N_b, N_f, lb, ub)
    pinn = PinnOnePeak(X_b_train, u_b)
    pinn.train(X_f_train, epochs, 'sde')
    time_end = time.time()
    print("Total time cost: ", time_end - time_start)