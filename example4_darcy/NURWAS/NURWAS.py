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
import torch.autograd as autograd
from scipy.interpolate import griddata

# select device and dtype
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32
# parameters for one peak problem
lb = np.array([0, 0, 0])
ub = np.array([1, 1, 1])
epochs = 20000

# generate data
N_b, N_f = 600, 10000

# random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_cube_boundary_points(num: int) -> torch.Tensor:
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

def u_x_boundary(x):
    u_b = torch.zeros_like(x[:,0])
    return u_b

# generate initial training dataset
def generate_peak1_samples(N_b, N_f, lb, ub):
    # np.random.seed(1)
    # within domain [lb, ub]x[lb, ub]
    X_f = lb + (ub - lb) * lhs(3, N_f)
    X_b_train = get_cube_boundary_points(N_b)
    u_b = u_x_boundary(X_b_train)
    return X_f, X_b_train.detach().cpu().numpy(), u_b.detach().cpu().numpy()

def sde_reflection(a):
    remainder = a % 1  # 向量化计算余数，保留原维度
    reflection = np.where(remainder < 0.5, remainder, 1 - remainder)  # 反射逻辑
    return reflection

sde_reflection_func = np.vectorize(sde_reflection)

def plot_z01_slice(points, pred_vals):
    """
    绘制z=0.1切面图（与第一个图样式一致），仅展示预测值
    参数：
        points: 验证点 (tensor, N×3)
        pred_vals: 预测值 (tensor, N)
    """
    # 1. 转换为numpy数组（处理tensor的设备和梯度）
    pts_np = points.detach().cpu().numpy() if hasattr(points, 'detach') else points
    pred_np = pred_vals.detach().cpu().numpy() if hasattr(pred_vals, 'detach') else pred_vals
    
    # 2. 筛选z≈0.1的点（容差±0.01）
    z_vals = pts_np[:, 2]
    z_mask = np.abs(z_vals - 0.1) < 1e-2
    
    if not np.any(z_mask):
        print("警告：未找到z≈0.1的验证点，跳过切面图绘制")
        return
    
    # 提取筛选后的点和预测值
    x_slice = pts_np[z_mask, 0]
    y_slice = pts_np[z_mask, 1]
    pred_slice = pred_np[z_mask]
    
    # 3. 构建网格用于插值（生成连续热力图）
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_slice.min(), x_slice.max(), 200),
        np.linspace(y_slice.min(), y_slice.max(), 200)
    )
    
    # 4. 插值预测值（立方插值保证平滑）
    pred_grid = griddata((x_slice, y_slice), pred_slice, (grid_x, grid_y), method='cubic')
    
    # 5. 绘制与第一个图一致的切面图
    # 调整figsize为正方形（匹配第一个图的比例）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # 预测值热力图：调整aspect为equal，保证X/Y轴比例一致
    im = ax.imshow(
        pred_grid, 
        extent=(x_slice.min(), x_slice.max(), y_slice.min(), y_slice.max()),
        origin='lower', 
        cmap='viridis', 
        aspect='equal'  # 关键：让X/Y轴比例一致，呈现正方形
    )
    
    # 【去掉】叠加原始采样点的scatter（第一个图无白色小点）
    
    # 图表样式配置
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # 【去掉】legend（无图例元素）
    
    # 颜色条（调整缩放适配图表）
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

class DNN(nn.Module):
    """This class carry out DNN"""

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
        self.net = DNN(3, 20, 1, 7).to(device)
        self.net.load_state_dict(torch.load('3d_best_model_first_LM.mdl'))
        # train boundary data
        self.u_b = torch.tensor(u_b, dtype=DTYPE).to(device)
        self.x_b = torch.tensor(X_b_train[:, 0].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.y_b = torch.tensor(X_b_train[:, 1].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        self.z_b = torch.tensor(X_b_train[:, 2].reshape(-1, 1), dtype=DTYPE, requires_grad=True).to(device)
        test_point = np.loadtxt("3D_darcy_adam_prediction.txt")
        self.points = test_point[:, 0:3]
        self.true_u = np.loadtxt("pred_u_FE_300.txt")
        self.optim_adam = torch.optim.Adam(self.net.parameters(), lr=1e-4)

    def net_u(self, x):
        u = self.net(x)
        return u

    def f_x(self, x):
        f = torch.ones_like(x[:,0]) * 10
        return f

    def alpha_x(self, x_op, radius: float = 0.2) -> torch.Tensor:
        
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

    def net_f(self, x):
        output_op = self.net_u(x)
        f_omega = self.f_x(x).reshape(-1, 1)
        a = self.alpha_x(x.cpu()).to(device).reshape(-1, 1)
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
        loss_op = a * (u_xx + u_yy + u_zz) + f_omega
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
        batch_sz = 10000
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
                ls = f_loss + 100 * u_b_loss
                ls.backward()
                self.optim_adam.step()

                if sampling_method == 'sde':
                    # 调整扰动系数（与二维逻辑一致，基于当前损失动态调整）
                    eta = 1. / 8 * (ls.item() / 1000)**(1/2)
                    
                    # 提取残差f（绝对值，加小量避免除零）
                    f = torch.pow(torch.abs(f_prediction), 2) + 1e-8  # 张量上计算平方+偏移
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

            u_predict, f_predict = self.predict(self.points)
            f_predict = f_predict.reshape(self.true_u.shape)
            print('current epoch: %d, loss: %.7f, loss_f: %.7f, loss_b: %.7f' % (i, ls.item(), f_loss, u_b_loss))

            for name, param in self.net.named_parameters():
                if param.grad is not None:
                        # 计算L2范数 (欧几里得范数)
                    l2_norm = torch.norm(param.grad).item()

            tmp1 = (np.linalg.norm(u_predict.squeeze() - self.true_u.flatten()) /
                        np.linalg.norm(self.true_u.flatten()))
            tmp2 = np.mean(f_predict ** 2)
            print("relative error:",tmp1)
            f_predict = f_predict.reshape(self.true_u.shape)

            self.error_fair.append(tmp1)
            self.loss_f.append(f_loss.to('cpu').detach().numpy())
            self.loss_f_fair.append(tmp2)
            self.loss_b.append(u_b_loss.to('cpu').detach().numpy())
            self.grad_total.append(l2_norm)
            if (i + 1) % 10000 == 0:
                plot_z01_slice(self.points, u_predict) 
    def predict(self, points):
        x = torch.tensor(points[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(points[:, 1:2], requires_grad=True).float().to(device)
        z = torch.tensor(points[:, 2:3], requires_grad=True).float().to(device)
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