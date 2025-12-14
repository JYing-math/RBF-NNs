import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functools

from torch import autograd
from functorch import make_functional, vmap, grad, jacrev, hessian
import matplotlib.tri as tri
from collections import namedtuple, OrderedDict
import datetime
import time
# 记录开始时间
start_time = time.perf_counter()
import sys
import os
import warnings
warnings.filterwarnings('ignore')

'''  Solve the following PDE

'''
'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)
'''-------------------------Pre-setup-------------------------'''
# iteration counts and check
tr_iter_max    = 2000                      # max. iteration
ts_input_new   = 1000                       # renew testing points
ls_check       = 10
ls_check0      = ls_check - 1
# number of training points and testing points
N_tsd_final = 250000 #100*N_trd
N_tsg_final = 1000   #10*N_trg
# tolerence for LM
tol_main    = 10**(-12)
tol_machine = 10**(-15)
mu_max      = 10**8
mu_ini      = 10**8
'''-------------------------Data generator-------------------------'''
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
    # 1. 处理单个样本的情况 (vmap 会传入单个样本)
    if x_op.dim() == 1:
        x_op = x_op.unsqueeze(0) # 现在 shape: (1, 3)

    # 2. 直接在 PyTorch 张量上操作
    centers = torch.tensor([
        [0.2, 0.2, 0.2], [0.2, 0.2, 0.8], [0.2, 0.8, 0.2], [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.2], [0.8, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.8, 0.8]
    ], dtype=x_op.dtype, device=x_op.device)
    
    # 3. 计算距离 (使用广播机制)
    dists = torch.norm(x_op.unsqueeze(1) - centers.unsqueeze(0), dim=2)
    
    # 4. 判断是否在任何一个角落区域内
    is_in_corner = (dists < radius).any(dim=1) # shape: (N,)
    
    # 5. 使用 `torch.where` 进行条件赋值 (非原地操作)
    # 创建一个全为 137.0 的张量
    alpha_high = torch.full_like(is_in_corner, 137.0, dtype=x_op.dtype)
    # 创建一个全为 0.018 的张量
    alpha_low = torch.full_like(is_in_corner, 0.018, dtype=x_op.dtype)
    
    # 根据 is_in_corner 的值，从 alpha_high 或 alpha_low 中选择元素
    alpha_tensor = torch.where(is_in_corner, alpha_high, alpha_low)
    
    # 或者使用更简洁的广播形式 (推荐)
    # alpha_tensor = torch.where(is_in_corner, 137.0, 0.018)
    
    return alpha_tensor

'''-------------------------Functions-------------------------'''

def f_x(x):
    f = torch.ones_like(x[:,0]) * 10
    return f

def u_x_boundary(x):
    u_b = torch.zeros_like(x[:,0])
    return u_b

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
'''-------------------------Loss functions-------------------------'''
def func_loss_o(func_params, x_o,f_o):
    a = alpha_x(x_o.cpu()).to(device).reshape(-1, 1)
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    d2u = jacrev(jacrev(f))(x_o, func_params)
    u_xx = d2u[0][0]
    u_yy = d2u[1][1]
    u_zz = d2u[2][2]
    loss_o = a * (u_xx + u_yy + u_zz) + f_o
    return loss_o

def func_loss_b(func_params, x_b, f_b):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return loss_b


'''-------------------------Levenberg-Marquardt (LM) optimizer-------------------------'''
# parameters counter
def count_parameters(func_params):
    return sum(p.numel() for p in func_params if p.requires_grad)

# get the model's parameter
def get_p_vec(func_params):
    p_vec = []
    cnt = 0
    for p in func_params:
        p_vec = p.contiguous().view(-1) if cnt == 0 else torch.cat([p_vec, p.contiguous().view(-1)])
        cnt = 1
    return p_vec

# Initialization of LM method
def generate_initial_LM(func_params, Xo_len, Xb_len):
    # data_length
    data_length = Xo_len + Xb_len # 输入数据长度和

    # p_vector p向量自然为model参数
    with torch.no_grad():
        p_vec_old = get_p_vec(func_params).double().to(device)

    # dp 初始所有参量搜索方向设置为0，其size应当和model参数一致
    dp_old = torch.zeros([count_parameters(func_params), 1]).double().to(device)

    # Loss 损失函数值同样设置为0
    L_old = torch.zeros([data_length, 1]).double().to(device)

    # Jacobian J矩阵同样
    J_old = torch.zeros([data_length, count_parameters(func_params)]).double().to(device)

    return p_vec_old, dp_old, L_old, J_old

def train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg):
    best_l2, best_epoch = 1000, 0
    # assign tuple elements of LM_set_up
    p_vec_o, dp_o, L_o, J_o, mu, criterion = LM_setup # old参数导入
    I_pvec = torch.eye(len(p_vec_o)).to(device) # 单位阵

    # assign tuple elements of data_input
    [X_o, F_o, X_b, F_b, NL, NL_sqrt] = tr_input #训练参数
    data_new = get_newData(func_params,density,epsilon,support, num_omega,new_num)
    data_new = torch.tensor(data_new, requires_grad=True).double().to(device)
    X_o = torch.cat([X_o, data_new], 0)
    torch.save(data_new,'data_new-1-sais',)
    # data_new = get_newData(params,epsilon,support, num_omega,new_points)
    # data_new = torch.tensor(data_new, requires_grad=True).double().to(device)
    # X_o = torch.cat((X_o, data_new), dim=0)
    print(len(data_new))
    # 随机扔掉一部分点保证合起来之后的点和最初的点数相同
    X_o = X_o[torch.randint(len(X_o), (num_omega,))]
    F_o = f_x(X_o)
    # 不扔点的话就要修改NL 和 NL_SQURT
    # len_sum = num_omega + 4 * num_b + len(data_new)
    # NL = [len_sum, num_omega + len(data_new), 4 * num_b]
    # NL_sqrt = np.sqrt(NL)
    # iteration counts and check
    Comput_old = True
    step = 0

    # try-except statement to avoid jam in the code
    try:
        # while (lossval[-1] > tol_main) and (step <= tr_iter_max):
        while (step <= tr_iter_max):
            torch.cuda.empty_cache()

            ############################################################
            # LM_optimizer
            if (Comput_old == True):  # need to compute loss_old and J_old

                ### computation of loss 计算各部分损失函数
                Lo = vmap((func_loss_o), (None, 0, 0))(func_params, X_o, F_o).flatten().detach()
                Lb = vmap((func_loss_b), (None, 0, 0))(func_params, X_b, F_b).flatten().detach()
                L = torch.cat((Lo / NL_sqrt[1], Lb / NL_sqrt[2]))
                L = L.reshape(NL[0], 1).detach()
                lsd_sum = torch.sum(Lo * Lo) / NL[1]
                lsb_sum = torch.sum(Lb * Lb) / NL[2]
                loss_dbg_old = [lsd_sum.item(), lsb_sum.item()]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]

            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = vmap(jacrev(func_loss_o), (None, 0, 0))(func_params, X_o, F_o)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_o = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_o, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_b), (None, 0, 0))(func_params, X_b, F_b)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_o / NL_sqrt[1], J_b / NL_sqrt[2])).detach()
                # 组装好了J矩阵
                ### info. normal equation of J
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                cnt = 0
                for p in func_params:
                    mm = torch.Tensor([p.shape]).tolist()[0]
                    num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                    p += dp[cnt:cnt + num].reshape(p.shape)
                    cnt += num

            ### Compute loss_new
            Lo = vmap((func_loss_o), (None, 0, 0))(func_params, X_o, F_o).flatten().detach()
            Lb = vmap((func_loss_b), (None, 0, 0))(func_params, X_b, F_b).flatten().detach()
            L = torch.cat((Lo / NL_sqrt[1], Lb / NL_sqrt[2]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsd_sum = torch.sum(Lo * Lo) / NL[1]
            lsb_sum = torch.sum(Lb * Lb) / NL[2]
            loss_dbg_new = [lsd_sum.item(), lsb_sum.item()]

            # strategy to update mu
            if (step > 0):

                with torch.no_grad():

                    # accept update
                    if loss_new < loss_old:
                        p_vec_old = p_vec.detach()
                        dp_old = dp
                        L_old = L
                        J_old = J
                        mu = max(mu / mu_div, tol_machine)
                        criterion = True  # False
                        Comput_old = False
                        lossval.append(loss_new)
                        lossval_dbg.append(loss_dbg_new)

                    else:
                        cosine = nn.functional.cosine_similarity(dp, dp_old, dim=0, eps=1e-15)
                        cosine_check = (1. - cosine) * loss_new > min(lossval)  # loss_old
                        if cosine_check:  # give up the direction
                            cnt = 0
                            for p in func_params:
                                mm = torch.Tensor([p.shape]).tolist()[0]
                                num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                                p -= dp[cnt:cnt + num].reshape(p.shape)
                                cnt += num
                            mu = min(mu_mul * mu, mu_max)
                            criterion = False
                            Comput_old = False
                        else:  # accept
                            p_vec_old = p_vec.detach()
                            dp_old = dp
                            L_old = L
                            J_old = J
                            mu = max(mu / mu_div, tol_machine)
                            criterion = True
                            Comput_old = False
                        lossval.append(loss_old)
                        lossval_dbg.append(loss_dbg_old)

            else:  # for old info.

                with torch.no_grad():

                    p_vec_old = p_vec.detach()
                    dp_old = dp
                    L_old = L
                    J_old = J
                    mu = max(mu / mu_div, tol_machine)
                    criterion = True
                    Comput_old = False
                    lossval.append(loss_new)
                    lossval_dbg.append(loss_dbg_new)

            with torch.no_grad():
                pred_u = func_model(func_params, Z).reshape(-1)
                truth_u = torch.from_numpy(np.loadtxt("pred_u_FE_300.txt")).to(device)
                # relative l2 loss
                l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
                    torch.mean(torch.pow(truth_u, 2)))
                # absolute loss
                abs_loss = torch.mean(torch.abs(pred_u - truth_u))
            if l2_loss < best_l2:
                best_l2 = l2_loss.item()
                best_epoch = step
                torch.save(func_params, 'best_model-1-pua.mdl')

            if step % ls_check == ls_check0:
                print('epoch:', step,
                      "loss:", '%.4e'% lossval[-1],
                      "l2_loss:", '%.4e' % l2_loss,
                      'abs_loss', '%.4e' % abs_loss,
                      'best_epoch', best_epoch,
                      'best_l2', '%.4e' % best_l2)
            step += 1

        print("Step %s: " % (step - 1))
        print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss

    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss


'''-------------------------generate new samples-------------------------'''
'''-------------------------LSF functions-------------------------'''
def get_newData(func_params,num,epsilon, support, N,new_num):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)

    count = 0  # 新取点的个数
    flag = True  # 第一次加点
    while (count < new_num):
        P = get_omega_points(num)
        fp = f_x(P)
        X1 = P[:, 0]
        X2 = P[:, 1]
        X3 = P[:, 2]
        P = P.to(device)
        fp = fp.to(device)
        res = torch.abs(vmap((func_loss_o), (None, 0, 0))(func_params, P, fp)).flatten().detach()
        res = np.array(res.cpu().numpy())
        failure = res > epsilon
        fail_X1 = (X1.cpu().detach().numpy() * failure).reshape(1, num)
        fail_X1 = fail_X1[fail_X1 != 0]
        fail_X2 = (X2.cpu().detach().numpy() * failure).reshape(1, num)
        fail_X2 = fail_X2[fail_X2 != 0]
        fail_X3 = (X3.cpu().detach().numpy() * failure).reshape(1, num)
        fail_X3 = fail_X3[fail_X3 != 0]
        d_support = support / N
        # m the number of expand samples
        m = 1
        data_new_temp = None
        for i in np.arange(0, fail_X1.shape[0]):
            sample_i = np.array([fail_X1[i], fail_X2[i], fail_X3[i]])
            
            # 为当前这个失败点生成 m 个有效点
            points_generated = 0
            temp_points = [] # 临时存储为当前失败点生成的有效点
            
            # 使用 while 循环，直到生成足够的有效点
            while points_generated < m:
                # 1. 生成一个候选点
                candidate = sample_i + np.sqrt(d_support) * (np.random.rand(3))
                
                # 2. 检查这个点是否在 [0, 1]^3 内
                if np.all(candidate >= 0) and np.all(candidate <= 1):
                    # 3. 如果有效，则添加到临时列表中
                    temp_points.append(candidate)
                    points_generated += 1

            # 将为当前失败点生成的 m 个有效点转换为 numpy 数组
            temp_points_np = np.array(temp_points)
            
            # 将其追加到最终的结果中
            if data_new_temp is None:
                data_new_temp = temp_points_np
            else:
                data_new_temp = np.append(data_new_temp, temp_points_np, axis=0)
        if (flag):
            data_new = data_new_temp
            flag = False
        else:
            data_new = np.append(data_new, data_new_temp, axis=0)
        count = len(data_new)
    data_new = data_new[torch.randint(len(data_new), (new_num,))]
    return data_new
'''-------------------------Train-------------------------'''
def generate_data(num_omega, num_b):
    xo = get_omega_points(num_omega)
    xb = get_cube_boundary_points(num_b)
    np.save('xo',xo)
    np.save('xb',xb)
    fo = f_x(xo)
    fb = u_x_boundary(xb)
    len_sum = num_omega + len(xb)

    NL = [len_sum, num_omega, len(xb)]
    NL_sqrt = np.sqrt(NL)

    xo_tr = torch.tensor(xo, requires_grad=True).double().to(device)
    xb_tr = torch.tensor(xb, requires_grad=True).double().to(device)
    fo_tr = torch.tensor(fo).double().to(device)
    fb_tr = torch.tensor(fb).double().to(device)

    return xo_tr, fo_tr, xb_tr, fb_tr, NL, NL_sqrt
# Essential namedtuples in the model
DataInput = namedtuple( "DataInput" , [ "X_o" , "F_o" , "X_b" , "F_b" , "NL" , "NL_sqrt"] )
LM_Setup  = namedtuple( "LM_Setup" , [ 'p_vec_o' , 'dp_o' , 'L_o' , 'J_o' , 'mu0' , 'criterion' ] )

# create names for storages
fname = 'test'
char_id = 'a'

# Network size
n_input = 3
n_hidden = 20
n_output = 1
n_depth = 7  # only used in deep NN
mu_div = 3.
mu_mul = 2.
#parameters for getting new samples
support = (1-0)*(1-0)*(1-0)
epsilon=10.3 # epsilon = 1, 10, 100
density = 50000
new_num = 1000
# number of training and test data points
c_addpt = 1.
num_omega = 10000
num_b = 600 #注意此处设置的为每个边取点数量
#num_if = 80

test_point = np.loadtxt("3D_darcy_adam_prediction.txt")
Z = torch.from_numpy(test_point[:, 0:3]).to(device)

# storages for errors, time instants, and IRK stages
relerr_loss = []
for char in char_id:
    # file name
    fname_char = fname + char

    torch.cuda.empty_cache()  # 清理变量

    # NN structure
    if n_depth == 1:  # Shallow NN
        model = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
    else:  # Deep NN
        model = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)

    # use Pytorch and functorch
    model.load_state_dict(torch.load('3d_best_model_first_LM.mdl'))
    func_model, func_params = make_functional(model)  # 获取model及其参数

    xo_tr, fo_tr, xb_tr, fb_tr, NL_tr, NL_sqrt_tr = generate_data(num_omega, num_b)
    tr_input = DataInput(X_o=xo_tr, F_o=fo_tr, X_b=xb_tr, F_b=fb_tr, NL=NL_tr, NL_sqrt=NL_sqrt_tr)

    # initialization of LM
    p_vec_old, dp_old, L_old, J_old = generate_initial_LM(func_params, NL_tr[1], NL_tr[2])  # 初始化LM算法
    print(f"No. of parameters = {len(p_vec_old)}")

    # LM_setup
    mu = 10 ** (8)  # 初始mu值设置大一点，有利于快速下降
    criterion = True
    LM_setup = LM_Setup(p_vec_o=p_vec_old, dp_o=dp_old, L_o=L_old, J_o=J_old, mu0=mu, criterion=criterion)  # 初始化参数导入

    # allocate loss
    lossval = []  # 总损失函数平均值
    lossval_dbg = []  # 各部分损失函数平均值
    lossval.append(1.)
    lossval_dbg.append([1., 1.])

    # start the timer
    cnt_start = time.time()

    # train the model by LM optimizer
    lossval, lossval_dbg, relerr_loss_char = train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg)
    relerr_loss.append(relerr_loss_char)

    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")

    print('ok')

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
pred_u = func_model(func_params, Z).reshape(-1)
plot_z01_slice(Z, pred_u)  
# plot evolution of loss
N_loss = len(lossval)
lossval = np.array(lossval).reshape(N_loss,1)
epochcol = np.linspace(1, N_loss, N_loss).reshape(N_loss,1)

plt.figure(figsize = (5,5))

plt.semilogy(epochcol, lossval)
plt.title('loss')
plt.xlabel('epoch')
plt.show()


