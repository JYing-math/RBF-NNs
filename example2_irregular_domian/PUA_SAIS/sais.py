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
    return torch.tensor(points)

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

def f_x(x):
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
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    d2u = jacrev(jacrev(f))(x_o, func_params)
    u_xx = d2u[0][0]
    u_yy = d2u[1][1]
    u_zz = d2u[2][2]
    loss_o =(u_xx + u_yy + u_zz) + f_o
    return loss_o

def func_loss_b(func_params, x_b, f_b):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return 1000*loss_b


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

    p_f, data_new = SAIS(func_params,N_1, N_2, p0, adaptive_iter_max)
    if (p_f < tolerance_p):
        print('p_f < tolerance_p',p_f)
    X_o = torch.cat([X_o, data_new], 0)
    # torch.save(data_new,'data_new-1-sais',)
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
                truth_u = u_x(Z).reshape(-1)
                # relative l2 loss
                l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
                    torch.mean(torch.pow(truth_u, 2)))
                # absolute loss
                abs_loss = torch.mean(torch.abs(pred_u - truth_u))
            if l2_loss < best_l2:
                best_l2 = l2_loss.item()
                best_epoch = step
                # torch.save(func_params, 'best_model-2-sais.mdl')

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
# lsf gx
def lsf(func_params, x):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    output_op = f(x, func_params)
    f_omega = f_x(x).reshape(-1, 1)
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
    loss_op = torch.abs((u_xx + u_yy + u_zz) + f_omega) - tolerance_r
    return loss_op.squeeze()
#通过多维高斯分布取点 dim默认为2
def get_gaussian_samples(h, num, dim=3, device='cuda'):
    # 初始化累积张量（直接在目标设备上）
    valid_points = torch.tensor([], device=device, dtype=torch.float64)
    # 批量采样大小（可根据需求调整，建议设为目标数量的1~2倍以减少循环）
    batch_size = max(num, 1000)  # 避免小批量
    
    while valid_points.shape[0] < num:
        # 批量采样（形状: [batch_size, dim]）
        samples = h.sample((batch_size,)).to(device=device, dtype=torch.float64)
        
        # 向量化条件判断（批量计算所有样本的条件）
        # 1. 所有坐标在 [-0.2, 0.2] 内
        in_large_cube = torch.all((samples >= -0.2) & (samples <= 0.2), dim=1)
        # 2. 不在 x∈[0,0.2] 且 y∈[0,0.2] 的小方块内
        x_in_range = (samples[:, 0] >= 0.0) & (samples[:, 0] <= 0.2)
        y_in_range = (samples[:, 1] >= 0.0) & (samples[:, 1] <= 0.2)
        not_in_small_cube = ~(x_in_range & y_in_range)
        
        # 合并条件掩码
        mask = in_large_cube & not_in_small_cube
        
        # 筛选有效样本并累积
        valid_batch = samples[mask]
        valid_points = torch.cat([valid_points, valid_batch], dim=0)
    
    # 截断到目标数量 + 开启梯度
    valid_points = valid_points[:num].requires_grad_(True)
    
    return valid_points
#自采样
def SAIS(params,N_1,N_2,p0,adaptive_iter_max):
    k = 1
    while(k<adaptive_iter_max):
        if(k == 1):
            x = get_omega_points(N_1)
        else:
            x = get_gaussian_samples(h,N_1)
        x = torch.tensor(x, requires_grad=True).double().to(device)
        lsf_x = lsf(params,x)
        _, indices = torch.sort(-lsf_x) #按lsf降序排列
        x_sorted = x[indices]
        N_eta =torch.numel(lsf_x[lsf_x > 0])
        N_p = int(np.floor(N_1 * p0))
        x_np = x_sorted[0:N_p, :].squeeze(1)
        if(N_eta < N_p):
            miu = torch.mean(x_np,0)
            cor = torch.cov(x_np.t())
            h = torch.distributions.MultivariateNormal(miu, cor)
            k += 1
        else:
            break
    miu_opt = torch.mean(x_np,0)
    cor_opt = torch.cov(x_np.t())
    # 正则化：添加微小对角项确保协方差正定（解决非正定错误）
    epsilon = 1e-6
    cor_opt = cor_opt + epsilon * torch.eye(3, device=device)
    h_opt = torch.distributions.MultivariateNormal(miu_opt, cor_opt)
    x_n2 = get_gaussian_samples(h_opt,N_2)
    print(x_n2.shape)
    x_n2 = torch.tensor(x_n2, requires_grad=True).double().to(device)
    x_adaptive = x_n2[lsf(params,x_n2) > 0]
    print(x_adaptive.shape)
    #pf中的indicator function是lsf即gx>0则为1 也就是只计X adaptive中的
    if(len(x_adaptive) > 0):
        p_f = torch.sum(1 / h_opt.log_prob(x_adaptive).exp())/N_2
    else:
        p_f = torch.tensor(0.).to(device)
    return p_f,x_adaptive


'''-------------------------Train-------------------------'''
def generate_data(num_omega, num_b):
    xo = get_omega_points(num_omega, x_left=-0.2, y_left=-0.2, z_left=-0.2,
                        x_right=0.2, y_right=0.2, z_right=0.2)
    xb = get_boundary_points(num_b, x_left=-0.2, y_left=-0.2, z_left=-0.2,
                        x_right=0.2, y_right=0.2, z_right=0.2)
    np.save('xo',xo)
    np.save('xb',xb)
    fo = f_x(xo)
    # 后100个点使用 u_x 计算
    fb = u_x(xb)
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
n_depth = 3  # only used in deep NN
mu_div = 3.
mu_mul = 2.
#parameters for getting new samples
tolerance_r = 0.4#0.1
tolerance_p = 0.1 #0.1
N_1 = 3000
N_2 = 10000
p0 =0.1
adaptive_iter_max = 10
# number of training and test data points
c_addpt = 1.
num_omega = 4000
num_b = 400 #注意此处设置的为每个边取点数量
#num_if = 80

x1 = torch.linspace(-0.2, 0.2, 30)
x2 = torch.linspace(-0.2, 0.2, 30)
x3 = torch.linspace(-0.2, 0.2, 30)
X1, X2, X3 = torch.meshgrid(x1, x2, x3, indexing='ij') # 建议加上 indexing='ij' 明确索引方式

# 将网格点展平并拼接成 (N, 3) 的形状
Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None], X3.flatten()[:, None]), dim=1)
# 2. 定义筛选条件并创建掩码
# 我们要保留的是：x1不在[0,1] 或者 x2不在[0,1] 的点
# 等价于：不满足 (x1在[0,1] 并且 x2在[0,1]) 的点
mask = ~((Z[:, 0] >= 0) & (Z[:, 0] <= 1) & (Z[:, 1] >= 0) & (Z[:, 1] <= 1))
# 3. 应用掩码，筛选出有效点
Z = Z[mask].to(device)

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
    func_params = torch.load('best_model.mdl', map_location=device)
    model.load_state_dict(func_params)
    # use Pytorch and functorch
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


# plot evolution of loss
N_loss = len(lossval)
lossval = np.array(lossval).reshape(N_loss,1)
epochcol = np.linspace(1, N_loss, N_loss).reshape(N_loss,1)

plt.figure(figsize = (5,5))

plt.semilogy(epochcol, lossval)
plt.title('loss')
plt.xlabel('epoch')
plt.show()

# ============================== 生成z=0均匀点并可视化 ==============================
def generate_z0_uniform_grid(x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), num_points=200):
    """生成z=0平面的均匀网格点（筛选有效区域）"""
    # 生成x/y均匀采样点
    x = torch.linspace(x_range[0], x_range[1], num_points)
    y = torch.linspace(y_range[0], y_range[1], num_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.zeros_like(X)  # z固定为0
    
    # 展平为(N,3)坐标并筛选有效区域（排除[0,0.2]×[0,0.2]）
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    valid_mask = ~((grid_points[:, 0] >= 0) & (grid_points[:, 0] <= 0.2) & 
                   (grid_points[:, 1] >= 0) & (grid_points[:, 1] <= 0.2))
    valid_points = grid_points[valid_mask].to(device)
    
    return valid_points, X, Y, valid_mask

def plot_z0_results(valid_points, pred_u, truth_u, X, Y, valid_mask, num_points=200):
    """绘制z=0平面的真实值/预测值/误差分布图"""
    # 重构全尺寸数组（无效区域填充NaN）
    pred_full = np.full((num_points, num_points), np.nan)
    truth_full = np.full((num_points, num_points), np.nan)
    error_full = np.full((num_points, num_points), np.nan)
    
    # 计算有效点的网格索引
    x_flat = valid_points[:, 0].cpu().detach().numpy()
    y_flat = valid_points[:, 1].cpu().detach().numpy()
    x_idx = np.round((x_flat - (-0.2)) / (0.4 / (num_points - 1))).astype(int)
    y_idx = np.round((y_flat - (-0.2)) / (0.4 / (num_points - 1))).astype(int)
    
    # 填充有效区域数据
    pred_full[x_idx, y_idx] = pred_u.cpu().detach().numpy()
    truth_full[x_idx, y_idx] = truth_u.cpu().detach().numpy()
    error_full[x_idx, y_idx] = np.abs(pred_full[x_idx, y_idx] - truth_full[x_idx, y_idx])
    
    # 创建子图
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    """
    # 1. 真实值热力图
    im1 = axes[0].imshow(truth_full.T, extent=[-0.2, 0.2, -0.2, 0.2], 
                         origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title('Truth Value (z=0)', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. 预测值热力图
    im2 = axes[1].imshow(pred_full.T, extent=[-0.2, 0.2, -0.2, 0.2], 
                         origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title('Predicted Value (z=0)', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    """
    # 3. 绝对误差热力图
    im3 = axes.imshow(error_full.T, extent=[-0.2, 0.2, -0.2, 0.2], 
                         origin='lower', cmap='Reds', aspect='auto', vmin=0)
    axes.set_xlabel('x')
    axes.set_ylabel('y')  
    plt.colorbar(im3, ax=axes)
    
    plt.tight_layout()
    plt.show()


# ============================== 执行可视化 ==============================
# 生成z=0均匀点
z0_points, X_grid, Y_grid, valid_mask = generate_z0_uniform_grid(num_points=200)

# 计算真实值和预测值0
truth_u =  u_x(z0_points).squeeze()
pred_u = func_model(func_params, z0_points).squeeze()

# 绘制结果图
plot_z0_results(z0_points, pred_u, truth_u, X_grid, Y_grid, valid_mask, num_points=200)

# 计算z=0平面的相对L2误差
z0_l2 = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
    torch.mean(torch.pow(truth_u, 2))
)
print(f"\nz=0平面相对L2误差：{z0_l2.item():.6f}")

