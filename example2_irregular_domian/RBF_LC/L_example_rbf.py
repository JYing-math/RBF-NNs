import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import functools
import copy
from scipy.interpolate import griddata
import torch.autograd as autograd
# File is in current working directory
from rbf_layer import RBFLayer, l_norm, rbf_gaussian
import rbf4derivative
from functorch import make_functional, vmap, grad, jacrev, hessian
import matplotlib.tri as tri
from collections import namedtuple, OrderedDict
import datetime
import time
import sys
import warnings
from scipy.stats.qmc import LatinHypercube
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
tr_iter_max    = 1000                      # max. iteration
ls_check       = 100
ls_check0      = ls_check - 1
# tolerence for LM
tol_main    = 10**(-12)
tol_machine = 10**(-15)
mu_max      = 10**8
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
def func_loss_o_nn(x_op: torch.Tensor, f_omega: torch.Tensor) -> torch.Tensor:
    output_op = model_nn(x_op)
    f_omega = f_omega.reshape(-1, 1)
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
    loss_op = (u_xx + u_yy + u_zz) + f_omega
    return loss_op

def func_loss_o_new(func_params,func_model, x_o, f_o):
    u = func_model(func_params, x_o)
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

def func_loss_b(func_params, x_b,f_b):
    def f(x, func_params):
        fx = model_nn(x)
        return fx.squeeze(0).squeeze(0)
    f_b = f_b[0]
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return 100*loss_b

def func_loss_b_new(func_params, func_model,x_b, f_b):
    u = func_model(func_params, x_b)
    loss_b = u - f_b
    loss_b = loss_b.reshape(-1)
    return alpha_ib*loss_b
'''-------------------------Levenberg-Marquardt (LM) optimizer-------------------------'''
# parameters counter
def count_parameters(func_params):
    return sum(p.numel() for p in func_params if p.requires_grad)

# get the model's parameter
def get_p_vec(func_params):
    p_vec = []
    cnt = 0
    for p in func_params:
        if p.requires_grad:
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

def train_PINNs_LM(func_params,func_model, LM_setup, tr_input, lossval, lossval_dbg):
    best_l2, best_epoch = 1000, 0
    # assign tuple elements of LM_set_up
    p_vec_o, dp_o, L_o, J_o, mu, criterion = LM_setup # old参数导入
    I_pvec = torch.eye(len(p_vec_o)) # 单位阵

    # assign tuple elements of data_input
    [X_o, F_o, X_b, F_b, NL, NL_sqrt] = tr_input #训练参数

    # iteration counts and check
    Comput_old = True
    step = 0
    Flag_LM = 0

    # try-except statement to avoid jam in the code
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):
#         while (step <= tr_iter_max):
            torch.cuda.empty_cache()
            ############################################################
            # LM_optimizer
            if (Comput_old == True):  # need to compute loss_old and J_old

                ### computation of loss 计算各部分损失函数
                Lo = func_loss_o_new(func_params, func_model, X_o, F_o).flatten().detach()
                Lb = func_loss_b_new(func_params, func_model, X_b, F_b).flatten().detach()
                L = torch.cat((Lo / NL_sqrt[1], Lb / NL_sqrt[2]))
                L = L.reshape(NL[0], 1).detach().cpu()
                lsd_sum = torch.sum(Lo * Lo) / NL[1]
                lsb_sum = torch.sum(Lb * Lb) / NL[2]
                loss_dbg_old = [lsd_sum.item(), lsb_sum.item()]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]

            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
#                 per_sample_grads = vmap(jacrev(func_loss_o), (None, 0, 0))(func_params, X_o, F_o)
#                 cnt = 0
#                 for g in per_sample_grads:
#                     g = g.detach()
#                     J_o = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_o, g.reshape(len(g), -1)])
#                     cnt = 1
                # Fortran code to replace the usage of vmap function
                grad_weight,grad_shape,grad_center = rbf4derivative.grad4params(X_o.detach().cpu().numpy(),func_params[0].cpu().detach().numpy(),\
                                              torch.exp(func_params[2]).detach().cpu().numpy(),func_params[1].detach().cpu().numpy(),\
                                              convection_coeff,center_flag)
                J_o = torch.tensor(grad_weight).reshape(len(grad_weight),-1)
                if center_flag != 0:
                  J_o = torch.hstack([J_o, torch.tensor(grad_center).reshape(len(grad_center), -1)])
                J_o = torch.hstack([J_o, torch.tensor(grad_shape).reshape(len(grad_shape), -1)])
                
#                 per_sample_grads = jacrev(func_loss_b_new,argnums=0)(func_params, X_b, F_b)
#                 cnt = 0
#                 for g in per_sample_grads:
#                     g = g.detach()
#                     J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
#                     cnt = 1
                
                grad_weight,grad_shape,grad_center = rbf4derivative.grad4params_bd(X_b.detach().cpu().numpy(),func_params[0].detach().cpu().numpy(),\
                                              torch.exp(func_params[2]).detach().cpu().numpy(),func_params[1].detach().cpu().numpy(),center_flag)
                
                J_b = torch.tensor(grad_weight).reshape(len(grad_weight),-1)
                if center_flag != 0:
                    J_b = torch.hstack([J_b, torch.tensor(grad_center).reshape(len(grad_center), -1)])
                J_b = torch.hstack([J_b, torch.tensor(grad_shape).reshape(len(grad_shape), -1)])
                J_b = J_b*alpha_ib
              
                J = torch.cat((J_o / NL_sqrt[1], J_b / NL_sqrt[2])).detach()
                # 组装好了J矩阵
                ### info. normal equation of J
                J_product = J.t() @ J
                rhs = - J.t() @ L.cpu()

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs).to(device)
                
                cnt = torch.tensor(0, dtype=torch.int64)
                cnt = cnt.to(device)
                for p in func_params:
                    if p.requires_grad:
                       mm = torch.Tensor([p.shape]).tolist()[0]
                       num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                       num = torch.tensor(num, device=device)
                       p += dp[cnt:cnt + num].reshape(p.shape)
                       cnt += num

            ### Compute loss_new
            Lo = func_loss_o_new(func_params, func_model, X_o, F_o).flatten().detach()
            Lb = func_loss_b_new(func_params, func_model, X_b, F_b).flatten().detach()
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
                                if p.requires_grad:
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
                mask = (Z_filtered[:,0] < x_left) | (Z_filtered[:,0] > (x_right)) | (Z_filtered[:,1] < y_left) | (Z_filtered[:,1] > (y_right)) | (Z_filtered[:,2] < z_left) | (Z_filtered[:,2] > (z_right))
                Z_local_outer = Z_filtered[mask]
                Z_local_outer = Z_local_outer.to(device)
                Z_local = Z_filtered[~mask]
                truth_u_local_outer = u_x(Z_local_outer)
                pred_outer = model_nn(Z_local_outer)
                pred_u = func_model(func_params, Z_local)
                truth_u_local = u_x(Z_local)
                abs_loss_local = torch.max(torch.abs(pred_u - truth_u_local))
                abs_loss_outer = torch.max(torch.abs(pred_outer - truth_u_local_outer))
                abs_loss = max(abs_loss_local.item(), abs_loss_outer.item())
                l2_loss = torch.pow(pred_u - truth_u_local, 2).sum()
                l2_loss += torch.pow(pred_outer - truth_u_local_outer, 2).sum()
                l2_loss_total = torch.pow(truth_u_local_outer, 2).sum()
                l2_loss_total += torch.pow(truth_u_local, 2).sum()
                l2_loss = torch.sqrt(l2_loss) / torch.sqrt(l2_loss_total)
            if l2_loss<best_l2:
                best_l2 = l2_loss.item()
                best_epoch = step
#                 torch.save(func_params, 'best_model.mdl')

            if step % ls_check == ls_check0:
                print('epoch:', step,
                      "loss:", '%.4e'% lossval[-1],
                      "l2_loss:", '%.4e' % l2_loss,
                      'abs_loss', '%.4e' % abs_loss,
                      'best_epoch', best_epoch,
                      'best_l2', '%.4e' % best_l2)
            step += 1

        
        print('epoch:', step,
                      "loss:", '%.4e'% lossval[-1],
                      "l2_loss:", '%.4e' % l2_loss,
                      'abs_loss', '%.4e' % abs_loss,
                      'best_epoch', best_epoch,
                      'best_l2', '%.4e' % best_l2)
#         print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        if step == tr_iter_max+1 and lossval[-1] >tol_main:
           Flag_LM = 1
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss,Flag_LM

    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss, Flag_LM
'''-------------------------generate new samples-------------------------'''
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
'''-------------------------Train-------------------------'''
# Network size of usual NN, must be the same as before
n_input = 3
n_hidden = 20
n_output = 1
n_depth = 3 
mu_div = 3.
mu_mul = 2.
# Network size of NN with RBF
n_input_rbf = 3
n_hidden_rbf = 40
n_output_rbf = 1
# number of training and test data points
num_omega = 1000
num_b = 100
alpha_ib = 100
convection_coeff = np.zeros(3)
# parameters for control how to add rbf locally
epsilon = 0.4
N_size = 20
eps_thresh = 0.05
# lower left point and upper right point of the computational domain
box_lower = np.array([-0.2, -0.2, -0.2])
box_upper = np.array([0.2, 0.2, 0.2])

x1 = torch.linspace(-0.2, 0.2, 40)
x2 = torch.linspace(-0.2, 0.2, 40)
x3 = torch.linspace(-0.2, 0.2, 40)
X1, X2, X3 = torch.meshgrid(x1, x2, x3, indexing='ij') # 建议加上 indexing='ij' 明确索引方式

# 将网格点展平并拼接成 (N, 3) 的形状
Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None], X3.flatten()[:, None]), dim=1)
# 2. 定义筛选条件并创建掩码
# 我们要保留的是：x1不在[0,1] 或者 x2不在[0,1] 的点
# 等价于：不满足 (x1在[0,1] 并且 x2在[0,1]) 的点
mask = ~((Z[:, 0] >= 0) & (Z[:, 0] <= 1) & (Z[:, 1] >= 0) & (Z[:, 1] <= 1))
# 3. 应用掩码，筛选出有效点
test_points = Z[mask].to(device).requires_grad_(True)

x1_small = torch.linspace(-0.2, 0.2, 30)
x2_small = torch.linspace(-0.2, 0.2, 30)
x3_small = torch.linspace(-0.2, 0.2, 30)
X1_small, X2_small, X3_small = torch.meshgrid(x1_small, x2_small, x3_small, indexing='ij') # 建议加上 indexing='ij' 明确索引方式

# 将网格点展平并拼接成 (N, 3) 的形状
Z_small = torch.cat((X1_small.flatten()[:, None], X2_small.flatten()[:, None], X3_small.flatten()[:, None]), dim=1)
# 2. 定义筛选条件并创建掩码
# 我们要保留的是：x1不在[0,1] 或者 x2不在[0,1] 的点
# 等价于：不满足 (x1在[0,1] 并且 x2在[0,1]) 的点
mask_small = ~((Z_small[:, 0] >= 0) & (Z_small[:, 0] <= 1) & (Z_small[:, 1] >= 0) & (Z_small[:, 1] <= 1))
# 3. 应用掩码，筛选出有效点
Z_filtered = Z_small[mask_small].to(device)

# NN structure
if n_depth == 1:  # Shallow NN
	model_nn = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
else:  # Deep NN
	model_nn = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)
model_nn = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)
func_params_nn = torch.load('best_model.mdl', map_location=device)
model_nn.load_state_dict(func_params_nn)   
ele_index, h_x, h_y, h_z, count_num_le = getlocal4rbf(epsilon, N_size, box_lower, box_upper, eps_thresh)
# residue calculations
pred_u_nn = model_nn(Z_filtered)
truth_u = u_x(Z_filtered)
# relative l2 loss
l2_loss_nn = torch.sqrt(torch.mean(torch.pow(pred_u_nn - truth_u, 2))) / torch.sqrt(torch.mean(torch.pow(truth_u, 2)))
print ('For NN without using RBF, the relative error is', l2_loss_nn.item())

# determine the location where the large residue is obtained
# combine the locations if they are connected to possibly reduce the number of NNs
list_matrix = []
matrix_tmp = ele_index.copy()
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
x_left = box_lower[0] + (list_matrix[0][2])*h_x
y_left = box_lower[1] + (list_matrix[0][0])*h_y
x_right = box_lower[0] +(list_matrix[0][3])*h_x
y_right = box_lower[1] +(list_matrix[0][1])*h_y
z_left = box_lower[2] + (list_matrix[0][4])*(h_z)
z_right = box_lower[2] +(list_matrix[0][5])*(h_z)


print(x_left,y_left,z_left,x_right,y_right,z_right)
def generate_data_local(num_omega, num_b):
    xo = get_omega_points(num_omega,x_left,y_left,z_left,x_right,y_right,z_right)
    xb = get_boundary_points(num_b,x_left,y_left,z_left,x_right,y_right,z_right).to(device)
    fo = f_x(xo)
    fb_model = model_nn(xb[:75])
    # 后100个点使用 u_x 计算
    fb_u_x = u_x(xb[75:])
    # 将两部分结果拼接起来，得到最终的 fb
    fb = torch.cat([fb_model, fb_u_x], dim=0)
    len_sum = len(xo) + len(xb)
    NL = [len_sum, len(xo), len(xb)]
    NL_sqrt = np.sqrt(NL)

    xo_tr = torch.tensor(xo, requires_grad=True).double().to(device)
    xb_tr = torch.tensor(xb, requires_grad=True).double().to(device)
    fo_tr = torch.tensor(fo).double().to(device)
    fb_tr = torch.tensor(fb).double().to(device)
    return xo_tr, fo_tr, xb_tr, fb_tr, NL, NL_sqrt

# Essential namedtuples in the model
DataInput = namedtuple( "DataInput" , [ "X_o" , "F_o" , "X_b" , "F_b" , "NL" , "NL_sqrt"] )
LM_Setup  = namedtuple( "LM_Setup" , [ 'p_vec_o' , 'dp_o' , 'L_o' , 'J_o' , 'mu0' , 'criterion' ] )



# storages for errors, time instants, and IRK stages
relerr_loss = []
torch.cuda.empty_cache()  # 清理变量
# rbf NN structure

# ------------------- 新增：RBF 中心初始化（确保在 Ω 内）-------------------
def is_in_omega(points):
    in_excavated = (points[:, 0] >= 0) & (points[:, 0] <= 1) & \
                   (points[:, 1] >= 0) & (points[:, 1] <= 1)
    return ~in_excavated

# 1. 定义局部区域边界（限制在 [-1,1]^3 内）
local_bounds = [
    (max(x_left, -1.0), min(x_right, 1.0)),
    (max(y_left, -1.0), min(y_right, 1.0)),
    (max(z_left, -1.0), min(z_right, 1.0))
]

# 2. LHS 生成候选点（2 倍于需要的数量）
num_candidates = n_hidden_rbf * 2
sampler = LatinHypercube(d=3)
sample = sampler.random(n=num_candidates)
for i in range(3):
    sample[:, i] = sample[:, i] * (local_bounds[i][1] - local_bounds[i][0]) + local_bounds[i][0]
candidates = torch.tensor(sample, dtype=torch.float64)

# 3. 过滤无效点
valid_mask = is_in_omega(candidates)
valid_centers = candidates[valid_mask]
if len(valid_centers) < n_hidden_rbf:
    raise ValueError(f"局部区域有效点不足！需要 {n_hidden_rbf} 个，仅找到 {len(valid_centers)} 个。")
center_init = valid_centers[:n_hidden_rbf]

model = RBFLayer(in_features_dim=n_input_rbf,
                 num_kernels=n_hidden_rbf,
                 out_features_dim=n_output_rbf,
                 constant_centers_parameter=False,
                 initial_centers_parameter=center_init,
                 radial_function=rbf_gaussian,
                 norm_function=l_norm,
                 normalization=False).double().to(device)
if model.constant_centers_parameter:
     center_flag = 0
else:
     center_flag = 1
# use Pytorch and functorch
func_model, func_params = make_functional(model)  # 获取model及其参数
xo_tr, fo_tr, xb_tr, fb_tr, NL_tr, NL_sqrt_tr = generate_data_local(num_omega, num_b)
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
lossval, lossval_dbg, relerr_loss_char, Flag_LM = train_PINNs_LM(func_params,func_model, LM_setup, tr_input, lossval, lossval_dbg)
print('Initial training with RBF completed.')
 
relerr_loss.append(relerr_loss_char)
end_start = time.time()
total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
print(f"total time : {total_T}")
print('ok')

# ============================== z=0平面均匀点可视化 ==============================
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

def predict_z0_points(valid_points, func_params, func_model, model_nn, x_left, y_left, x_right, y_right, z_left, z_right):
    """预测z=0平面点的值（局部用RBF，外部用基础NN）"""
    pred_u = torch.zeros(len(valid_points), device=device)
    # 判断点是否在RBF局部区域内
    local_mask = (valid_points[:, 0] >= x_left) & (valid_points[:, 0] <= x_right) & \
                 (valid_points[:, 1] >= y_left) & (valid_points[:, 1] <= y_right) & \
                 (valid_points[:, 2] >= z_left) & (valid_points[:, 2] <= z_right)
    
    # 局部区域用RBF模型预测
    local_points = valid_points[local_mask]
    if len(local_points) > 0:
        pred_local = func_model(func_params, local_points).squeeze()
        pred_u[local_mask] = pred_local
    
    # 外部区域用基础NN模型预测
    outer_points = valid_points[~local_mask]
    if len(outer_points) > 0:
        pred_outer = model_nn(outer_points).squeeze()
        pred_u[~local_mask] = pred_outer
    
    # 计算真实值
    truth_u = u_x(valid_points).squeeze()
    
    return pred_u, truth_u

def plot_z0_results(valid_points, pred_u, truth_u, X, Y, valid_mask, num_points=200):
    """绘制z=0平面的真实值/预测值/误差热力图"""
    # 重构全尺寸数组（无效区域填充NaN）
    pred_full = np.full((num_points, num_points), np.nan)
    truth_full = np.full((num_points, num_points), np.nan)
    error_full = np.full((num_points, num_points), np.nan)
    
    # 计算有效点的网格索引
    x_flat = valid_points[:, 0].cpu().detach().numpy()
    y_flat = valid_points[:, 1].cpu().detach().numpy()
    step = (0.4) / (num_points - 1)  # 步长
    x_idx = np.round((x_flat - (-0.2)) / step).astype(int)
    y_idx = np.round((y_flat - (-0.2)) / step).astype(int)
    
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
    # 标记RBF局部区域
    rect = Rectangle((x_left, y_left), x_right-x_left, y_right-y_left, 
                     linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
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
# 生成z=0平面200×200均匀点
z0_points, X_grid, Y_grid, valid_mask = generate_z0_uniform_grid(num_points=200)

# 预测z=0平面点的值（结合RBF局部模型和基础NN）
pred_u_z0, truth_u_z0 = predict_z0_points(
    z0_points, func_params, func_model, model_nn, 
    x_left, y_left, x_right, y_right, z_left, z_right
)

# 计算z=0平面的整体误差
z0_l2 = torch.sqrt(torch.mean(torch.pow(pred_u_z0 - truth_u_z0, 2))) / torch.sqrt(
    torch.mean(torch.pow(truth_u_z0, 2))
)
print(f"\nz=0平面相对L2误差：{z0_l2.item():.6f}")

# 绘制结果图
plot_z0_results(z0_points, pred_u_z0, truth_u_z0, X_grid, Y_grid, valid_mask, num_points=200)

# ============================== 绘制训练损失曲线 ==============================
N_loss = len(lossval)
lossval = np.array(lossval).reshape(N_loss,1)
epochcol = np.linspace(1, N_loss, N_loss).reshape(N_loss,1)

plt.figure(figsize = (5,5))
plt.semilogy(epochcol, lossval)
plt.title('Training Loss Evolution')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.grid(True, alpha=0.3)
plt.show()

