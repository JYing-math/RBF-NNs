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


'''-------------------------Functions-------------------------'''

Re = 50

def u_x(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    u = ((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
    u = torch.unsqueeze(u, 1)
    return u

def f_x(x):
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
    f_o = f_o[0]
    du = jacrev(f)(x_o, func_params)
    d2u = jacrev(jacrev(f))(x_o, func_params)
    u_x = du[0]
    u_xx = d2u[0][0]
    u_yy = d2u[1][1]
    u_zz = d2u[2][2]
    loss_o = -(u_xx + u_yy + u_zz) + Re * u_x - f_o
    return loss_o

def func_loss_b(func_params, x_b,f_b):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    f_b = f_b[0]
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return 100*loss_b


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

def train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg, tolerance_r):
    best_l2, best_epoch = 1000, 0
    # assign tuple elements of LM_set_up
    p_vec_o, dp_o, L_o, J_o, mu, criterion = LM_setup # old参数导入
    I_pvec = torch.eye(len(p_vec_o)).to(device) # 单位阵

    # assign tuple elements of data_input
    [X_o, F_o, X_b, F_b, NL, NL_sqrt] = tr_input #训练参数

    p_f, data_new = SAIS(func_params,N_1, N_2, p0, adaptive_iter_max, tolerance_r)
    if (p_f < tolerance_p):
        print('p_f < tolerance_p',p_f)
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
                truth_u = u_x(Z).reshape(-1)
                # relative l2 loss
                l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
                    torch.mean(torch.pow(truth_u, 2)))
                # absolute loss
                abs_loss = torch.mean(torch.abs(pred_u - truth_u))
            if l2_loss < best_l2:
                best_l2 = l2_loss.item()
                best_epoch = step
                torch.save(func_params, 'best_model-2-sais.mdl')

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
def lsf(func_params, x, tolerance_r):
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
    # 初始化一个空列表来存储有效点
    valid_points = []
    
    # 循环直到收集到足够的有效点
    while len(valid_points) < num:
        # 从高斯分布中采样一个点
        sample = h.sample()  # 形状: (dim,)
        
        # 检查这个点的所有坐标是否都在 [0, 1] 范围内
        # .all() 会检查张量中所有元素是否都为 True
        is_valid = (sample >= 0).all() and (sample <= 1).all()
        
        if is_valid:
            valid_points.append(sample)
    
    # 将列表中的所有有效点堆叠成一个张量
    # 并移动到指定设备，设置数据类型为 float64，开启梯度计算
    result = torch.stack(valid_points).to(device, dtype=torch.float64).requires_grad_(True)
    
    return result
#自采样
def SAIS(params,N_1,N_2,p0,adaptive_iter_max, tolerance_r):
    k = 1
    while(k<adaptive_iter_max):
        if(k == 1):
            x = get_omega_points(N_1)
        else:
            x = get_gaussian_samples(h,N_1)
        x = torch.tensor(x, requires_grad=True).double().to(device)
        lsf_x = lsf(params,x, tolerance_r)
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
    x_n2 = torch.tensor(x_n2, requires_grad=True).double().to(device)
    x_adaptive = x_n2[lsf(params,x_n2, tolerance_r) > 0]
    #pf中的indicator function是lsf即gx>0则为1 也就是只计X adaptive中的
    if(len(x_adaptive) > 0):
        p_f = torch.sum(1 / h_opt.log_prob(x_adaptive).exp())/N_2
    else:
        p_f = torch.tensor(0.).to(device)
    return p_f,x_adaptive


'''-------------------------Train-------------------------'''
def generate_data(num_omega, num_b):
    xo = get_omega_points(num_omega)
    xb = get_boundary_points(num_b)
    fo = f_x(xo)
    fb = u_x(xb)
    len_sum = num_omega + 6 * num_b
    NL = [len_sum, num_omega, 6 * num_b]
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
n_hidden = 30
n_output = 1
n_depth = 3  # only used in deep NN
mu_div = 3.
mu_mul = 2.
#parameters for getting new samples
tolerance_r = 0.005#0.1
tolerance_p = 0.1 #0.1
N_1 = 3000
N_2 = 10000
p0 =0.1
adaptive_iter_max = 10
# number of training and test data points
c_addpt = 1.
num_omega = 500
num_b = 50 #注意此处设置的为每个边取点数量
#num_if = 80

x1 = torch.linspace(0, 1, 30)
x2 = torch.linspace(0, 1, 30)
x3 = torch.linspace(0, 1, 30)
X1, X2,X3 = torch.meshgrid(x1, x2,x3)
Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None],X3.flatten()[:, None]),dim=1)
Z = Z.to(device)

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
    func_model, func_params = make_functional(model)  # 获取model及其参数
    loaded_params = torch.load('best_model.mdl', map_location=device)  # map_location直接传device更简洁
    # 确保加载的参数是元组，且每个参数张量都在目标设备上
    func_params = tuple(p.to(device).double() for p in loaded_params)

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
    lossval, lossval_dbg, relerr_loss_char = train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg, tolerance_r)
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


