import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools

# File is in current working directory
from rbf_layer import RBFLayer, l_norm, rbf_gaussian
import rbf4derivative as rbf4python

from functorch import make_functional, vmap, grad, jacrev, hessian

from collections import namedtuple, OrderedDict
import datetime
import time


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
tr_iter_max    = 5000                     # max. iteration
ls_check       = 100
ls_check0      = ls_check - 1
# tolerence for LM
tol_main    = 10**(-12)
tol_machine = 10**(-15)
mu_max      = 10**8
'''-------------------------Data generator-------------------------'''
# Ω:[0,1]×[0,1]x[0,1]上随机取点
def get_omega_points(num,x_left=0.0,y_left=0.0,z_left=0.0,x_right=1.0,y_right=1.0,z_right=1.0):
    x1 = torch.rand(num, 1)*(x_right-x_left) + x_left
    x2 = torch.rand(num, 1)*(y_right-y_left) + y_left
    x3 = torch.rand(num, 1)*(z_right-z_left) + z_left
    x = torch.cat((x1, x2, x3), dim=1)
    return x

# 边界上取点
# 边界上取点
def get_boundary_points(num,x_left=0.0,y_left=0.0,z_left=0.0,x_right=1.0,y_right=1.0,z_right=1.0):
    index1 = torch.rand(num, 1)*(x_right-x_left) + x_left
    index2 = torch.rand(num, 1)*(y_right-y_left) + y_left
    index3 = torch.rand(num, 1)*(z_right-z_left) + z_left
    xb1 = torch.cat((index1, index2, torch.ones_like(index3)*z_left), dim=1)
    xb2 = torch.cat((index1, index2, torch.ones_like(index3)*z_right), dim=1)
    xb3 = torch.cat((index1, torch.ones_like(index2)*y_left, index3), dim=1)
    xb4 = torch.cat((index1, torch.ones_like(index2)*y_right, index3), dim=1)
    xb5 = torch.cat((torch.ones_like(index1)*x_left, index2, index3), dim=1)
    xb6 = torch.cat((torch.ones_like(index1)*x_right, index2, index3), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)
    fb1 = u_x(xb1)
    fb2 = u_x(xb2)
    fb3 = u_x(xb3)
    fb4 = u_x(xb4)
    fb5 = func_model_nn(func_params_nn, xb5.to(device)).cpu()
    fb6 = u_x(xb6)
    fb = torch.cat((fb1, fb2, fb3, fb4, fb5, fb6), dim=0)
    return xb,fb
'''-------------------------Define functions-------------------------'''
Re_old = 150
Re = 200
print ('Re=',Re)
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
        fx = func_model_nn(func_params, x)
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
    loss_o = -(u_xx + u_yy + u_zz) + Re * u_x - f_o
    return loss_o

def func_loss_b(func_params, x_b,f_b):
    def f(x, func_params):
        fx = func_model_nn(func_params, x)
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

# Initialization of LM method
def train_PINNs_LM(func_params, func_model, LM_setup, tr_input, lossval, lossval_dbg, truth_u_local, Z_local):
    best_l2, best_epoch = 1000, 0
    # assign tuple elements of LM_set_up
    p_vec_o, dp_o, L_o, J_o, mu, criterion = LM_setup # old参数导入
    I_pvec = torch.eye(len(p_vec_o)).to(device) # 单位阵

    # assign tuple elements of data_input
    [X_o, F_o, X_b, F_b, NL, NL_sqrt] = tr_input #训练参数

    # iteration counts and check
    Comput_old = True
    step = 0

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
#                 per_sample_grads = vmap(jacrev(func_loss_o), (None, 0, 0))(func_params, X_o, F_o)
#                 cnt = 0
#                 for g in per_sample_grads:
#                     g = g.detach()
#                     J_o = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_o, g.reshape(len(g), -1)])
#                     cnt = 1
                # Fortran code to replace the usage of vmap function
                grad_weight,grad_shape,grad_center = rbf4python.grad4params(X_o.cpu().detach().numpy(),func_params[0].cpu().detach().numpy(),\
                                              torch.exp(func_params[2]).cpu().detach().numpy(),func_params[1].cpu().detach().numpy(),\
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
                
                grad_weight,grad_shape,grad_center = rbf4python.grad4params_bd(X_b.cpu().detach().numpy(),func_params[0].cpu().detach().numpy(),\
                                              torch.exp(func_params[2]).cpu().detach().numpy(),func_params[1].cpu().detach().numpy(),center_flag)
                
                J_b = torch.tensor(grad_weight).reshape(len(grad_weight),-1)
                if center_flag != 0:
                    J_b = torch.hstack([J_b, torch.tensor(grad_center).reshape(len(grad_center), -1)])
                J_b = torch.hstack([J_b, torch.tensor(grad_shape).reshape(len(grad_shape), -1)])
                J_b = J_b*alpha_ib
              
                J = torch.cat((J_o / NL_sqrt[1], J_b / NL_sqrt[2])).detach()
                # 组装好了J矩阵
                ### info. normal equation of J
                J_product = J.t() @ J
                rhs = - J.to(device).t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product.to(device) + mu * I_pvec, rhs)
                
                cnt = 0
                for p in func_params:
                    if p.requires_grad:
                       mm = torch.Tensor([p.shape]).tolist()[0]
                       num = int(functools.reduce(lambda x, y: x * y, mm, 1))
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
                pred_u = func_model(func_params, Z_local)
                # relative l2 loss
                l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u_local, 2))) / torch.sqrt(
                    torch.mean(torch.pow(truth_u_local, 2)))
                # absolute loss
                abs_loss = torch.mean(torch.abs(pred_u - truth_u_local))
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
def getlocal4rbf(func_params,epsilon, N_size, box_lower, box_upper):
    def f(x, func_params):
        fx = func_model_nn(func_params, x)
        return fx.squeeze(0).squeeze(0)
        
#     fp = f_x(Z)
#     fp = fp.to(device)
#     res = torch.abs(vmap((func_loss_o), (None, 0, 0))(func_params, Z, fp)).flatten().detach().numpy()
    h_x = (box_upper[0] -  box_lower[0])/N_size
    h_y = (box_upper[1] -  box_lower[1])/N_size
    h_z = (box_upper[2] -  box_lower[2])/N_size

    ele_index = np.zeros([N_size,N_size,N_size],dtype='int')
    
    for i in range(N_size):
      for j in range(N_size):
        for k in range(N_size):
          x_left = box_lower[0] + k*h_x
          y_left = box_lower[1] + j*h_y
          z_left = box_lower[2] + i*h_z
          test_point = get_omega_points(50,x_left,y_left,z_left,x_left+h_x,y_left+h_y,z_left+h_z)
          t_temp = f_x(test_point)
          t_temp = t_temp.to(device)
          res_temp = torch.abs(vmap((func_loss_o), (None, 0, 0))(func_params, test_point, t_temp)).flatten().detach().numpy()
#           index_local = np.arange(len(Z))[(Z[:,0] >= x_left) & (Z[:,0] <= (x_left+h_x)) &
#                                           (Z[:,1] >= y_left) & (Z[:,1] <= (y_left+h_y)) &
#                                           (Z[:,2] >= z_left) & (Z[:,2] <= (z_left+h_z)) ]
#           res_local = res[index_local]
          ratio = res_temp[res_temp>epsilon].size/50
          if ratio > eps_thresh:
             ele_index[k,j,i]=1
    
    return ele_index, h_x, h_y, h_z
'''-------------------------Train-------------------------'''
# Network size
n_input = 3
n_hidden = 30
n_output = 1
n_depth = 3 
mu_div = 3.
mu_mul = 2.
# Network size of NN with RBF
n_input_rbf = 3
n_hidden_rbf = 80
n_output_rbf = 1
# number of training and test data points
num_omega = 500
num_b = 50
alpha_ib = 1.0
convection_coeff = np.array([Re,0,0])
# parameters for control how to add rbf locally
epsilon = 0.05
N_size = 30
eps_thresh = 0.05
# lower left point and upper right point of the computational domain
box_lower = np.array([0.0,0.0,0.0])
box_upper = np.array([1.0,1.0,1.0])
#parameters for getting new samples
x1 = torch.linspace(0, 1, 30)
x2 = torch.linspace(0, 1, 30)
x3 = torch.linspace(0, 1, 30)
X1, X2, X3 = torch.meshgrid(x1, x2, x3)
Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None],X3.flatten()[:, None]),dim=1)
Z = Z.to(device)

# NN structure
if n_depth == 1:  # Shallow NN
	model_nn = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
else:  # Deep NN
	model_nn = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)
func_model_nn, func_params_nn = make_functional(model_nn)  # 获取model及其参数
func_params_nn = torch.load('best_model.mdl',map_location=torch.device(device))
try:
  ele_index  = np.loadtxt('ele_index.txt')
  tmp_int = int(np.rint(np.power(ele_index.size,1/3)))
  ele_index = ele_index.reshape(tmp_int,tmp_int,tmp_int)
  h_x = (box_upper[0] -  box_lower[0])/N_size
  h_y = (box_upper[1] -  box_lower[1])/N_size
  h_z = (box_upper[2] -  box_lower[2])/N_size
except:
  ele_index, h_x, h_y, h_z = getlocal4rbf(func_params_nn,epsilon, N_size, box_lower, box_upper)
  np.savetxt('ele_index.txt',ele_index.reshape(-1))
# residue calculations
pred_u_nn = func_model_nn(func_params_nn, Z)
truth_u = u_x(Z)
# relative l2 loss
l2_loss_nn = torch.sqrt(torch.mean(torch.pow(pred_u_nn - truth_u, 2))) / torch.sqrt(torch.mean(torch.pow(truth_u, 2)))
print ('For NN without using RBF, the relative error is', l2_loss_nn.item())

# determine the location where the large residue is obtained
# combine the locations if they are connected to possibly reduce the number of NNs
indices = np.where(ele_index == 1)
x_left = box_lower[0] + indices[0].min()*h_x
x_right = box_lower[0] + (indices[0].max()+1)*h_x
y_left = box_lower[1] + indices[1].min()*h_y
y_right = box_lower[1] + (indices[1].max()+1)*h_y
z_left = box_lower[2] + indices[2].min()*h_z
z_right = box_lower[2] + (indices[2].max()+1)*h_z
left_point_local = torch.tensor([x_left,y_left,z_left])
right_point_local = torch.tensor([x_right,y_right,z_right])
print ('local left point:',left_point_local)
print ('local right point:',right_point_local)
Z_local = Z[(Z[:,0] >= x_left) & (Z[:,0] <= (x_right)) & 
            (Z[:,1] >= y_left) & (Z[:,1] <= (y_right)) & 
            (Z[:,2] >= z_left) & (Z[:,2] <= (z_right))]
#parameters for getting new samples
# x1_local = torch.linspace(x_left, x_right, 40)
# x2_local = torch.linspace(y_left, y_right, 40)
# x3_local = torch.linspace(z_left, z_right, 40)
# X1_local, X2_local, X3_local = torch.meshgrid(x1_local, x2_local, x3_local)
# Z_local = torch.cat((X1_local.flatten()[:, None], X2_local.flatten()[:, None],X3_local.flatten()[:, None]),dim=1)
Z_local = Z_local.to(device)
truth_u_local = u_x(Z_local).to(device)

def generate_data_local(num_omega, num_b):
    xo = get_omega_points(num_omega,x_left,y_left,z_left,x_right,y_right,z_right)
    xb,fb = get_boundary_points(num_b,x_left,y_left,z_left,x_right,y_right,z_right)
    fo = f_x(xo)
#     fb = func_model_nn(func_params_nn, xb)
    # here we use the exact boundary condition as the interior side have very close
    # approximation while others are just the domain boundary.
#     fb = u_x(xb)
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
# rbf NN structure
try:
  initial_weights = torch.tensor(np.loadtxt('weights_params_Re%g.txt'%Re_old)).reshape(1,-1)
  initial_shape = torch.tensor(np.loadtxt('shape_params_Re%g.txt'%Re_old))
  center_init = torch.tensor(np.loadtxt('center_params_Re%g.txt'%Re_old))
  print ('load initial guess with Re=',Re_old)
  model = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   constant_weights_parameters = False,\
                   constant_shape_parameter = False,\
                   initial_centers_parameter = center_init,\
                   initial_shape_parameter = initial_shape,\
                   initial_weights_parameters = initial_weights,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
                   normalization=False).double().to(device)
except:
  center_init = torch.zeros([n_hidden_rbf,3])
  center_init[:,0] = torch.rand(n_hidden_rbf)*(x_right-x_left)+x_left
  center_init[:,1] = torch.rand(n_hidden_rbf)*(y_right-y_left)+y_left
  center_init[:,2] = torch.rand(n_hidden_rbf)*(z_right-z_left)+z_left
  print ('Uniform distribution for initial guess...')
  model = RBFLayer(in_features_dim=n_input_rbf,\
                   num_kernels=n_hidden_rbf,\
                   out_features_dim=n_output_rbf,\
                   constant_centers_parameter = False,\
                   initial_centers_parameter = center_init,\
                   radial_function=rbf_gaussian,\
                   norm_function=l_norm,\
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
# lossval, lossval_dbg, relerr_loss_char, Flag_LM = train_PINNs_LM(func_params,func_model, LM_setup, tr_input, lossval, lossval_dbg,truth_u_local, Z_local)
lossval, lossval_dbg, relerr_loss_char = train_PINNs_LM(func_params,func_model, LM_setup, tr_input, lossval, lossval_dbg,truth_u_local, Z_local)
relerr_loss.append(relerr_loss_char)
end_start = time.time()
total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
print(f"total time : {total_T}")
print('ok')


Z_local_outer = Z[(Z[:,0] < x_left) | (Z[:,0] > (x_right)) | 
                  (Z[:,1] < y_left) | (Z[:,1] > (y_right)) |
                  (Z[:,2] < z_left) | (Z[:,2] > (z_right)) ]
Z_local_outer = Z_local_outer.to(device)
truth_u_local_outer = u_x(Z_local_outer)
pred_outer = func_model_nn(func_params_nn, Z_local_outer)
pred_u = func_model(func_params, Z_local)
l2_loss = torch.pow(pred_u - truth_u_local, 2).sum()
l2_loss += torch.pow(pred_outer - truth_u_local_outer, 2).sum()
l2_loss_total = torch.pow(truth_u_local_outer, 2).sum()
l2_loss_total += torch.pow(truth_u_local, 2).sum()
l2_loss = torch.sqrt(l2_loss) / torch.sqrt(l2_loss_total)
print ('For NN, the relative error is', l2_loss.item())

np.savetxt('weights_params_Re%g.txt'%Re,func_params[0].cpu().detach().numpy())
np.savetxt('shape_params_Re%g.txt'%Re,func_params[2].cpu().detach().numpy())
np.savetxt('center_params_Re%g.txt'%Re,func_params[1].cpu().detach().numpy())


# plot evolution of loss
N_loss = len(lossval)
lossval = np.array(lossval).reshape(N_loss,1)
epochcol = np.linspace(1, N_loss, N_loss).reshape(N_loss,1)

plt.figure(figsize = (5,5))

plt.semilogy(epochcol, lossval)
plt.title('loss')
plt.xlabel('epoch')
plt.show()


