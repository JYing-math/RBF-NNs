import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import optim, autograd
from matplotlib import pyplot as plt
from matplotlib import cm
import torchvision.models as models
# from mpl_toolkits.axes_grid1 import make_axes_locatable

''''-------------------------Step1：取点--------------------------------------------'''

# Ω上取点：在圆心为（1,1)，半径为2的圆内均匀取点
def get_omega_points(num: int,radius = 2) -> torch.Tensor:
    x1 = 4 * torch.rand(num, 1) - 1
    x2 = 4 * torch.rand(num, 1) - 1
    x = torch.cat((x1, x2), dim=1)
    r = torch.sqrt(torch.sum(torch.pow(x - 1, 2), 1))
    index=torch.where(r < radius+1e-16)[0]
    points = torch.index_select(x, axis=0, index = index)
    return points

# Ω边缘取点：在圆心为（1,1)，半径为2的圆上均匀取点
def get_boundary_points(num: int, radius=2, center_x1=1, center_x2=1) -> torch.Tensor:
    theta = torch.rand((num, 1)) * 2 * torch.pi   # [0, 2phi），随机取n个点
    x1 = torch.cos(theta) * radius + center_x1
    x2 = torch.sin(theta) * radius + center_x2     
    boundary_points = torch.concat((x1, x2), 1)
    return boundary_points

# Γ上取点：在圆心为（1,1)，半径为1的圆上均匀取点
def get_interface_points(num: int, radius=1, center_x1=1, center_x2=1) -> torch.Tensor:
    theta = torch.rand((num, 1)) * 2 * torch.pi
    x1 = torch.cos(theta) * radius + center_x1
    x2 = torch.sin(theta) * radius + center_x2 
    interface_points = torch.concat((x1, x2), 1)
    return interface_points

def get_omega_p_points(x: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(torch.sum(torch.pow(x- 1, 2), 1))
    index = torch.where(r < 1)[0]
    xo1 = x[index]
    return xo1

def get_omega_n_points(x: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(torch.sum(torch.pow(x - 1, 2), 1))
    index = torch.where(r >= 1)[0]
    xo2 = x[index]
    return xo2

# 构造三维数据[x1,x2,-1]
def dimension3_data_subtract1(x: torch.Tensor) -> torch.Tensor:
    x3_1 =  torch.ones(x.shape[0])*(-1)
    x3_1 = torch.unsqueeze(x3_1, 1)
    x1 = torch.concat((x, x3_1), 1)
    return x1

# 构造三维数据[x1,x2,1]
def dimension3_data_add1(x: torch.Tensor) -> torch.Tensor:
    x31 =  torch.ones(x.shape[0], dtype=torch.float32)
    x31 = torch.unsqueeze(x31, 1)
    x2 = torch.concat((x, x31), 1)
    return x2

''''-------------------------Step2：函数定义--------------------------------------------'''

# 获取界面处的法向导数
def get_normal_interface(x: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(torch.sum(torch.pow(x-1, 2), 1))
    n1 = x[:, 0]
    n2 = x[:, 1]
    n1 = (n1-1) / r
    n2 = (n2-1) / r
    n1 = torch.unsqueeze(n1, 1)
    n2 = torch.unsqueeze(n2, 1)
    n = torch.cat((n1, n2), dim=1)
    return n

def a(x: torch.Tensor) -> torch.Tensor:
    x = x - 1
    r2 = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    a_x = torch.where(r2 < 1, 1., 2.)
    return a_x

def f_x(x: torch.Tensor) -> torch.Tensor:
    f_x = 8*torch.tanh(torch.sum(x, 1))-8*torch.tanh(torch.sum(x, 1))**3
    f_x = torch.unsqueeze(f_x, 1)
    return f_x

def phi(x: torch.Tensor) -> torch.Tensor:
    phi_x = -1*torch.tanh(torch.sum(x,1))
    phi_x = torch.unsqueeze(phi_x, 1)
    return phi_x

def g_x(x: torch.Tensor) -> torch.Tensor:
    gx = torch.tanh(torch.sum(x, 1))
    gx = torch.unsqueeze(gx,1)
    return gx

def u_x(x: torch.Tensor) -> torch.Tensor:
    r3 = torch.sqrt(torch.sum(torch.pow(x-1, 2), 1))
    u_x = torch.where(r3 < 1, 2*torch.tanh(torch.sum(x, 1)), torch.tanh(torch.sum(x, 1)))
    u_x = torch.unsqueeze(u_x,1)
    return u_x

'''-------------------------Empty cache and check devices-------------------------'''

torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

''''-------------------------Step3：网络搭建--------------------------------------------'''

class NeuralNet_Shallow(torch.nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output

    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        super(NeuralNet_Shallow, self).__init__()
        self.ln1 = nn.Linear(in_dim, h_dim)
        self.act1 = nn.Sigmoid()
        # self.act1 = nn.Tanh()
        # self.act1 = nn.ReLU()

        self.ln2 = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        return out
    
''''-------------------------Step4：超参数设置--------------------------------------------'''

n_input = 3
mr_list = [100,400,1600,6400]
n_hidden_list = [10,20,40,80]
n_output = 1
# 损失函数的权值设置
learning_rate = 1e-3
# 优化器设置
epochs = 10000  # 10000
epsilon = 1e-16

''''-------------------------Step5：定义损失函数--------------------------------------------'''

# 计算区域Omega内的loss
def func_loss_op(x_op: torch.Tensor, xop_tr: torch.Tensor) -> torch.Tensor:
    f_omega = f_x(x_op)
    output_op = model(xop_tr)
    grad_op = autograd.grad(outputs=output_op, inputs=xop_tr,
                            grad_outputs=torch.ones_like(output_op),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_op[:, 0],1)
    du_dy = torch.unsqueeze(grad_op[:, 1],1)
    grad_op1 = autograd.grad(outputs=du_dx, inputs=xop_tr,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_op2 = autograd.grad(outputs=du_dy, inputs=xop_tr,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_op1[:, 0],1)
    u_yy = torch.unsqueeze(grad_op2[:, 1],1)
    loss_op = (u_xx + u_yy + f_omega)**2
    return torch.mean(loss_op)

def func_loss_on(x_on: torch.Tensor,xon_tr: torch.Tensor) -> torch.Tensor:
    f_omega = f_x(x_on)
    output_on = model(xon_tr)
    grad_on = autograd.grad(outputs=output_on, inputs=xon_tr,
                            grad_outputs=torch.ones_like(output_on),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    du_dx = torch.unsqueeze(grad_on[:, 0],1)
    du_dy = torch.unsqueeze(grad_on[:, 1],1)
    grad_on1 = autograd.grad(outputs=du_dx, inputs=xon_tr,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_on2 = autograd.grad(outputs=du_dy, inputs=xon_tr,
                            grad_outputs=torch.ones_like(du_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_on1[:, 0],1)
    u_yy = torch.unsqueeze(grad_on2[:, 1],1)
    loss_on = (2*u_xx + 2*u_yy + f_omega)**2
    return torch.mean(loss_on)

#计算边界Omega上loss
def func_loss_b(x_b: torch.Tensor,xb1: torch.Tensor) -> torch.Tensor:
    g_omega = g_x(x_b)
    grad_g = autograd.grad(outputs=g_omega, inputs=x_b,
                            grad_outputs=torch.ones_like(g_omega),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    g_omega_x = torch.unsqueeze(grad_g[:, 0],1)
    g_omega_y = torch.unsqueeze(grad_g[:, 1],1)
    grad_g1 = autograd.grad(outputs=g_omega_x, inputs=x_b,
                            grad_outputs=torch.ones_like(g_omega_x),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_g2 = autograd.grad(outputs=g_omega_y, inputs=x_b,
                            grad_outputs=torch.ones_like(g_omega_y),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u = model(xb1)
    grad_b = autograd.grad(outputs=u, inputs=xb1,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_x = torch.unsqueeze(grad_b[:, 0],1)
    u_y = torch.unsqueeze(grad_b[:, 1],1)
    grad_b1 = autograd.grad(outputs=u_x, inputs=xb1,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_b2 = autograd.grad(outputs=u_y, inputs=xb1,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(grad_b1[:, 0],1)
    u_yy = torch.unsqueeze(grad_b2[:, 1],1)
    u_xy = torch.unsqueeze(grad_b1[:, 1],1)
    u_yx = torch.unsqueeze(grad_b2[:, 0],1)
    g_omega_xx =torch.unsqueeze(grad_g1[:, 0],1)
    g_omega_xy =torch.unsqueeze(grad_g1[:, 1],1)
    g_omega_yx =torch.unsqueeze(grad_g2[:, 0],1)
    g_omega_yy =torch.unsqueeze(grad_g2[:, 1],1)
    loss_b = torch.mean((u-g_omega)**2) + torch.mean((u_x-g_omega_x)**2) + torch.mean((u_y-g_omega_y)**2) 
    + torch.mean((u_xx-g_omega_xx)**2) + torch.mean((u_xy-g_omega_xy)**2) + torch.mean((u_yx-g_omega_yx)**2)
    + torch.mean((u_yy-g_omega_yy)**2)
    return loss_b

#计算界面Γ上的loss
def func_loss_if(x_if: torch.Tensor,xif1: torch.Tensor, x_if1: torch.Tensor, nor: torch.Tensor) -> torch.Tensor:
    u1 = model(x_if1)
    u2 = model(xif1)
    exact_phi = phi(x_if)
    grad_u1 = autograd.grad(outputs=u1, inputs=x_if1,
                            grad_outputs=torch.ones_like(u1),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u1_x = torch.unsqueeze(grad_u1[:, 0],1)
    u1_y = torch.unsqueeze(grad_u1[:, 1],1)
    grad_u11 = autograd.grad(outputs=u1_x, inputs=x_if1,
                            grad_outputs=torch.ones_like(u1_x),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_u12 = autograd.grad(outputs=u1_y, inputs=x_if1,
                            grad_outputs=torch.ones_like(u1_y),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    grad_u2 = autograd.grad(outputs=u2, inputs=xif1,
                            grad_outputs=torch.ones_like(u2),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u2_x = torch.unsqueeze(grad_u2[:, 0],1)
    u2_y = torch.unsqueeze(grad_u2[:, 1],1)
    grad_u21 = autograd.grad(outputs=u2_x, inputs=xif1,
                            grad_outputs=torch.ones_like(u2_x),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_u22 = autograd.grad(outputs=u2_y, inputs=xif1,
                            grad_outputs=torch.ones_like(u2_y),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    grad_phi = autograd.grad(outputs=exact_phi, inputs=x_if,
                            grad_outputs=torch.ones_like(exact_phi),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_x = torch.unsqueeze(grad_phi[:, 0],1)
    phi_y = torch.unsqueeze(grad_phi[:, 1],1)
    grad_phi1 = autograd.grad(outputs=phi_x, inputs=x_if,
                            grad_outputs=torch.ones_like(phi_x),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_phi2 = autograd.grad(outputs=phi_y, inputs=x_if,
                            grad_outputs=torch.ones_like(phi_y),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    
    u1_xx = torch.unsqueeze(grad_u11[:, 0],1)
    u1_xy = torch.unsqueeze(grad_u11[:, 1],1)
    u1_yx = torch.unsqueeze(grad_u12[:, 0],1)
    u1_yy = torch.unsqueeze(grad_u12[:, 1],1)
    u2_xx = torch.unsqueeze(grad_u21[:, 0],1)
    u2_xy = torch.unsqueeze(grad_u21[:, 1],1)
    u2_yx = torch.unsqueeze(grad_u22[:, 0],1)
    u2_yy = torch.unsqueeze(grad_u22[:, 1],1)
    phi_xx = torch.unsqueeze(grad_phi1[:, 0],1)
    phi_xy = torch.unsqueeze(grad_phi1[:, 1],1)
    phi_yx = torch.unsqueeze(grad_phi2[:, 0],1)
    phi_yy = torch.unsqueeze(grad_phi2[:, 1],1)   
    loss_if =  torch.mean((u2 - u1- exact_phi)**2) + torch.mean((u2_x - u1_x - phi_x)**2) 
    + torch.mean((u2_y - u1_y- phi_y)**2) + torch.mean((u2_xx - u1_xx- phi_xx)**2)
    + torch.mean((u2_xy - u1_xy-phi_xy)**2) + torch.mean((u2_yx - u1_yx- phi_yx)**2) 
    + torch.mean((u2_yy - u1_yy- phi_yy)**2)
    loss1 = torch.mean((2*torch.sum(torch.cat((u2_x,u2_y),1)*nor,1) - (torch.sum(torch.cat((u1_x,u1_y),1)*nor,1)))**2)
    loss2 = torch.mean((2*torch.sum(torch.cat((u2_xx,u2_xy),1)*nor,1) - (torch.sum(torch.cat((u1_xx,u1_xy),1)*nor,1)))**2) 
    + torch.mean((2*torch.sum(torch.cat((u2_yx,u2_yy),1)*nor,1) - (torch.sum(torch.cat((u1_yx,u1_yy),1)*nor,1)))**2)
    loss = loss_if + loss1 + loss2
    return loss

''''-------------------------Step7：定义误差函数--------------------------------------------'''

# 定义L2误差和相对L2误差
# compute the relative l2 error
def compute_rel_l2_loss(points_in: torch.Tensor, points_out: torch.Tensor, truth_u: torch.Tensor) -> torch.Tensor:
    points_in = points_in.to(device)
    points_out = points_out.to(device)
    truth_u = truth_u.to(device)
    pred_u = torch.concat((model(points_in), model(points_out)), 0).to(device)
    loss_l2 = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2)))
    loss_rel_l2 = loss_l2 / torch.sqrt(torch.mean(torch.pow(truth_u, 2)))
    return loss_l2, loss_rel_l2

def compute_h2_loss(points_in: torch.Tensor, points_out: torch.Tensor, truth_u: torch.Tensor, dtrue_dx: torch.Tensor, dtrue_dy: torch.Tensor, d2true_dx2: torch.Tensor, d2true_dy2: torch.Tensor, d2true_dxdy: torch.Tensor) -> torch.Tensor:
    model.cpu()
    points = torch.concat((points_in, points_out), 0)
    output = model(points)
    du = autograd.grad(outputs=output, inputs=points,
                            grad_outputs=torch.ones_like(output),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_x = torch.unsqueeze(du[:, 0],1)
    u_y = torch.unsqueeze(du[:, 1],1)
    d2u1 = autograd.grad(outputs=u_x, inputs=points,
                            grad_outputs=torch.ones_like(u_x),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    d2u2 = autograd.grad(outputs=u_y, inputs=points,
                            grad_outputs=torch.ones_like(u_y),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_xx = torch.unsqueeze(d2u1[:, 0],1)
    u_xy = torch.unsqueeze(d2u1[:, 1],1)
    u_yy = torch.unsqueeze(d2u2[:, 1],1)
    h2_loss0 = torch.mean(torch.pow(output - truth_u, 2))
    h2_loss1 = torch.mean(torch.pow(u_x - dtrue_dx, 2))
    h2_loss2 = torch.mean(torch.pow(u_y - dtrue_dy, 2))
    h2_loss3 = torch.mean(torch.pow(u_xx - d2true_dx2, 2))
    h2_loss4 = torch.mean(torch.pow(u_yy - d2true_dy2, 2))
    h2_loss5 = torch.mean(torch.pow(u_xy - d2true_dxdy, 2))
    loss_h2 = torch.sqrt(h2_loss0 + h2_loss1 + h2_loss2 + h2_loss3 + h2_loss4 + h2_loss5)
    return loss_h2

points = get_omega_points(5000)
points_op = get_omega_p_points(points).double().requires_grad_(True)
points_on = get_omega_n_points(points).double().requires_grad_(True)
points_op_tr = dimension3_data_subtract1(points_op).double().requires_grad_(True)
points_on_tr = dimension3_data_add1(points_on).double().requires_grad_(True)
truth_u = torch.concat((u_x(points_op), u_x(points_on)), 0)
dtrue_dx = dtrue_dy = torch.concat((2 * (1 - 0.25 * torch.pow(u_x(points_op), 2)),
                                            1 - torch.pow(u_x(points_on), 2)), 0)

d2true_dx2 = d2true_dy2 = d2true_dxdy = torch.concat((
                -2 * u_x(points_op) * (1 - 0.25 * torch.pow(u_x(points_op), 2)),
                -2 * u_x(points_on) * (1 - torch.pow(u_x(points_on), 2))), 0)

''''-------------------------Step8：模型训练--------------------------------------------'''

error_l2 = np.zeros((5,4))
error_h2 = np.zeros((5,4))
df = pd.DataFrame([], columns = ['best_h2_10','best_h2_20','best_h2_40','best_h2_80']) 

for j in range(5):
    for i in range(4):
        mr = mr_list[i]
        n_hidden = n_hidden_list[i]
        rate = learning_rate
        model = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=rate)
        mb = m_gama = int(np.floor(mr/10))
       # model_path = 'D:/code/model/model_checkpoint.pth'
        best_l2, best_rel_l2, best_h2, best_epoch = 1000, 1000, 1000, 0
        optimizer = optimizer
            # 加载模型
        # model = torch.load(model_path)
        for epoch in range(1, 1 + epochs):
            if epoch % 1000 == 0:
                rate  = rate / 2           
            xr = get_omega_points(num=mr)
            xb = get_boundary_points(num=mb).double().requires_grad_(True)
            xi = get_interface_points(num=m_gama).double().requires_grad_(True)
            xop = get_omega_p_points(xr).double().requires_grad_(True)
            xon = get_omega_n_points(xr).double().requires_grad_(True)
            xop_tr = dimension3_data_subtract1(xop).double().requires_grad_(True)
            xon_tr = dimension3_data_add1(xon).double().requires_grad_(True)
            xb1 = dimension3_data_add1(xb).double().requires_grad_(True)
            xif1 = dimension3_data_add1(xi).double().requires_grad_(True)
            xif_1 = dimension3_data_subtract1(xi).double().requires_grad_(True)
            nor = get_normal_interface(xi).double()
            xb = xb.to(device)
            xi = xi.to(device)
            xop = xop.to(device)
            xon = xon.to(device)
            xop_tr = xop_tr.to(device)
            xon_tr = xon_tr.to(device)
            xb1 = xb1.to(device)
            xif1 = xif1.to(device)
            xif_1 = xif_1.to(device)
            nor =nor.to(device)
            loss_1 = func_loss_op(xop,xop_tr)           
            loss_2 = func_loss_on(xon,xon_tr)
            loss_mb = func_loss_b(xb,xb1)
            loss_m_if = func_loss_if(xi,xif1,xif_1,nor)
            loss = loss_1 + loss_2 + loss_mb + loss_m_if
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l2_loss, rel_l2_loss = compute_rel_l2_loss(points_op_tr, points_on_tr,truth_u)
            h2_loss = compute_h2_loss(points_op_tr, points_on_tr, truth_u, dtrue_dx,dtrue_dy,d2true_dx2,d2true_dy2,d2true_dxdy)
            model = model.to(device)
            if h2_loss < best_h2:
                best_rel_l2 = rel_l2_loss
                best_l2 = l2_loss
                best_h2 = h2_loss
                best_epoch = epoch
                torch.save(model, 'model%d_pth'%mr)
            print('epoch:', epoch,
                  'Point,n_hidden:', mr,n_hidden,
                  'loss:', '%.6f' % loss,
                  'rel_l2:', '%.6f' % rel_l2_loss,
                  'best_epoch',  best_epoch,
                  'best_l2:','%.6f'% best_l2,
                  'best_rel_l2', '%.6f' % best_rel_l2,
                  'best_h2:','%.6f' % best_h2)
        error_h2[j][i] = best_h2    
        print('best epoch:', best_epoch, 'best rel_l2 error:', best_rel_l2)
        print('best l2 error:', best_l2)
        print('number of points in Omega:', mr)
    df2 = pd.DataFrame([[error_h2[j][0],error_h2[j][1],error_h2[j][2],error_h2[j][3]]],
                           columns = ['best_h2_10','best_h2_20','best_h2_40','best_h2_80'])
    df = df._append(df2)
    Document = pd.ExcelWriter(r'exmaple1_h2_10000.xlsx')
    df.to_excel(excel_writer=Document,sheet_name="epoch_10000",index=False)
    
ave_h2 = np.mean(error_h2,0)
df2 = pd.DataFrame([[ave_h2[0],ave_h2[1],ave_h2[2],ave_h2[3]]],
                           columns = ['best_h2_10','best_h2_20','best_h2_40','best_h2_80'])
df = df._append(df2)
Document = pd.ExcelWriter(r'exmaple1_h2_10000.xlsx')
df.to_excel(excel_writer=Document,sheet_name="epoch_10000",index=False)
Document._save()
Document.close()

x = np.array([1, 2, 3])
y = ave_h2[1:]/ave_h2[0:-1]
straight_line = np.ones(3)*np.sqrt(2)/2
plt.plot(x,straight_line)
plt.scatter(x,y)
plt.xlabel('number of n_hidden')
plt.ylabel('best_h2')
plt.show()

