# ============================== 1. 导入依赖库 ==============================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time
# 解决OpenMP多线程冲突（PyTorch与其他库重复加载问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float64)

# ============================== 2. 全局配置与参数定义 ==============================
def set_global_config():
    """
    设置全局超参数与计算设备
    返回：
        config (dict): 超参数字典
        device (torch.device): 计算设备（GPU优先）
    """
    config = {
        "dim": 3,                  # 问题维度（3D）
        "basis_num_minus_1": 5000,  # 基函数个数-1（实际基函数数=M+1）
        "max_regions": 10,          # 最大子区域数（预留内存）
        "weight_std": 1,            # 基函数权重正态分布标准差
        "bias_range": 1,            # 基函数偏置均匀分布范围（[-b, b]）
        "tolerance": 1e-4,          # 迭代终止阈值（平均残差<tol停止）
        "init_internal_pts": 8000,    # 初始区域内部配置点
        "init_boundary_pts": 3000,    # 初始区域边界配置点
        "subregion_pts": 20,        # 子区域配置点                                         
        "subregion_radius": 0.1,   # 子区域球体半径
        "max_iter": 1              # 最大区域分解迭代次数
    }  
    
    # 自动选择计算设备（GPU优先，无则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备：{device}")
    
    return config, device


# ============================== 3. 初始化工具函数 ==============================
def init_basis_params(config, device):
    """
    初始化基函数的权重、偏置、系数容器
    参数：
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        weights (list): 各区域基函数权重（shape: [max_regions, M, dim]）
        biases (list): 各区域基函数偏置（shape: [max_regions, M]）
        alphas (torch.Tensor): 基函数系数（shape: [max_regions, M+1]）
    """
    M = config["basis_num_minus_1"]
    max_regions = config["max_regions"]
    dim = config["dim"]
    R = config["weight_std"]
    b = config["bias_range"]
    
    # 初始化权重（正态分布）
    weights = [
        torch.normal(0, R, size=(M, dim), dtype=torch.float64, device=device)
        for _ in range(max_regions)
    ]
    
    # 初始化偏置（均匀分布）
    biases = [
        torch.rand((M,), dtype=torch.float64, device=device) * b 
        for _ in range(max_regions)
    ]
    
    # 初始化系数（全0，后续求解更新）
    alphas = torch.zeros(
        (max_regions, M + 1), dtype=torch.float64, device=device
    )
    
    return weights, biases, alphas


def sample_sphere_interior(center, r, num_points, device, cube_constraint=True):
    """
    在球体内部（严格小于半径）随机取点，可选限制在 [0,1]^3 内。
    
    Args:
        center (torch.Tensor): 球心坐标，shape=(3,)
        r (float): 球体半径
        num_points (int): 需采样的点数
        device (torch.device): 计算设备（cpu/cuda）
        cube_constraint (bool): 是否限制点在 [0,1]^3 范围内（默认True）
    
    Returns:
        torch.Tensor: 采样点坐标，shape=(num_points, 3)
    """
    sampled_points = []
    # 将 r 转换为 Tensor（同设备、同类型）
    r_tensor = torch.tensor(r, device=device, dtype=center.dtype)
    
    while len(sampled_points) < num_points:
        # 1. 生成三维标准正态分布向量（方向均匀随机）
        norm_vec = torch.randn((num_points * 2, 3), device=device, dtype=center.dtype)
        # 过滤掉模长接近0的异常向量（概率极低）
        norm_vec = norm_vec[torch.norm(norm_vec, dim=1) > 1e-6]
        if len(norm_vec) == 0:
            continue
        
        # 2. 归一化向量（得到单位球面上的点）
        unit_vec = norm_vec / torch.norm(norm_vec, dim=1, keepdim=True)
        
        # 3. 生成随机半径（r * (uniform(0,1))^(1/3)）：保证球内体积均匀分布
        random_radius = r_tensor * torch.rand((len(unit_vec), 1), device=device, dtype=center.dtype) ** (1/3)
        
        # 4. 缩放+平移：得到目标球内部的点
        candidate_points = center + unit_vec * random_radius
        
        # 5. 筛选：严格在球内（距离<r）+ 可选[0,1]^3约束
        dist_to_center = torch.norm(candidate_points - center, dim=1)
        in_sphere = dist_to_center < (r_tensor - 1e-6)  # 用 Tensor 计算，避免类型问题
        if cube_constraint:
            in_cube = (candidate_points >= 0).all(dim=1) & (candidate_points <= 1).all(dim=1)
            valid_points = candidate_points[in_sphere & in_cube]
        else:
            valid_points = candidate_points[in_sphere]
        
        # 收集有效点，直到满足数量
        sampled_points.extend(valid_points)
    
    # 截取指定数量的点并返回
    return torch.stack(sampled_points[:num_points], dim=0)


def sample_sphere_surface(center, r, num_points, device, cube_constraint=True):
    """
    在球面上（距离=半径）随机取点，可选限制在 [0,1]^3 内。
    
    Args:
        center (torch.Tensor): 球心坐标，shape=(3,)
        r (float): 球体半径
        num_points (int): 需采样的点数
        device (torch.device): 计算设备（cpu/cuda）
        cube_constraint (bool): 是否限制点在 [0,1]^3 范围内（默认True）
    
    Returns:
        torch.Tensor: 采样点坐标，shape=(num_points, 3)
    """
    sampled_points = []
    # 关键修复：将 r 转换为同设备、同类型的 Tensor
    r_tensor = torch.tensor(r, device=device, dtype=center.dtype)
    
    while len(sampled_points) < num_points:
        # 1. 生成三维标准正态分布向量（方向均匀随机）
        norm_vec = torch.randn((num_points * 2, 3), device=device, dtype=center.dtype)
        # 过滤掉模长接近0的异常向量
        norm_vec = norm_vec[torch.norm(norm_vec, dim=1) > 1e-6]
        if len(norm_vec) == 0:
            continue
        
        # 2. 归一化向量（得到单位球面上的点）
        unit_vec = norm_vec / torch.norm(norm_vec, dim=1, keepdim=True)
        
        # 3. 缩放+平移：得到目标球面上的点（距离球心=半径）
        candidate_points = center + unit_vec * r_tensor
        
        # 4. 筛选：严格在球面上（允许微小数值误差）+ 可选[0,1]^3约束
        dist_to_center = torch.norm(candidate_points - center, dim=1)
        # 两个输入都是 Tensor，类型匹配
        on_surface = torch.isclose(dist_to_center, r_tensor, atol=1e-5)
        if cube_constraint:
            in_cube = (candidate_points >= 0).all(dim=1) & (candidate_points <= 1).all(dim=1)
            valid_points = candidate_points[on_surface & in_cube]
        else:
            valid_points = candidate_points[on_surface]
        
        # 收集有效点，直到满足数量
        sampled_points.extend(valid_points)
    
    # 截取指定数量的点并返回
    return torch.stack(sampled_points[:num_points], dim=0)

# 定义网络模型
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

def init_config_points(config, device):
    """
    初始化初始区域的内部点与边界点（3D立方体域：[-1,1]^3）
    参数：
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        internal_pts (torch.Tensor): 内部点（shape: [N^3, dim]）
        boundary_pts (torch.Tensor): 边界点（shape: [6*N_g^2, dim]）
    """
    dim = config["dim"]
    N = config["init_internal_pts"]
    N_g = config["init_boundary_pts"]
    internal_pts = get_omega_points(N).to(device)
    boundary_pts = get_boundary_points(N_g).to(device)
    
    return internal_pts, boundary_pts


# ============================== 4. 核心算法函数 ==============================
def basis_psi(points, region_idx, centers, weights, biases, device):
    """
    计算基函数值 ψ(x)（含常数项m=0）
    参数：
        points (torch.Tensor): 输入点（shape: [N, dim]）
        region_idx (int): 区域索引
        centers (torch.Tensor): 各区域圆心（shape: [max_regions+1, dim]）
        weights (list): 基函数权重
        biases (list): 基函数偏置
        device (torch.device): 计算设备
    返回：
        psi_vals (torch.Tensor): 基函数值（shape: [N, M+1]）
    """
    # 子区域圆心平移（对齐原点，提升拟合效果）
    points_shifted = points - centers[region_idx]
    N = points_shifted.shape[0]
    
    # m=0：常数基函数（全1）
    psi_const = torch.ones((N, 1), dtype=torch.float64, device=device)
    
    # m>0：双曲正切基函数 tanh(w·x + b)
    w = weights[region_idx]  # [M, dim]
    b = biases[region_idx]  # [M,]
    b_expanded = b.unsqueeze(0).repeat(N, 1)  # [N, M]
    psi_act = torch.tanh(torch.matmul(points_shifted, w.T) + b_expanded)  # [N, M]
    
    # 拼接常数项与激活项
    return torch.cat([psi_const, psi_act], dim=1)


def laplacian_psi(points, region_idx, centers, weights, biases, config, device):
    """
    计算基函数的拉普拉斯算子 ∇²ψ(x)（PDE核心算子）
    参数：
        points (torch.Tensor): 输入点（shape: [N, dim]）
        region_idx (int): 区域索引
        centers (torch.Tensor): 各区域圆心
        weights (list): 基函数权重
        biases (list): 基函数偏置
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        laplacian_vals (torch.Tensor): 拉普拉斯值（shape: [N, M+1]）
    """
    M = config["basis_num_minus_1"]
    dim = config["dim"]
    
    # 先计算基函数值
    psi_vals = basis_psi(points, region_idx, centers, weights, biases, device)
    # 子区域平移
    points_shifted = points - centers[region_idx]
    
    # 拉普拉斯核心计算（基于tanh二阶导数：-2tanh(u)+2tanh³(u)）
    laplacian_core = -2 * psi_vals + 2 * psi_vals ** 3
    
    # 构造对角矩阵D（权重平方和，对应各维度二阶导叠加）
    w = weights[region_idx]  # [M, dim]
    diag_vals = torch.zeros((M + 1,), dtype=torch.float64, device=device)
    for i in range(dim):
        diag_vals[1:] += w[:, i] ** 2  # m=0项权重为0
    D = torch.diag(diag_vals)  # [M+1, M+1]
    # 最终拉普拉斯值：core @ D
    return torch.matmul(laplacian_core, D)


def dx_basis_psi(points, region_idx, centers, weights, biases, device):
    """
    计算基函数对 x 方向的偏导数 dψ/dx。

    参数:
        points (torch.Tensor): 输入点 (shape: [N, dim])
        region_idx (int): 区域索引
        centers (torch.Tensor): 各区域圆心 (shape: [max_regions+1, dim])
        weights (list): 基函数权重
        biases (list): 基函数偏置
        device (torch.device): 计算设备

    返回:
        dx_psi_vals (torch.Tensor): 基函数对 x 方向的偏导数值 (shape: [N, M+1])
    """
    # 1. 计算原始的基函数值，因为导数的计算需要用到它
    psi_vals = basis_psi(points, region_idx, centers, weights, biases, device)
    
    # 2. 计算平移后的点
    points_shifted = points - centers[region_idx]
    
    # 3. 获取当前区域的权重
    w = weights[region_idx]  # shape: [M, dim]
    
    # 4. 初始化导数结果张量，形状与 psi_vals 相同
    dx_psi_vals = torch.zeros_like(psi_vals, device=device)
    
    # --- 对 x 方向求导 ---
    # 对于 m=0 的常数项，导数为 0，保持初始值即可。
    
    # 对于 m>0 的激活项:
    # 4.1 获取 x 方向的权重 (假设 x 是第 0 维)
    w_x = w[:, 0]  # shape: [M,]
    
    # 4.2 计算 (1 - psi_act^2)，并扩展维度以便广播
    # psi_vals[:, 1:] 是所有点在所有激活基函数上的值
    sech_squared = 1 - psi_vals[:, 1:] ** 2  # shape: [N, M]
    
    # 4.3 应用链式法则: d(psi_act)/dx = (1 - psi_act^2) * w_x
    # 将 w_x 扩展为 [1, M] 以便与 [N, M] 的 sech_squared 相乘
    dx_psi_vals[:, 1:] = sech_squared * w_x
    return dx_psi_vals

'''-------------------------Functions-------------------------'''
Re = 100

def u_x(points, config, device):
    x1 = points[:, 0]
    x2 = points[:, 1]
    x3 = points[:, 2]
    u = ((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
    u = torch.unsqueeze(u, 1)
    return u

def pde_source(points, config, device):
    x1 = points[:, 0]
    x2 = points[:, 1]
    x3 = points[:, 2]
    du_dx = - Re * (torch.exp(Re * x1) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
    d2u_dx2 = -(Re ** 2 * torch.sin(x2) * torch.sin(x3) * torch.exp(Re * x1)) / (torch.exp(torch.tensor(Re)) - 1)
    d2u_dy2 = -((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3))/ (torch.exp(torch.tensor(Re)) - 1)
    d2u_dz2 = -((torch.exp(torch.tensor(Re)) - torch.exp(Re * x1)) * torch.sin(x2) * torch.sin(x3)) / (torch.exp(torch.tensor(Re)) - 1)
    f = -(d2u_dx2 + d2u_dy2 + d2u_dz2) + Re * du_dx
    f = torch.unsqueeze(f, 1)
    return f

def pde_analytical_sol(points, device):
    pass

def pde_analytical_normal(points, config, device):
    pass

def qr_least_squares(A, b, device):
    """
    QR分解求解最小二乘问题 A·x = b（比直接求逆更稳定）
    参数：
        A (torch.Tensor): 系数矩阵（shape: [M, N]）
        b (torch.Tensor): 右端项（shape: [M, 1]）
        device (torch.device): 计算设备
    返回：
        x (torch.Tensor): 最小二乘解（shape: [N,]）
    """
    # QR分解（reduced模式：只保留有效秩）
    Q, R = torch.linalg.qr(A, mode="reduced")
    # 转换为上三角方程：R·x = Q^T·b
    rhs = torch.matmul(Q.T, b)
    # 解上三角方程
    x = torch.linalg.solve_triangular(R, rhs, upper=True)
    return x.squeeze()


def find_max_residual_point(internal_pts, alphas, centers, weights, biases, config, device):
    """
    找到内部点中残差最大的点（作为下一个子区域的圆心）
    参数：
        internal_pts (torch.Tensor): 内部点（shape: [N, dim]）
        alphas (torch.Tensor): 基函数系数
        centers (torch.Tensor): 各区域圆心
        weights (list): 基函数权重
        biases (list): 基函数偏置
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        max_residual_pt (torch.Tensor): 最大残差点（shape: [dim,]）
        avg_residual (float): 平均残差
    """
    # 计算每个点的残差：|∇²ψ·α - f|²
    laplacian_vals = - laplacian_psi(internal_pts, 0, centers, weights, biases, config, device) + Re * dx_basis_psi(internal_pts, 0, centers, weights, biases, device)
    f_vals = pde_source(internal_pts, config, device)
    residuals = torch.abs(torch.matmul(laplacian_vals, alphas[0].unsqueeze(1)) - f_vals)
    # 找最大残差点与平均残差
    max_residual, max_idx = torch.max(residuals, dim=0)
    avg_residual = torch.mean(residuals).item()
    
    return internal_pts[max_idx], avg_residual


def domain_decomposition(center, original_internal, original_boundary, config, device):
    """
    区域分解：生成新子区域的配置点，并更新原区域的配置点
    参数：
        center (torch.Tensor): 新子区域圆心（shape: [dim,]）
        original_internal (torch.Tensor): 原区域内部点
        original_boundary (torch.Tensor): 原区域边界点
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        sub_internal (torch.Tensor): 子区域内部点
        sub_boundary (torch.Tensor): 子区域边界点（原边界中属于子区域的部分）
        sub_interface (torch.Tensor): 子区域交界点（球体表面）
        new_original_internal (torch.Tensor): 原区域剩余内部点
        new_original_boundary (torch.Tensor): 原区域剩余边界点
    """
    r = config["subregion_radius"]
    N1 = config["subregion_pts"]
    dim = config["dim"]
    
    sub_internal = sample_sphere_interior(center, r, N1**3, device, cube_constraint=True)
    
    # -------------------------- 2. 生成子区域交界点（球体表面） --------------------------
    sub_interface = sample_sphere_surface(center, r, N1**2, device, cube_constraint=True)
    
    # -------------------------- 3. 更新原区域配置点（排除子区域内的点） --------------------------
    # 原区域剩余内部点（距离>r）
    dist_original_internal = torch.norm(original_internal - center, dim=1)
    new_original_internal = original_internal[dist_original_internal > r]
    
    # 原区域剩余边界点（分离子区域内的部分）
    dist_original_boundary = torch.norm(original_boundary - center, dim=1)
    sub_boundary = original_boundary[dist_original_boundary <= r]  # 子区域内的原边界点
    new_original_boundary = original_boundary[dist_original_boundary > r]  # 原区域剩余边界点
    
    return sub_internal, sub_boundary, sub_interface, new_original_internal, new_original_boundary


# ============================== 5. 数值求解主逻辑 ==============================
def solve_pde_annb(config, device):
    """
    ANNB方法求解3D PDE主逻辑：初始化→初始求解→自适应区域分解→迭代优化
    参数：
        config (dict): 超参数字典
        device (torch.device): 计算设备
    返回：
        results (dict): 求解结果（系数、区域圆心、配置点、误差）
    """
    # -------------------------- 1. 初始化基础参数与容器 --------------------------
    M = config["basis_num_minus_1"]
    max_regions = config["max_regions"]
    dim = config["dim"]
    tol = config["tolerance"]
    max_iter = config["max_iter"]
    
    # 基函数参数（权重、偏置、系数）
    weights, biases, alphas = init_basis_params(config, device)
    
    # 初始区域配置点
    init_internal, init_boundary = init_config_points(config, device)
    
    # 区域相关容器（内部点、边界点、交界点、圆心）
    region_internal = [torch.zeros(0, dim, device=device) for _ in range(max_regions)]
    region_boundary = [torch.zeros(0, dim, device=device) for _ in range(max_regions)]
    region_interface = [torch.zeros(0, dim, device=device) for _ in range(max_regions)]
    region_centers = torch.zeros((max_regions + 1, dim), device=device)  # centers[0]为初始区域中心（原点）
    region_centers[0] = 0.5  # 利用广播机制，dim维都会被设为0.5
    # 初始化初始区域（K=0）
    region_internal[0] = init_internal
    region_boundary[0] = init_boundary
    
    # 当前子区域数（初始为0，仅初始区域）
    current_K = 0
    
    # -------------------------- 2. 初始区域求解（K=0） --------------------------
    print("\n=== 初始区域求解（K=0） ===")
    # 构造初始区域的A矩阵与b向量
    # 边界条件：ψ·α = g
    A_boundary = basis_psi(region_boundary[0], 0, region_centers, weights, biases, device)
    b_boundary = u_x(region_boundary[0], config, device)
    
    # 内部PDE条件：∇²ψ·α = f
    A_internal = - laplacian_psi(region_internal[0], 0, region_centers, weights, biases, config, device) + Re * dx_basis_psi(region_internal[0], 0, region_centers, weights, biases, device)
    b_internal = pde_source(region_internal[0], config, device)
    
    # 拼接A和b（合并边界与内部条件）
    A_init = torch.cat([100*A_boundary, A_internal], dim=0)
    b_init = torch.cat([100*b_boundary, b_internal], dim=0)
    # QR分解求解初始区域系数
    alphas[0] = qr_least_squares(A_init, b_init, device)
    # 计算初始区域残差与最大残差点（首个子区域圆心）
    max_res_pt, avg_residual = find_max_residual_point(
        region_internal[0], alphas, region_centers, weights, biases, config, device
    )
    region_centers[1] = max_res_pt  # 首个子区域圆心为最大残差点
    
    x1 = torch.linspace(0, 1, 30)
    x2 = torch.linspace(0, 1, 30)
    x3 = torch.linspace(0, 1, 30)
    X1, X2,X3 = torch.meshgrid(x1, x2,x3)
    Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None],X3.flatten()[:, None]),dim=1)
    verify_points = Z.to(device)
    global_analytical = u_x(verify_points, config, device).squeeze()
    global_approx = torch.zeros_like(global_analytical, device=device)
    psi_val = basis_psi(verify_points, 0, region_centers, weights, biases, device)
    global_approx = torch.matmul(psi_val, alphas[0].unsqueeze(1)).squeeze()
    # 计算全局相对L2误差
    global_err_sq = torch.sum((global_approx - global_analytical) ** 2)
    global_total_sq = torch.sum(global_analytical ** 2)
    global_relative_err = torch.sqrt(global_err_sq / global_total_sq).item()
    print(f"初次全局相对L2误差：{global_relative_err:.6f}")
    # 计算初始区域总损失
    init_loss = torch.sum((torch.matmul(A_init, alphas[0].unsqueeze(1)) - b_init) ** 2).item()
    # 记录结束时间
    end_time = time.perf_counter()

    # 计算并打印运行时间（保留6位小数）
    print(f"程序运行时间：{end_time - start_time:.6f} 秒")
    print(f"初始区域最大残差点：{region_centers[1]}")
    print(f"初始区域平均残差：{avg_residual:.6f}")
    print(f"初始区域总损失：{init_loss:.6f}")
    
    # -------------------------- 3. 自适应子区域分解迭代 --------------------------
    print("\n=== 开始自适应区域分解迭代 ===")
    while avg_residual > tol and current_K < max_iter:
        current_K += 1
        print(f"\n--- 分解第 {current_K} 个子区域 ---")
        
        # 1. 区域分解：生成新子区域配置点，更新原区域配置点
        sub_internal, sub_boundary, sub_interface, new_original_internal, new_original_boundary = domain_decomposition(
            center=region_centers[current_K],
            original_internal=region_internal[0],
            original_boundary=region_boundary[0],
            config=config,
            device=device
        )
        
        # 保存新子区域的配置点
        region_internal[current_K] = sub_internal
        region_boundary[current_K] = sub_boundary
        region_interface[current_K] = sub_interface
        
        # 更新初始区域的配置点（排除新子区域的部分）
        region_internal[0] = new_original_internal
        region_boundary[0] = new_original_boundary
        
        # 2. 重新求解初始区域系数（因配置点已更新）
        print(f"重新求解初始区域系数（剩余内部点：{region_internal[0].shape[0]}）")
        A_boundary_new = basis_psi(region_boundary[0], 0, region_centers, weights, biases, device)
        b_boundary_new = u_x(region_boundary[0], config, device)
        A_internal_new = - laplacian_psi(region_internal[0], 0, region_centers, weights, biases, config, device) + Re * dx_basis_psi(region_internal[0], 0, region_centers, weights, biases, device)
        b_internal_new = pde_source(region_internal[0], config, device)
        
        # 边界条件加权（提升边界拟合优先级）
        A_original_update = torch.cat([100 * A_boundary_new, A_internal_new], dim=0)
        b_original_update = torch.cat([100 * b_boundary_new, b_internal_new], dim=0)
        
        # 求解更新后的初始区域系数
        alphas[0] = qr_least_squares(A_original_update, b_original_update, device)
        
        # 3. 求解新子区域系数（遍历权重缩放系数s，选损失最小的）
        print(f"求解第 {current_K} 个子区域系数（内部点：{sub_internal.shape[0]}）")
        # 权重缩放系数s的候选（此处取s=3，可扩展更多候选）
        s_candidates = range(3, 4)
        best_loss = float('inf')
        best_s = 3
        best_weight = weights[current_K].clone()
        best_alpha = alphas[current_K].clone()
        
        for s in s_candidates:
            # 生成缩放后的权重（s越大，权重范围越广）
            scaled_weight = torch.normal(
                0, config["weight_std"] * (s + 1),
                size=(M, dim), dtype=torch.float64, device=device
            )
            weights[current_K] = scaled_weight
            
            # 构造子区域的A矩阵与b向量（4个条件：交界点值、内部PDE、交界点法向导数、边界值）
            # 交界点值条件：ψ·α = g
            A_interface = basis_psi(sub_interface, current_K, region_centers, weights, biases, device)
            b_interface = torch.matmul(basis_psi(sub_interface, 0, region_centers, weights, biases, device), alphas[0]).unsqueeze(1)
            
            # 内部PDE条件：∇²ψ·α = f
            A_sub_internal = - laplacian_psi(sub_internal, current_K, region_centers, weights, biases, config, device) + Re * dx_basis_psi(sub_internal, current_K, region_centers, weights, biases, device)
            b_sub_internal = pde_source(sub_internal, config, device)
            
            # 子区域边界条件：ψ·α = g
            A_sub_boundary = basis_psi(sub_boundary, current_K, region_centers, weights, biases, device)
            b_sub_boundary = u_x(sub_boundary, config, device)
            # 拼接A和b
            A_sub = torch.cat([A_interface, A_sub_internal, 100*A_sub_boundary], dim=0)
            b_sub = torch.cat([b_interface, b_sub_internal, 100*b_sub_boundary], dim=0)
            # 求解子区域系数
            alpha_sub = qr_least_squares(A_sub, b_sub, device)
            alphas[current_K] = alpha_sub
            
            # 计算当前s的损失与误差
            current_loss = torch.sum((torch.matmul(A_sub, alpha_sub.unsqueeze(1)) - b_sub) ** 2).item()
            
            print(f"子区域系数：s={s}，损失={current_loss:.6f}")
            
            # 更新最优s
            if current_loss < best_loss:
                best_loss = current_loss
                best_s = s
                best_weight = scaled_weight.clone()
                best_alpha = alpha_sub.clone()
        
        # 保存最优参数
        weights[current_K] = best_weight
        alphas[current_K] = best_alpha
        
        # 4. 寻找下一个子区域的圆心（原区域剩余内部点的最大残差点）
        max_res_pt, avg_residual = find_max_residual_point(
            region_internal[0], alphas, region_centers, weights, biases, config, device
        )
        region_centers[current_K + 1] = max_res_pt
        
        print(f"第 {current_K} 个子区域：最优s={best_s}，平均残差={avg_residual:.6f}，下一个圆心={max_res_pt}")
        print("\n=== 计算全局相对L2误差 ===")
        # 为每个验证点匹配所属区域并计算近似解
        for idx, pt in enumerate(verify_points):
            # 找到点所属的区域（优先子区域，再初始区域）
            region_idx = 0
            for k in range(1, current_K + 1):
                if torch.norm(pt - region_centers[k]) <= config["subregion_radius"]:
                    region_idx = k
                    break
            # 计算该区域的近似解
            psi_val = basis_psi(pt.unsqueeze(0), region_idx, region_centers, weights, biases, device)
            global_approx[idx] = torch.matmul(psi_val, alphas[region_idx].unsqueeze(1)).squeeze()
        # 计算全局相对L2误差
        # 拆分 verify_points 的 x/y/z 坐标
        global_err_sq = torch.sum((global_approx - global_analytical) ** 2)
        global_total_sq = torch.sum(global_analytical ** 2)
        global_relative_err = torch.sqrt(global_err_sq / global_total_sq).item()
        print(f"全局相对L2误差：{global_relative_err:.6f}")
            # 记录结束时间
        end_time = time.perf_counter()

        # 计算并打印运行时间（保留6位小数）
        print(f"程序运行时间：{end_time - start_time:.6f} 秒")
    print(region_centers)
    x = verify_points.cpu().numpy()[:, 0]  # 所有点的x坐标
    y = verify_points.cpu().numpy()[:, 1]  # 所有点的y坐标
    z = verify_points.cpu().numpy()[:, 2]  # 所有点的z坐标
    # 预测值（global_approx）作为颜色映射依据

    # 2. 创建 3D 绘图对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    approx_vals = global_approx.cpu().numpy()
    # 3. 绘制三维散点图
    # s：点的大小（根据数据量调整，数据多则调小）
    # c：颜色映射的数值（即预测值）
    # cmap：颜色映射方案（如 viridis、plasma，可自行替换）
    scatter = ax.scatter(x, y, z, c=approx_vals, s=5, cmap='viridis', alpha=0.7)

    # 4. 添加颜色条（说明颜色对应的预测值大小）
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Predicted Value (global_approx)', fontsize=12)

    # 5. 设置坐标轴标签和标题
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)
    ax.set_zlabel('Z Coordinate', fontsize=10)
    ax.set_title('3D Distribution of Predicted Values (global_approx)', fontsize=14, pad=20)

    # 6. 调整视角（elev=仰角，azim=方位角，可自行调整）
    ax.view_init(elev=20, azim=45)

    plt.show()
    
    # -------------------------- 5. 整理结果返回 --------------------------
    results = {
        "alphas": alphas,                  # 基函数系数
        "region_centers": region_centers,  # 各区域圆心
        "region_internal": region_internal,  # 各区域内部点
        "region_boundary": region_boundary,  # 各区域边界点
        "region_interface": region_interface,  # 各区域交界点
        "global_relative_err": global_relative_err,  # 全局相对误差
        "current_K": current_K,             # 最终子区域数
        "weights": weights,                # 新增：实际求解的权重
        "biases": biases                   # 新增：实际求解的偏置
    }
    return results



# ============================== 7. 主函数（程序入口） ==============================
if __name__ == "__main__":
    # 1. 设置全局配置
    config, device = set_global_config()
    # 记录开始时间
    start_time = time.perf_counter()
    # 2. 求解3D PDE
    results = solve_pde_annb(config, device)
    
    print("\n=== 程序执行完成 ===")