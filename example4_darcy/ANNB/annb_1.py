# ============================== 1. 导入依赖库 ==============================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 解决OpenMP多线程冲突（PyTorch与其他库重复加载问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float64)
from scipy.interpolate import griddata  # 新增：用于切面图插值
# ============================== 2. 全局配置与参数定义 ==============================
def set_global_config():
    """ 
    设置全局超参数与计算设备（新增主/子区域基函数数量配置）
    返回：
        config (dict): 超参数字典
        device (torch.device): 计算设备（GPU优先）
    """
    config = {
        "dim": 3,                  # 问题维度（3D）
        "main_basis_num": 6000,    # 主区域基函数数
        "sub_basis_num": 100,      # 子区域基函数数
        "num_sub_regions": 8,      # 最大子区域数量
        "main_weight_std": 5,      # 主区域基函数权重正态分布标准差
        "sub_weight_std": 5,       # 子区域基函数权重正态分布标准差
        "bias_range": 1,           # 基函数偏置均匀分布范围（[-b, b]）
        "subregion_radius": 0.2,   # 子区域球体半径
        "init_internal_pts": 6000, # 主区域初始内部点
        "init_boundary_pts": 3000, # 初始立方体边界点
        "subregion_internal_pts": 300, # 每个子区域内部点数量
        "subregion_interface_pts": 100, # 每个子区域交界点数量
        "residual_tol": 1e-6,      # 残差收敛阈值
        "max_iter": 8              # 最大迭代次数（子区域数）
    }
    
    # 自动选择计算设备（GPU优先，无则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备：{device}")
    
    return config, device

# ============================== 3. 初始化工具函数 ==============================
def init_basis_params(config, device):
    """
    分区域初始化基函数参数（主区域+最大子区域数）
    返回：
        weights (list): 各区域基函数权重 [主区域, 子区域1, ..., 子区域8]
        biases (list): 各区域基函数偏置 [主区域, 子区域1, ..., 子区域8]
        alphas (list): 基函数系数容器 [主区域, 子区域1, ..., 子区域8]
    """
    main_M = config["main_basis_num"]  # 主区域基函数数（不含常数项）
    sub_M = config["sub_basis_num"]    # 子区域基函数数（不含常数项）
    dim = config["dim"]
    main_R = config["main_weight_std"] # 主区域权重标准差
    sub_R = config["sub_weight_std"]   # 子区域权重标准差
    b = config["bias_range"]
    num_sub = config["num_sub_regions"]
    
    # 初始化权重：主区域main_M个，子区域各sub_M个
    weights = []
    weights.append(torch.normal(0, main_R, size=(main_M, dim), dtype=torch.float64, device=device))
    for _ in range(num_sub):
        weights.append(torch.normal(0, sub_R, size=(sub_M, dim), dtype=torch.float64, device=device))
    
    # 初始化偏置：主区域main_M个，子区域各sub_M个
    biases = []
    biases.append(torch.rand((main_M,), dtype=torch.float64, device=device) * 2 * b - b)
    for _ in range(num_sub):
        biases.append(torch.rand((sub_M,), dtype=torch.float64, device=device) * 2 * b - b)
    
    # 初始化系数：主区域(main_M+1)个，子区域各(sub_M+1)个（+1为常数项）
    alphas = []
    alphas.append(torch.zeros((main_M + 1,), dtype=torch.float64, device=device))
    for _ in range(num_sub):
        alphas.append(torch.zeros((sub_M + 1,), dtype=torch.float64, device=device))
    
    return weights, biases, alphas

def sample_sphere_interior(center, r, num_points, device, cube_constraint=True):
    """球体内部随机取点"""
    sampled_points = []
    r_tensor = torch.tensor(r, device=device, dtype=center.dtype)
    
    while len(sampled_points) < num_points:
        norm_vec = torch.randn((num_points * 2, 3), device=device, dtype=center.dtype)
        norm_vec = norm_vec[torch.norm(norm_vec, dim=1) > 1e-6]
        if len(norm_vec) == 0:
            continue
        
        unit_vec = norm_vec / torch.norm(norm_vec, dim=1, keepdim=True)
        random_radius = r_tensor * torch.rand((len(unit_vec), 1), device=device, dtype=center.dtype) ** (1/3)
        candidate_points = center + unit_vec * random_radius
        
        dist_to_center = torch.norm(candidate_points - center, dim=1)
        in_sphere = dist_to_center < (r_tensor - 1e-6)
        if cube_constraint:
            in_cube = (candidate_points >= 0).all(dim=1) & (candidate_points <= 1).all(dim=1)
            valid_points = candidate_points[in_sphere & in_cube]
        else:
            valid_points = candidate_points[in_sphere]
        
        sampled_points.extend(valid_points)
    
    return torch.stack(sampled_points[:num_points], dim=0)

def sample_sphere_surface(center, r, num_points, device, cube_constraint=True):
    """球体表面随机取点"""
    sampled_points = []
    r_tensor = torch.tensor(r, device=device, dtype=center.dtype)
    
    while len(sampled_points) < num_points:
        norm_vec = torch.randn((num_points * 2, 3), device=device, dtype=center.dtype)
        norm_vec = norm_vec[torch.norm(norm_vec, dim=1) > 1e-6]
        if len(norm_vec) == 0:
            continue
        
        unit_vec = norm_vec / torch.norm(norm_vec, dim=1, keepdim=True)
        candidate_points = center + unit_vec * r_tensor
        
        dist_to_center = torch.norm(candidate_points - center, dim=1)
        on_surface = torch.isclose(dist_to_center, r_tensor, atol=1e-5)
        if cube_constraint:
            in_cube = (candidate_points >= 0).all(dim=1) & (candidate_points <= 1).all(dim=1)
            valid_points = candidate_points[on_surface & in_cube]
        else:
            valid_points = candidate_points[on_surface]
        
        sampled_points.extend(valid_points)
    
    return torch.stack(sampled_points[:num_points], dim=0)

def get_omega_points(num, device, x_left=0.0, y_left=0.0, x_right=1.0, y_right=1.0, z_left=0.0, z_right=1.0):
    """生成立方体内部点"""
    x_coords = torch.rand(num, 1, device=device) * (x_right - x_left) + x_left
    y_coords = torch.rand(num, 1, device=device) * (y_right - y_left) + y_left
    z_coords = torch.rand(num, 1, device=device) * (z_right - z_left) + z_left
    return torch.cat([x_coords, y_coords, z_coords], dim=1)

def get_cube_boundary_points(num, device):
    """生成立方体边界点"""
    if num % 6 != 0:
        raise ValueError(f"num必须是6的倍数，当前为{num}")
    points_per_face = num // 6
    rand_coords_1 = torch.rand(points_per_face, device=device)
    rand_coords_2 = torch.rand(points_per_face, device=device)
    
    face_x0 = torch.stack([torch.zeros_like(rand_coords_1), rand_coords_1, rand_coords_2], dim=1)
    face_x1 = torch.stack([torch.ones_like(rand_coords_1), rand_coords_1, rand_coords_2], dim=1)
    face_y0 = torch.stack([rand_coords_1, torch.zeros_like(rand_coords_1), rand_coords_2], dim=1)
    face_y1 = torch.stack([rand_coords_1, torch.ones_like(rand_coords_1), rand_coords_2], dim=1)
    face_z0 = torch.stack([rand_coords_1, rand_coords_2, torch.zeros_like(rand_coords_1)], dim=1)
    face_z1 = torch.stack([rand_coords_1, rand_coords_2, torch.ones_like(rand_coords_1)], dim=1)
    
    return torch.cat([face_x0, face_x1, face_y0, face_y1, face_z0, face_z1], dim=0)

def alpha_x(x_op, radius: float = 0.2) -> torch.Tensor:
    """计算alpha(x)系数"""
    x = x_op[:,0].flatten()
    y = x_op[:,1].flatten()
    z = x_op[:,2].flatten()
    centers = np.array([
        [0.2, 0.2, 0.2], [0.2, 0.2, 0.8], [0.2, 0.8, 0.2], [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.2], [0.8, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.8, 0.8]
    ])
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    alpha_np = np.ones_like(x_np) * 0.018
    
    for center in centers:
        dist = np.sqrt((x_np-center[0])**2 + (y_np-center[1])**2 + (z_np-center[2])**2)
        alpha_np[dist < radius] = 137
    
    return torch.tensor(alpha_np, dtype=x.dtype, device=x.device)

# ============================== 4. 核心算法函数 ==============================
def basis_psi(points, region_idx, centers, weights, biases, device):
    """计算指定区域的基函数值"""
    points_shifted = points - centers[region_idx]
    N = points_shifted.shape[0]
    
    # 常数项（m=0）
    psi_const = torch.ones((N, 1), dtype=torch.float64, device=device)
    
    # 激活项（m>0）
    M = weights[region_idx].shape[0]
    w = weights[region_idx]  # [M, dim]
    b = biases[region_idx]  # [M,]
    b_expanded = b.unsqueeze(0).repeat(N, 1)
    psi_act = torch.tanh(torch.matmul(points_shifted, w.T) + b_expanded)  # [N, M]
    
    return torch.cat([psi_const, psi_act], dim=1)  # [N, M+1]

def laplacian_psi(points, region_idx, centers, weights, biases, device):
    """计算基函数的拉普拉斯"""
    # 1. 计算基函数值
    psi_vals = basis_psi(points, region_idx, centers, weights, biases, device)
    
    # 2. 分离常数项和激活项
    N = points.shape[0]
    psi_act = psi_vals[:, 1:]  # 提取激活项部分 (N, M)
    w = weights[region_idx]    # 获取当前区域的权重 (M, dim)
    
    # 3. 计算激活项的拉普拉斯
    one_minus_psi_sq = 1.0 - psi_act ** 2  # shape=(N, M)
    w_norm_sq = torch.sum(w ** 2, dim=1)  # shape=(M,)
    laplacian_act = -2.0 * psi_act * one_minus_psi_sq * w_norm_sq  # shape=(N, M)
    
    # 4. 常数项的拉普拉斯为 0
    laplacian_const = torch.zeros((N, 1), dtype=torch.float64, device=device)  # shape=(N, 1)
    
    # 5. 拼接结果
    laplacian_vals = torch.cat([laplacian_const, laplacian_act], dim=1)
    
    return laplacian_vals

def pde_source(points, device):
    """PDE源项f(x)=10"""
    return torch.ones_like(points[:,0], device=device) * 10

def u_x_boundary(points, device):
    """边界条件u=0"""
    return torch.zeros_like(points[:,0], device=device)

def qr_least_squares(global_A, global_b, device):
    """全局最小二乘求解"""
    # QR分解（full模式，确保稳定性）
    Q, R = torch.linalg.qr(global_A, mode="reduced")
    rhs = torch.matmul(Q.T, global_b)
    # 解上三角方程
    x = torch.linalg.solve_triangular(R, rhs, upper=True)
    return x.squeeze()

def find_max_residual_point(internal_pts, alphas, centers, weights, biases, config, device):
    """
    找到内部点中残差最大的点（作为下一个子区域的圆心）
    返回：
        max_residual_pt (torch.Tensor): 最大残差点
        avg_residual (float): 平均残差
    """
    # 计算每个点的残差：|α∇²ψ·α - f|
    alpha_pts = alpha_x(internal_pts).reshape(-1, 1)
    laplacian_vals = alpha_pts * laplacian_psi(internal_pts, 0, centers, weights, biases, device)
    f_vals = - pde_source(internal_pts, device).unsqueeze(1)
    residuals = torch.abs(torch.matmul(laplacian_vals, alphas[0].unsqueeze(1)) - f_vals)
    
    # 找最大残差点与平均残差
    max_residual, max_idx = torch.max(residuals, dim=0)
    avg_residual = torch.mean(residuals).item()
    
    return internal_pts[max_idx], avg_residual

def domain_decomposition(center, original_internal, original_boundary, config, device):
    """
    区域分解：生成新子区域的配置点，并更新原区域的配置点
    返回：
        sub_internal: 子区域内部点
        sub_boundary: 子区域边界点
        sub_interface: 子区域交界点
        new_original_internal: 原区域剩余内部点
        new_original_boundary: 原区域剩余边界点
    """
    r = config["subregion_radius"]
    sub_internal_num = config["subregion_internal_pts"]
    sub_interface_num = config["subregion_interface_pts"]
    
    # 1. 生成子区域内部点（球内）
    sub_internal = sample_sphere_interior(center, r, sub_internal_num, device, cube_constraint=True)
    
    # 2. 生成子区域交界点（球面）
    sub_interface = sample_sphere_surface(center, r, sub_interface_num, device, cube_constraint=True)
    
    # 3. 分离原区域边界点中属于子区域的部分
    dist_original_boundary = torch.norm(original_boundary - center, dim=1)
    sub_boundary = original_boundary[dist_original_boundary <= r]
    new_original_boundary = original_boundary[dist_original_boundary > r]
    
    # 4. 原区域剩余内部点（排除子区域内的点）
    dist_original_internal = torch.norm(original_internal - center, dim=1)
    new_original_internal = original_internal[dist_original_internal > r]
    
    return sub_internal, sub_boundary, sub_interface, new_original_internal, new_original_boundary

# ============================== 5. 自适应求解主逻辑 ==============================
def solve_pde_adaptive(config, device):
    """
    自适应区域分解求解3D PDE：基于残差迭代确定子区域圆心和区域
    """
    # -------------------------- 1. 初始化 --------------------------
    num_sub_max = config["num_sub_regions"]
    tol = config["residual_tol"]
    max_iter = config["max_iter"]
    
    # 生成主区域初始配置点
    main_internal = get_omega_points(config["init_internal_pts"], device)
    main_boundary = get_cube_boundary_points(config["init_boundary_pts"], device)
    
    # 初始化基函数参数
    weights, biases, alphas = init_basis_params(config, device)
    
    # 区域容器初始化
    region_centers = torch.zeros((num_sub_max + 1, 3), device=device)  # 0:主区域，1~8:子区域
    region_centers[0] = torch.tensor([0.5, 0.5, 0.5], device=device)  # 主区域中心固定
    region_internal = [main_internal] + [torch.empty(0, 3, device=device) for _ in range(num_sub_max)]
    region_boundary = [main_boundary] + [torch.empty(0, 3, device=device) for _ in range(num_sub_max)]
    region_interface = [torch.empty(0, 3, device=device) for _ in range(num_sub_max + 1)]
    
    current_K = 0  # 当前子区域数
    avg_residual = float('inf')
    
    # 加载验证数据
    test_point = np.loadtxt("3D_darcy_adam_prediction.txt")
    verify_points = torch.from_numpy(test_point[:, 0:3]).to(device)
    analytical_sol = torch.from_numpy(np.loadtxt("pred_u_FE_300.txt")).to(device)
    global_approx = torch.zeros_like(analytical_sol, device=device)
    
    # -------------------------- 2. 初始主区域求解 --------------------------
    print("\n=== 初始主区域求解 ===")
    # 构造初始主区域的线性系统
    A_boundary = basis_psi(region_boundary[0], 0, region_centers, weights, biases, device)
    b_boundary = u_x_boundary(region_boundary[0], device).unsqueeze(1)
    
    alpha_pts = alpha_x(region_internal[0]).reshape(-1, 1)
    A_internal = alpha_pts * laplacian_psi(region_internal[0], 0, region_centers, weights, biases, device)
    b_internal = -pde_source(region_internal[0], device).unsqueeze(1)
    
    # 拼接并求解
    A_init = torch.cat([1000*A_boundary, 10*A_internal], dim=0)
    b_init = torch.cat([1000*b_boundary, 10*b_internal], dim=0)
    alphas[0] = qr_least_squares(A_init, b_init, device)
    
    # 计算初始误差
    max_res_pt, avg_residual = find_max_residual_point(
        region_internal[0], alphas, region_centers, weights, biases, config, device
    )
    region_centers[1] = max_res_pt  # 首个子区域圆心为最大残差点
    
    # 初始全局误差
    psi_val = basis_psi(verify_points, 0, region_centers, weights, biases, device)
    global_approx = torch.matmul(psi_val, alphas[0].unsqueeze(1)).squeeze()
    global_err = torch.sqrt(torch.sum((global_approx - analytical_sol)**2) / torch.sum(analytical_sol**2)).item()
    
    print(f"初始全局相对L2误差：{global_err:.6f}")
    print(f"初始平均残差：{avg_residual:.6f}，首个子区域圆心：{max_res_pt}")
    
    # -------------------------- 3. 自适应迭代分解 --------------------------
    print("\n=== 开始自适应区域分解迭代 ===")
    while avg_residual > tol and current_K < max_iter:
        current_K += 1
        print(f"\n--- 迭代 {current_K}/{max_iter}：分解第 {current_K} 个子区域 ---")
        
        # 3.1 区域分解：生成新子区域，更新主区域
        sub_center = region_centers[current_K]
        sub_internal, sub_boundary, sub_interface, new_main_internal, new_main_boundary = domain_decomposition(
            sub_center, region_internal[0], region_boundary[0], config, device
        )
        
        # 更新区域配置点
        region_internal[current_K] = sub_internal
        region_boundary[current_K] = sub_boundary
        region_interface[current_K] = sub_interface
        region_internal[0] = new_main_internal
        region_boundary[0] = new_main_boundary
        
        print(f"子区域 {current_K}：内部点{len(sub_internal)}，边界点{len(sub_boundary)}，交界点{len(sub_interface)}")
        print(f"主区域剩余：内部点{len(new_main_internal)}，边界点{len(new_main_boundary)}")
        
        # 3.2 重新求解主区域系数（配置点已更新）
        print("重新求解主区域系数...")
        A_boundary_new = basis_psi(region_boundary[0], 0, region_centers, weights, biases, device)
        b_boundary_new = u_x_boundary(region_boundary[0], device).unsqueeze(1)
        
        alpha_new = alpha_x(region_internal[0]).reshape(-1, 1)
        A_internal_new = alpha_new * laplacian_psi(region_internal[0], 0, region_centers, weights, biases, device)
        b_internal_new = -pde_source(region_internal[0], device).unsqueeze(1)
        
        A_main = torch.cat([1000*A_boundary_new, 10*A_internal_new], dim=0)
        b_main = torch.cat([1000*b_boundary_new, 10*b_internal_new], dim=0)
        alphas[0] = qr_least_squares(A_main, b_main, device)
        
        # 3.3 求解新子区域系数（权重缩放优化）
        print("求解子区域系数（权重缩放优化）...")
        s_candidates = range(0, 10)
        best_loss = float('inf')
        best_alpha = None
        best_weight = None
        
        for s in s_candidates:
            # 缩放权重
            scaled_weight = torch.normal(0, s if s > 0 else 1, 
                                       size=(config["sub_basis_num"], 3), 
                                       device=device)
            weights[current_K] = scaled_weight
            
            # 构造子区域线性系统
            # 交界点连续性条件
            A_interface = basis_psi(sub_interface, current_K, region_centers, weights, biases, device)
            b_interface = torch.matmul(
                basis_psi(sub_interface, 0, region_centers, weights, biases, device),
                alphas[0].unsqueeze(1)
            )
            
            # 内部PDE条件
            alpha_sub = alpha_x(sub_internal).reshape(-1, 1)
            A_sub_internal = alpha_sub * laplacian_psi(sub_internal, current_K, region_centers, weights, biases, device)
            b_sub_internal = -pde_source(sub_internal, device).unsqueeze(1)
            
            # 边界条件
            A_sub_boundary = basis_psi(sub_boundary, current_K, region_centers, weights, biases, device)
            b_sub_boundary = u_x_boundary(sub_boundary, device).unsqueeze(1)
            
            # 拼接并求解
            A_sub = torch.cat([100*A_interface, A_sub_internal, 1000*A_sub_boundary], dim=0)
            b_sub = torch.cat([100*b_interface, b_sub_internal, 1000*b_sub_boundary], dim=0)
            alpha_sub = qr_least_squares(A_sub, b_sub, device)
            
            # 计算损失
            loss = torch.sum((torch.matmul(A_sub, alpha_sub.unsqueeze(1)) - b_sub)**2).item()
            
            # 更新最优解
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha_sub
                best_weight = scaled_weight
        
        # 保存最优参数
        weights[current_K] = best_weight
        alphas[current_K] = best_alpha
        print(f"子区域 {current_K} 最优损失：{best_loss:.6f}")
        
        # 3.4 寻找下一个子区域圆心
        if current_K < max_iter:
            max_res_pt, avg_residual = find_max_residual_point(
                region_internal[0], alphas, region_centers, weights, biases, config, device
            )
            region_centers[current_K + 1] = max_res_pt
            print(f"下一个子区域圆心：{max_res_pt}，当前平均残差：{avg_residual:.6f}")
        
        # 3.5 计算全局误差
        print("计算全局相对L2误差...")
        # 批量判断验证点所属区域
        pt_expanded = verify_points.unsqueeze(1)  # [N, 1, 3]
        center_expanded = region_centers[1:current_K+1].unsqueeze(0)  # [1, K, 3]
        distances = torch.norm(pt_expanded - center_expanded, dim=2)  # [N, K]
        in_subregion = distances <= config["subregion_radius"]  # [N, K]
        
        # 逐点计算近似解
        for idx in range(len(verify_points)):
            # 优先匹配子区域
            sub_matches = torch.where(in_subregion[idx])[0]
            if len(sub_matches) > 0:
                region_idx = sub_matches[0] + 1
            else:
                region_idx = 0
            
            psi_val = basis_psi(verify_points[idx:idx+1], region_idx, region_centers, weights, biases, device)
            global_approx[idx] = torch.matmul(psi_val, alphas[region_idx].unsqueeze(1)).squeeze()
        
        # 计算相对L2误差
        global_err = torch.sqrt(torch.sum((global_approx - analytical_sol)**2) / torch.sum(analytical_sol**2)).item()
        print(f"当前全局相对L2误差：{global_err:.6f}")
    
    # -------------------------- 4. 结果可视化 --------------------------
    print("\n=== 绘制3D预测结果 ===")
    # plot_3d_prediction(verify_points, global_approx)
    plot_z01_slice(verify_points, global_approx)
    # -------------------------- 5. 整理结果 --------------------------
    results = {
        "alphas": alphas,
        "weights": weights,
        "biases": biases,
        "region_centers": region_centers[:current_K+1],
        "region_internal": region_internal[:current_K+1],
        "region_boundary": region_boundary[:current_K+1],
        "region_interface": region_interface[:current_K+1],
        "global_relative_err": global_err,
        "global_approx": global_approx,
        "analytical_sol": analytical_sol,
        "final_sub_regions": current_K
    }
    
    return results

def plot_3d_prediction(points, pred_vals):
    """绘制3D预测值分布图"""
    x = points.detach().cpu().numpy()[:, 0]
    y = points.detach().cpu().numpy()[:, 1]
    z = points.detach().cpu().numpy()[:, 2]
    pred = pred_vals.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=pred, s=5, cmap='viridis', alpha=0.7)
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Predicted Value', fontsize=12)
    
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)
    ax.set_zlabel('Z Coordinate', fontsize=10)
    ax.set_title('3D Distribution of Predicted Values (Adaptive Regions)', fontsize=14, pad=20)
    ax.view_init(elev=20, azim=45)
    
    plt.show()

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
# ============================== 6. 程序入口 ==============================
if __name__ == "__main__":
    # 1. 配置与设备
    config, device = set_global_config()
    
    # 2. 记录总运行时间
    start_total = time.perf_counter()
    
    # 3. 自适应求解PDE
    results = solve_pde_adaptive(config, device)
    
    # 4. 输出总结
    total_time = time.perf_counter() - start_total
    print("\n=== 程序执行完成 ===")
    print(f"总运行时间：{total_time:.6f}秒")
    print(f"最终子区域数量：{results['final_sub_regions']}")