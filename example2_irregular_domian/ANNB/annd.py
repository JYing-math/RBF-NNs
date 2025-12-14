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
        "weight_std": 10,            # 基函数权重正态分布标准差
        "bias_range": 1,            # 基函数偏置均匀分布范围（[-b, b]）
        "tolerance": 1e-4,          # 迭代终止阈值（平均残差<tol停止）
        "init_internal_pts": 8000,    # 初始区域内部配置点
        "init_boundary_pts": 5000,    # 初始区域边界配置点
        "subregion_pts": 8000,        # 子区域配置点
        "subregion_radius": 0.1,   # 子区域球体半径
        "max_iter": 1             # 最大区域分解迭代次数
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
    在球体内部（严格小于半径）随机取点，可选限制在 [-1,1]^3 \ [0,1]×[0,1]×[-1,1] 区域内。

    Args:
        center (torch.Tensor): 球心坐标，shape=(3,)
        r (float): 球体半径
        num_points (int): 需采样的点数
        device (torch.device): 计算设备（cpu/cuda）
        cube_constraint (bool): 是否限制点在 [-1,1]^3 \ [0,1]×[0,1]×[-1,1] 范围内（默认True）

    Returns:
        torch.Tensor: 采样点坐标，shape=(num_points, 3)
    """
    sampled_points = []
    # 将 r 转换为 Tensor（同设备、同类型）
    r_tensor = torch.tensor(r, device=device, dtype=center.dtype)

    while len(sampled_points) < num_points:
        # 1. 生成三维标准正态分布向量（方向均匀随机）
        num_candidates = num_points * 4  # 一次生成更多候选点以提高效率
        norm_vec = torch.randn((num_candidates, 3), device=device, dtype=center.dtype)
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

        # 5. 筛选：严格在球内（距离<r）+ 可选区域约束
        dist_to_center = torch.norm(candidate_points - center, dim=1)
        in_sphere = dist_to_center < (r_tensor - 1e-6)  # 用 Tensor 计算，避免类型问题

        if cube_constraint:
            # 修正1: 正确判断点是否在大立方体 [-1, 1]^3 内
            # 使用 torch.all(..., dim=1) 检查每个点的所有坐标是否都满足条件
            in_large_cube = torch.all(candidate_points >= -0.2, dim=1) & torch.all(candidate_points <= 0.2, dim=1)

            # 修正2: 正确判断点是否在挖去的小立方体 [0, 1]×[0, 1]×[-1, 1] 内
            # 使用 & 进行按元素逻辑与
            in_small_cube = (candidate_points[:, 0] >= 0.0) & (candidate_points[:, 0] <= 0.2) & \
                            (candidate_points[:, 1] >= 0.0) & (candidate_points[:, 1] <= 0.2)
            
            # 修正3: 正确计算不在小立方体内的点
            # 使用 ~ 进行按元素逻辑非
            not_in_small_cube = ~in_small_cube

            # 最终的有效点是满足所有条件的点
            valid_mask = in_sphere & in_large_cube & not_in_small_cube
            valid_points = candidate_points[valid_mask]
        # 收集有效点，直到满足数量
        sampled_points.extend(valid_points)

    # 截取指定数量的点并返回
    return torch.stack(sampled_points[:num_points], dim=0)

def generate_z0_uniform_grid(config, device, weights, biases, alphas, region_centers, current_K, 
                             x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), num_points=200):
    # -------------------------- 1. 生成均匀网格点 --------------------------
    # 生成x/y均匀采样点
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    y = torch.linspace(y_range[0], y_range[1], num_points, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')  # (num_points, num_points)
    Z = torch.zeros_like(X)  # z固定为0
    
    # 展平为(N, 3)的坐标张量
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    # 筛选有效区域（排除[0,0.2]×[0,0.2]）
    valid_mask = ~((grid_points[:, 0] >= 0) & (grid_points[:, 0] <= 0.2) & 
                   (grid_points[:, 1] >= 0) & (grid_points[:, 1] <= 0.2))
    grid_points = grid_points[valid_mask]
    
    # -------------------------- 2. 计算真实值(truth_u) --------------------------
    truth_u = u_x(grid_points).squeeze()
    
    # -------------------------- 3. 计算预测值(pred_u) --------------------------
    pred_u = torch.zeros_like(truth_u, device=device)
    subregion_radius = config["subregion_radius"]
    
    # 遍历每个点，判断所属区域并计算预测值
    for idx, pt in enumerate(grid_points):
        # 默认属于初始区域(0)
        region_idx = 0
        # 检查是否属于子区域（优先匹配）
        for k in range(1, current_K + 1):
            if torch.norm(pt - region_centers[k]) <= subregion_radius:
                region_idx = k
                break
        
        # 计算对应区域的基函数值并预测
        psi_val = basis_psi(pt.unsqueeze(0), region_idx, region_centers, weights, biases, device)
        pred_u[idx] = torch.matmul(psi_val, alphas[region_idx].unsqueeze(1)).squeeze()
    
    # -------------------------- 4. 准备网格数据用于绘图 --------------------------
    x_grid = X.cpu().numpy()
    y_grid = Y.cpu().numpy()
    
    return grid_points, pred_u, truth_u, x_grid, y_grid


def plot_z0_uniform_results(grid_points, pred_u, truth_u, x_grid, y_grid, num_points=200):
    """
    可视化z=0平面均匀点的预测结果（二维热力图）
    """
    # -------------------------- 1. 重构网格数据 --------------------------
    # 创建全尺寸数组（包含无效区域）
    pred_full = np.full((num_points, num_points), np.nan)
    truth_full = np.full((num_points, num_points), np.nan)
    error_full = np.full((num_points, num_points), np.nan)
    
    # 计算原始网格索引
    x_flat = grid_points[:, 0].cpu().numpy()
    y_flat = grid_points[:, 1].cpu().numpy()
    x_idx = np.round((x_flat - (-0.2)) / (0.4 / (num_points - 1))).astype(int)
    y_idx = np.round((y_flat - (-0.2)) / (0.4 / (num_points - 1))).astype(int)
    
    # 填充有效区域的预测值、真实值、误差
    pred_full[x_idx, y_idx] = pred_u.cpu().numpy()
    truth_full[x_idx, y_idx] = truth_u.cpu().numpy()
    error_full[x_idx, y_idx] = np.abs(pred_full[x_idx, y_idx] - truth_full[x_idx, y_idx])
    
    # -------------------------- 2. 绘制热力图 --------------------------
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    """
    # 真实值热力图
    im1 = axes[0].imshow(truth_full.T, extent=[-0.2, 0.2, -0.2, 0.2], origin='lower', 
                         cmap='viridis', aspect='auto')
    axes[0].set_title('Truth Value (z=0)', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # 预测值热力图
    im2 = axes[1].imshow(pred_full.T, extent=[-0.2, 0.2, -0.2, 0.2], origin='lower', 
                         cmap='viridis', aspect='auto')
    axes[1].set_title('Predicted Value (z=0)', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    """
    # 误差热力图
    im3 = axes.imshow(error_full.T, extent=[-0.2, 0.2, -0.2, 0.2], origin='lower', 
                         cmap='Reds', aspect='auto', vmin=0)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.colorbar(im3, ax=axes)
    
    plt.tight_layout()
    plt.show()
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
            in_large_cube = torch.all(candidate_points >= -0.2, dim=1) & torch.all(candidate_points <= 0.2, dim=1)

            # 修正2: 正确判断点是否在挖去的小立方体 [0, 1]×[0, 1]×[-1, 1] 内
            # 使用 & 进行按元素逻辑与
            in_small_cube = (candidate_points[:, 0] >= 0.0) & (candidate_points[:, 0] <= 0.2) & \
                            (candidate_points[:, 1] >= 0.0) & (candidate_points[:, 1] <= 0.2)
            
            # 修正3: 正确计算不在小立方体内的点
            # 使用 ~ 进行按元素逻辑非
            not_in_small_cube = ~in_small_cube
            valid_points = candidate_points[on_surface & in_large_cube & not_in_small_cube]
        # 收集有效点，直到满足数量
        sampled_points.extend(valid_points)
    
    # 截取指定数量的点并返回
    return torch.stack(sampled_points[:num_points], dim=0)

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
    internal_pts = get_omega_points(N, x_left=-0.2, y_left=-0.2, z_left=-0.2,
                        x_right=0.2, y_right=0.2, z_right=0.2).to(device)
    boundary_pts = get_boundary_points(N_g, x_left=-0.2, y_left=-0.2, z_left=-0.2,
                           x_right=0.2, y_right=0.2, z_right=0.2).to(device)
    
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
    N = points.shape[0]
    
    # m=0：常数基函数（全1）
    psi_const = torch.ones((N, 1), dtype=torch.float64, device=device)
    
    # m>0：双曲正切基函数 tanh(w·x + b)
    w = weights[region_idx]  # [M, dim]
    b = biases[region_idx]  # [M,]
    b_expanded = b.unsqueeze(0).repeat(N, 1)  # [N, M]
    psi_act = torch.tanh(torch.matmul(points, w.T) + b_expanded)  # [N, M]
    
    # 拼接常数项与激活项
    return torch.cat([psi_const, psi_act], dim=1)


def laplacian_psi(points, region_idx, centers, weights, biases, config, device):
    """
    计算基函数 basis_psi 自身的拉普拉斯算子 ∇²ψ(x)。

    参数:
        points (torch.Tensor): 输入点，shape=(N, dim)
        region_idx (int): 区域索引
        centers (torch.Tensor): 各区域圆心，shape=(max_regions+1, dim)
        weights (list): 基函数权重，list of torch.Tensor, 每个元素 shape=(M, dim)
        biases (list): 基函数偏置，list of torch.Tensor, 每个元素 shape=(M,)
        device (torch.device): 计算设备

    返回:
        laplacian_vals (torch.Tensor): 拉普拉斯算子的值，shape=(N, M+1)
    """
    # 1. 首先计算基函数本身的值，这是后续计算的基础
    psi_vals = basis_psi(points, region_idx, centers, weights, biases, device)
    
    # 2. 分离常数项和激活项
    N = points.shape[0]
    psi_act = psi_vals[:, 1:]  # 提取激活项部分 (N, M)
    w = weights[region_idx]    # 获取当前区域的权重 (M, dim)
    
    # 3. 计算激活项的拉普拉斯
    # 计算 1 - psi^2 (tanh的一阶导数)
    one_minus_psi_sq = 1.0 - psi_act ** 2  # shape=(N, M)
    
    # 计算每个权重向量的 L2 范数的平方 ||w_m||^2
    w_norm_sq = torch.sum(w ** 2, dim=1)  # shape=(M,)
    # 根据公式计算拉普拉斯: -2 * psi * (1 - psi^2) * ||w_m||^2
    # 利用广播机制 (Broadcasting) 高效计算
    laplacian_act = -2.0 * psi_act * one_minus_psi_sq * w_norm_sq  # shape=(N, M)
    
    # 4. 常数项的拉普拉斯为 0
    laplacian_const = torch.zeros((N, 1), dtype=torch.float64, device=device)  # shape=(N, 1)
    
    # 5. 拼接结果，保持与 basis_psi 输出相同的维度 (N, M+1)
    laplacian_vals = torch.cat([laplacian_const, laplacian_act], dim=1)
    
    return laplacian_vals


def pde_source(points, config, device):
    """
    三维Poisson方程中的源项 f(x, y, z)。
    它是通过对 u(x, y, z) 求拉普拉斯算子得到的，即 f = -Δu。
    """
    r_squared = torch.sum(torch.pow(points, 2), 1)
    r = torch.pow(r_squared, -2/3)
    # 为了数值稳定性，避免 r=0 时出现无穷大
    eps = 1e-8
    r = torch.clamp(r, min=eps)
    fx = -(10/9)*r
    return fx.unsqueeze(1)

def u_x(x):
    """
    三维区域上的函数 u(x, y, z)。
    拓展自二维版本 u(x, y) = (x^2 + y^2)^(1/3)。
    """
    r_squared = torch.sum(torch.pow(x, 2), 1)
    ux = torch.pow(r_squared, 1/3)
    return ux.unsqueeze(1)

def u_x_boundary(points):
    """
    三维区域边界上的函数 g(x, y, z)。
    通常 g 是 u 在边界上的限制，即 g(x) = u(x) for x ∈ ∂Ω。
    """
    return u_x(points)

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
    laplacian_vals = - laplacian_psi(internal_pts, 0, centers, weights, biases, config, device)
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
    N2 = config["init_boundary_pts"]
    sub_internal = sample_sphere_interior(center, r, N1, device, cube_constraint=True)
    
    # -------------------------- 2. 生成子区域交界点（球体表面） --------------------------
    sub_interface = sample_sphere_surface(center, r, N2, device, cube_constraint=True)
    
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
    region_centers[0] = - 0.1  # 利用广播机制，dim维都会被设为0.
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
    b_boundary = u_x_boundary(region_boundary[0])
    
    # 内部PDE条件：∇²ψ·α = f
    A_internal = - laplacian_psi(region_internal[0], 0, region_centers, weights, biases, config, device)
    b_internal = pde_source(region_internal[0], config, device)
    
    # 拼接A和b（合并边界与内部条件）
    A_init = torch.cat([1000*A_boundary, A_internal], dim=0)
    b_init = torch.cat([1000*b_boundary, b_internal], dim=0)
    print(A_init.shape, b_init.shape)
    # QR分解求解初始区域系数
    alphas[0] = qr_least_squares(A_init, b_init, device)
    # 计算初始区域残差与最大残差点（首个子区域圆心）
    max_res_pt, avg_residual = find_max_residual_point(
        region_internal[0], alphas, region_centers, weights, biases, config, device
    )
    print(f"初始区域最大残差点：", max_res_pt.shape)
    region_centers[1] = max_res_pt  # 首个子区域圆心为最大残差点
    
    x1 = torch.linspace(-0.2, 0.2, 30)
    x2 = torch.linspace(-0.2, 0.2, 30)
    x3 = torch.linspace(-0.2, 0.2, 30)
    X1, X2, X3 = torch.meshgrid(x1, x2, x3, indexing='ij') # 建议加上 indexing='ij' 明确索引方式

    # 将网格点展平并拼接成 (N, 3) 的形状
    Z = torch.cat((X1.flatten()[:, None], X2.flatten()[:, None], X3.flatten()[:, None]), dim=1)
    # 2. 定义筛选条件并创建掩码
    # 我们要保留的是：x1不在[0,1] 或者 x2不在[0,1] 的点
    # 等价于：不满足 (x1在[0,1] 并且 x2在[0,1]) 的点
    mask = ~((Z[:, 0] >= 0) & (Z[:, 0] <= 0.2) & (Z[:, 1] >= 0) & (Z[:, 1] <= 0.2))
    # 3. 应用掩码，筛选出有效点
    verify_points = Z[mask].to(device)
    truth_u = u_x(verify_points).squeeze()
    psi_val = basis_psi(verify_points, 0, region_centers, weights, biases, device)
    pred_u = torch.matmul(psi_val, alphas[0].unsqueeze(1)).squeeze()
    print( pred_u.shape, truth_u.shape)
    # 计算全局相对L2误差
    l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
                    torch.mean(torch.pow(truth_u, 2)))
    print(f"初次全局相对L2误差：{l2_loss.item():.6f}")
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
        b_boundary_new = u_x_boundary(region_boundary[0])
        A_internal_new = - laplacian_psi(region_internal[0], 0, region_centers, weights, biases, config, device)
        b_internal_new = pde_source(region_internal[0], config, device)
        
        # 边界条件加权（提升边界拟合优先级）
        A_original_update = torch.cat([1000 * A_boundary_new, A_internal_new], dim=0)
        b_original_update = torch.cat([1000 * b_boundary_new, b_internal_new], dim=0)
        
        # 求解更新后的初始区域系数
        alphas[0] = qr_least_squares(A_original_update, b_original_update, device)
        
        # 3. 求解新子区域系数（遍历权重缩放系数s，选损失最小的）
        print(f"求解第 {current_K} 个子区域系数（内部点：{sub_internal.shape[0]}）")
        # 权重缩放系数s的候选（此处取s=3，可扩展更多候选）
        s_candidates = range(5, 10)
        best_loss = float('inf')
        best_s = 6    
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
            A_sub_internal = - laplacian_psi(sub_internal, current_K, region_centers, weights, biases, config, device)
            b_sub_internal = pde_source(sub_internal, config, device)
            
            # 子区域边界条件：ψ·α = g
            A_sub_boundary = basis_psi(sub_boundary, current_K, region_centers, weights, biases, device)
            b_sub_boundary = u_x_boundary(sub_boundary)
            # 拼接A和b
            A_sub = torch.cat([100*A_interface, A_sub_internal, 1000*A_sub_boundary], dim=0)
            b_sub = torch.cat([100*b_interface, b_sub_internal, 1000*b_sub_boundary], dim=0)
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
            pred_u[idx] = torch.matmul(psi_val, alphas[region_idx].unsqueeze(1)).squeeze()
        # 计算全局相对L2误差
        # 拆分 verify_points 的 x/y/z 坐标
        l2_loss = torch.sqrt(torch.mean(torch.pow(pred_u - truth_u, 2))) / torch.sqrt(
                        torch.mean(torch.pow(truth_u, 2)))
        print(f"全局相对L2误差：{l2_loss.item():.6f}")
            # 记录结束时间
        end_time = time.perf_counter()

        # 计算并打印运行时间（保留6位小数）
        print(f"程序运行时间：{end_time - start_time:.6f} 秒")

    print(region_centers)
    grid_points, pred_u_z0, truth_u_z0, x_grid, y_grid = generate_z0_uniform_grid(config, device, weights, biases, alphas, region_centers, current_K, 
                             x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), num_points=200)
    plot_z0_uniform_results(grid_points, pred_u_z0, truth_u_z0, x_grid, y_grid)
    x = verify_points.cpu().numpy()[:, 0]  # 所有点的x坐标
    y = verify_points.cpu().numpy()[:, 1]  # 所有点的y坐标
    z = verify_points.cpu().numpy()[:, 2]  # 所有点的z坐标
    # 预测值（global_approx）作为颜色映射依据

    # 2. 创建 3D 绘图对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    approx_vals = pred_u.cpu().numpy()
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
        "global_relative_err": l2_loss.item(),  # 全局相对误差
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
    h = 1e-4
    results = solve_pde_annb(config, device)
    
    print("\n=== 程序执行完成 ===")