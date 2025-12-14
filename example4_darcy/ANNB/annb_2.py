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
        "main_basis_num": 6000,    # 主区域（region_centers[0]）基函数数
        "sub_basis_num": 100,      # 子区域（region_centers[1:8]）基函数数
        "num_sub_regions": 8,      # 子区域数量（1-8共8个）
        "main_weight_std": 5,      # 主区域基函数权重正态分布标准差
        "sub_weight_std": 5,       # 子区域基函数权重正态分布标准差（与主区域不同）
        "bias_range": 1,           # 基函数偏置均匀分布范围（[-b, b]）
        "subregion_radius": 0.2,   # 子区域球体半径
        "init_internal_pts": 6000, # 主区域初始内部点（后续拆分）
        "init_boundary_pts": 3000, # 初始立方体边界点（后续拆分）
        "subregion_internal_pts": 300, # 每个子区域内部点数量
        "subregion_interface_pts": 100 # 每个子区域交界点数量
    }
    
    # 自动选择计算设备（GPU优先，无则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备：{device}")
    
    return config, device

# ============================== 3. 初始化工具函数 ==============================
def init_basis_params(config, device):
    """
    分区域初始化基函数参数：主区域5000基，子区域各500基
    返回：
        weights (list): 各区域基函数权重 [主区域, 子区域1, ..., 子区域8]
        biases (list): 各区域基函数偏置 [主区域, 子区域1, ..., 子区域8]
        alphas (list): 基函数系数容器 [主区域, 子区域1, ..., 子区域8]
    """
    main_M = config["main_basis_num"]  # 主区域基函数数（不含常数项）
    sub_M = config["sub_basis_num"]    # 子区域基函数数（不含常数项）
    dim = config["dim"]
    main_R = config["main_weight_std"] # 主区域权重标准差
    sub_R = config["sub_weight_std"]   # 子区域权重标准差（与主区域不同）
    b = config["bias_range"]
    num_sub = config["num_sub_regions"]
    
    # 初始化权重：主区域main_M个，子区域各sub_M个
    weights = []
    # 主区域权重 [main_M, dim] - 使用主区域标准差
    weights.append(torch.normal(0, main_R, size=(main_M, dim), dtype=torch.float64, device=device))
    # 子区域权重 [sub_M, dim] × 8 - 使用子区域标准差
    for _ in range(num_sub):
        weights.append(torch.normal(0, sub_R, size=(sub_M, dim), dtype=torch.float64, device=device))
    
    # 初始化偏置：主区域main_M个，子区域各sub_M个
    biases = []
    # 主区域偏置 [main_M,]
    biases.append(torch.rand((main_M,), dtype=torch.float64, device=device) * 2 * b - b)
    # 子区域偏置 [sub_M,] × 8
    for _ in range(num_sub):
        biases.append(torch.rand((sub_M,), dtype=torch.float64, device=device) * 2 * b - b)
    
    # 初始化系数：主区域(main_M+1)个，子区域各(sub_M+1)个（+1为常数项）
    alphas = []
    alphas.append(torch.zeros((main_M + 1,), dtype=torch.float64, device=device))
    for _ in range(num_sub):
        alphas.append(torch.zeros((sub_M + 1,), dtype=torch.float64, device=device))
    
    return weights, biases, alphas

def sample_sphere_interior(center, r, num_points, device, cube_constraint=True):
    """球体内部随机取点（复用原逻辑）"""
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
    """球体表面随机取点（复用原逻辑）"""
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
    """生成立方体内部点（添加device参数）"""
    x_coords = torch.rand(num, 1, device=device) * (x_right - x_left) + x_left
    y_coords = torch.rand(num, 1, device=device) * (y_right - y_left) + y_left
    z_coords = torch.rand(num, 1, device=device) * (z_right - z_left) + z_left
    return torch.cat([x_coords, y_coords, z_coords], dim=1)

def get_cube_boundary_points(num, device):
    """生成立方体边界点（添加device参数）"""
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
    """复用原alpha(x)计算逻辑"""
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
    """计算指定区域的基函数值（复用原逻辑，适配不同基函数数量）"""
    points_shifted = points - centers[region_idx]
    N = points_shifted.shape[0]
    
    # 常数项（m=0）
    psi_const = torch.ones((N, 1), dtype=torch.float64, device=device)
    
    # 激活项（m>0）：数量为该区域的基函数数（不含常数项）
    M = weights[region_idx].shape[0]
    w = weights[region_idx]  # [M, dim]
    b = biases[region_idx]  # [M,]
    b_expanded = b.unsqueeze(0).repeat(N, 1)
    psi_act = torch.tanh(torch.matmul(points_shifted, w.T) + b_expanded)  # [N, M]
    
    return torch.cat([psi_const, psi_act], dim=1)  # [N, M+1]

def laplacian_psi(points, region_idx, centers, weights, biases, device):
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

def pde_source(points, device):
    """PDE源项f(x)=10（复用原逻辑）"""
    return torch.ones_like(points[:,0], device=device) * 10

def u_x_boundary(points, device):
    """边界条件u=0（添加device参数）"""
    return torch.zeros_like(points[:,0], device=device)

def qr_least_squares(global_A, global_b, device):
    """全局最小二乘求解（适配大矩阵）"""
    # QR分解（full模式，确保稳定性）
    Q, R = torch.linalg.qr(global_A, mode="reduced")
    rhs = torch.matmul(Q.T, global_b)
    # 解上三角方程
    x = torch.linalg.solve_triangular(R, rhs, upper=True)
    return x.squeeze()

# ============================== 5. 生成所有区域配置点 ==============================
def generate_all_regions(config, centers, device):
    """生成主区域和8个子区域的配置点（内部点、边界点、交界点）"""
    num_sub = config["num_sub_regions"]
    r = config["subregion_radius"]
    main_internal_num = config["init_internal_pts"]
    main_boundary_num = config["init_boundary_pts"]
    sub_internal_num = config["subregion_internal_pts"]
    sub_interface_num = config["subregion_interface_pts"]
    
    # 1. 生成主区域初始配置点（整个立方体）
    main_internal_all = get_omega_points(main_internal_num, device)
    main_boundary_all = get_cube_boundary_points(main_boundary_num, device)
    
    # 2. 子区域配置点容器
    sub_regions = []
    for k in range(num_sub):
        sub_center = centers[k+1]  # 子区域圆心（索引1-8）
        # 子区域内部点（球内）
        sub_internal = sample_sphere_interior(sub_center, r, sub_internal_num, device)
        # 子区域交界点（球面，与主区域交界）
        sub_interface = sample_sphere_surface(sub_center, r, sub_interface_num, device)
        # 子区域边界点（属于原立方体边界的部分）
        dist_to_center = torch.norm(main_boundary_all - sub_center, dim=1)
        sub_boundary = main_boundary_all[dist_to_center <= r]
        # 保存子区域配置点
        sub_regions.append({
            "internal": sub_internal,
            "boundary": sub_boundary,
            "interface": sub_interface,
            "center": sub_center
        })
    
    # 3. 更新主区域配置点（排除所有子区域的点）
    main_internal_mask = torch.ones(len(main_internal_all), dtype=torch.bool, device=device)
    main_boundary_mask = torch.ones(len(main_boundary_all), dtype=torch.bool, device=device)
    
    for k in range(num_sub):
        sub_center = centers[k+1]
        # 排除主区域内部点中属于子区域的部分
        dist_internal = torch.norm(main_internal_all - sub_center, dim=1)
        main_internal_mask = main_internal_mask & (dist_internal > r)
        # 排除主区域边界点中属于子区域的部分
        dist_boundary = torch.norm(main_boundary_all - sub_center, dim=1)
        main_boundary_mask = main_boundary_mask & (dist_boundary > r)
    
    main_internal = main_internal_all[main_internal_mask]
    main_boundary = main_boundary_all[main_boundary_mask]
    
    print(f"主区域：内部点{len(main_internal)}个，边界点{len(main_boundary)}个")
    for k in range(num_sub):
        print(f"子区域{k+1}：内部点{len(sub_regions[k]['internal'])}个，边界点{len(sub_regions[k]['boundary'])}个，交界点{len(sub_regions[k]['interface'])}个")
    
    return {
        "main": {"internal": main_internal, "boundary": main_boundary},
        "subs": sub_regions
    }

# ============================== 6. 构造全局线性系统 ==============================
def build_global_system(regions, centers, weights, biases, config, device):
    """整合所有区域的条件，构造全局A矩阵和b向量"""
    main_region = regions["main"]
    sub_regions = regions["subs"]
    num_sub = config["num_sub_regions"]
    total_coeffs = get_total_coeffs_num(config)  # 新增：获取总系数个数（9009）
    
    # 存储所有条件的A块和b块
    A_blocks = []
    b_blocks = []
    
    # -------------------------- 1. 主区域条件（关键修改）--------------------------
    # 1.1 主区域边界条件：ψ_main·α_main = 0 → 仅主区域系数列有值，其他列全0
    if len(main_region["boundary"]) > 0:
        psi_main_bound = basis_psi(main_region["boundary"], 0, centers, weights, biases, device)
        # 构造总列数的矩阵（9009列），仅主区域系数列（0-5000）赋值
        A_main_bound = torch.zeros((len(main_region["boundary"]), total_coeffs), dtype=torch.float64, device=device)
        A_main_bound[:, 0:get_coeffs_end_idx(-1, config)] = psi_main_bound  # 主区域系数列：0-5000
        b_main_bound = u_x_boundary(main_region["boundary"], device).unsqueeze(1)  # [N_bound, 1]
        A_blocks.append(1000*A_main_bound)
        b_blocks.append(1000*b_main_bound)
    
    # 1.2 主区域内部PDE条件：α(x)∇²ψ_main·α_main = f(x) → 仅主区域系数列有值
    if len(main_region["internal"]) > 0:
        alpha_main = alpha_x(main_region["internal"]).reshape(-1, 1)
        laplacian_main = laplacian_psi(main_region["internal"], 0, centers, weights, biases, device)

        # 构造总列数的矩阵（9009列），仅主区域系数列赋值
        A_main_internal = torch.zeros((len(main_region["internal"]), total_coeffs), dtype=torch.float64, device=device)
        A_main_internal[:, 0:get_coeffs_end_idx(-1, config)] = - alpha_main * laplacian_main  # 主区域系数列：0-5000
        b_main_internal = pde_source(main_region["internal"], device).unsqueeze(1)  # [N_int, 1]
        A_blocks.append(10*A_main_internal)
        b_blocks.append(10*b_main_internal)
    
    # -------------------------- 2. 子区域条件（无需修改）--------------------------
    for k in range(num_sub):
        sub = sub_regions[k]
        sub_idx = k + 1  # 子区域索引（1-8）
        
        # 2.1 子区域边界条件：ψ_sub·α_sub = 0
        if len(sub["boundary"]) > 0:
            psi_sub_bound = basis_psi(sub["boundary"], sub_idx, centers, weights, biases, device)
            A_sub_bound = torch.zeros((len(sub["boundary"]), total_coeffs), dtype=torch.float64, device=device)
            A_sub_bound[:, get_coeffs_start_idx(k, config):get_coeffs_end_idx(k, config)] = psi_sub_bound
            b_sub_bound = u_x_boundary(sub["boundary"], device).unsqueeze(1)
            A_blocks.append(1000*A_sub_bound)
            b_blocks.append(1000*b_sub_bound)
        
        # 2.2 子区域内部PDE条件：α(x)∇²ψ_sub·α_sub = f(x)
        if len(sub["internal"]) > 0:
            alpha_sub = alpha_x(sub["internal"]).reshape(-1, 1)
            laplacian_sub = laplacian_psi(sub["internal"], sub_idx, centers, weights, biases, device)
            A_sub_internal = torch.zeros((len(sub["internal"]), total_coeffs), dtype=torch.float64, device=device)
            A_sub_internal[:, get_coeffs_start_idx(k, config):get_coeffs_end_idx(k, config)] =  - alpha_sub * laplacian_sub
            b_sub_internal = pde_source(sub["internal"], device).unsqueeze(1)
            A_blocks.append(A_sub_internal)
            b_blocks.append(b_sub_internal)
        
        # 2.3 交界条件：ψ_main·α_main = ψ_sub·α_sub → ψ_main·α_main - ψ_sub·α_sub = 0
        if len(sub["interface"]) > 0:
            psi_main_interface = basis_psi(sub["interface"], 0, centers, weights, biases, device)
            psi_sub_interface = basis_psi(sub["interface"], sub_idx, centers, weights, biases, device)
            A_interface = torch.zeros((len(sub["interface"]), total_coeffs), dtype=torch.float64, device=device)
            A_interface[:, 0:get_coeffs_end_idx(-1, config)] = psi_main_interface  # 主区域系数列
            A_interface[:, get_coeffs_start_idx(k, config):get_coeffs_end_idx(k, config)] -= psi_sub_interface  # 子区域系数列（负号）
            b_interface = torch.zeros((len(sub["interface"]), 1), dtype=torch.float64, device=device)
            A_blocks.append(100*A_interface)
            b_blocks.append(100*b_interface)
    
    # -------------------------- 3. 拼接全局A和b --------------------------
    global_A = torch.cat(A_blocks, dim=0)
    global_b = torch.cat(b_blocks, dim=0)
    
    print(f"全局线性系统规模：A({global_A.shape[0]}, {global_A.shape[1]}), b({global_b.shape[0]}, 1)")
    return global_A, global_b

def get_total_coeffs_num(config):
    """计算总系数个数：主区域(main_M+1) + 8×子区域(sub_M+1)"""
    main_coeffs = config["main_basis_num"] + 1
    sub_coeffs_per = config["sub_basis_num"] + 1
    return main_coeffs + config["num_sub_regions"] * sub_coeffs_per

def get_coeffs_start_idx(sub_idx, config):
    """获取第k个子区域的系数起始索引（sub_idx=-1表示主区域）"""
    if sub_idx == -1:
        return 0
    main_coeffs = config["main_basis_num"] + 1
    sub_coeffs_per = config["sub_basis_num"] + 1
    return main_coeffs + sub_idx * sub_coeffs_per

def get_coeffs_end_idx(sub_idx, config):
    """获取第k个子区域的系数结束索引（sub_idx=-1表示主区域）"""
    if sub_idx == -1:
        return config["main_basis_num"] + 1
    return get_coeffs_start_idx(sub_idx, config) + (config["sub_basis_num"] + 1)

# ============================== 7. 计算全局L2误差 ==============================
def compute_global_l2_error(verify_points, analytical_sol, centers, weights, biases, alphas, config, device):
    """计算全局相对L2误差（批量优化版）"""
    num_points = len(verify_points)
    global_approx = torch.zeros(num_points, dtype=torch.float64, device=device)
    r = config["subregion_radius"]
    num_sub = config["num_sub_regions"]
    
    # ===================== 优化1：批量判断所有点的所属区域 =====================
    # 1.1 计算所有点到每个子区域中心的距离 (num_points, num_sub)
    sub_centers = centers[1:num_sub+1]  # 提取所有子区域中心 (num_sub, 3)
    # 广播计算距离：(num_points, 1, 3) - (1, num_sub, 3) → (num_points, num_sub, 3)
    pt_expanded = verify_points.unsqueeze(1)  # (num_points, 1, 3)
    center_expanded = sub_centers.unsqueeze(0)  # (1, num_sub, 3)
    distances = torch.norm(pt_expanded - center_expanded, dim=2)  # (num_points, num_sub)
    
    # 1.2 判断每个点是否属于某个子区域 (优先匹配第一个满足条件的子区域)
    in_subregion = distances <= (r + 1e-12)  # (num_points, num_sub)，布尔矩阵
    # 初始化区域索引为0（主区域），shape=(num_points,)
    region_indices = torch.zeros(num_points, dtype=torch.int64, device=device)
    # 找到每个点第一个匹配的子区域索引（+1因为子区域索引从1开始）
    if num_sub > 0:
        # 找到每行第一个True的位置 (num_points,)
        first_sub_idx = torch.argmax(in_subregion.int(), dim=1)  # 无匹配时返回0
        # 只有当存在匹配时，才更新区域索引
        has_match = torch.any(in_subregion, dim=1)  # (num_points,)
        region_indices[has_match] = first_sub_idx[has_match] + 1  # 子区域索引从1开始
    
    # ===================== 优化2：按区域分组批量计算近似解 =====================
    # 遍历所有可能的区域（0=主区域，1~num_sub=子区域）
    for region_idx in range(num_sub + 1):
        # 2.1 筛选该区域的所有点
        mask = (region_indices == region_idx)
        if not torch.any(mask):
            continue  # 该区域无点，跳过
        region_pts = verify_points[mask]  # (num_region_pts, 3)
        
        # 2.2 批量计算该区域的基函数值 (num_region_pts, M+1)
        psi_vals = basis_psi(region_pts, region_idx, centers, weights, biases, device)
        
        # 2.3 批量计算近似解：psi_vals @ alpha → (num_region_pts,)
        alpha = alphas[region_idx].unsqueeze(1)  # (M+1, 1)
        region_approx = torch.matmul(psi_vals, alpha).squeeze(1)  # (num_region_pts,)
        
        # 2.4 赋值到全局近似解数组
        global_approx[mask] = region_approx
    
    # ===================== 计算相对L2误差（逻辑不变） =====================
    # 避免分母为0（分析解全0时的保护）
    analytical_norm = torch.sqrt(torch.mean(torch.pow(analytical_sol, 2)))
    if analytical_norm < 1e-12:
        l2_loss = torch.sqrt(torch.mean(torch.pow(global_approx - analytical_sol, 2)))
        print(f"分析解范数接近0，计算绝对L2误差：{l2_loss.item():.6f}")
    else:
        l2_loss = torch.sqrt(torch.mean(torch.pow(global_approx - analytical_sol, 2))) / analytical_norm
        print(f"全局相对L2误差：{l2_loss.item():.6f}")
    
    return l2_loss, global_approx

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
# ============================== 8. 主求解函数 ==============================
def solve_pde_fixed_regions(config, device):
    """固定区域求解3D PDE：主区域5000基，8个子区域各500基"""
    # 1. 初始化区域圆心（固定）
    region_centers = torch.tensor([
        [0.5, 0.5, 0.5],  # 主区域
        [0.2, 0.2, 0.2], [0.2, 0.2, 0.8], [0.2, 0.8, 0.2], [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.2], [0.8, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.8, 0.8]
    ], dtype=torch.float64, device=device)
    
    # 2. 初始化基函数参数
    weights, biases, alphas = init_basis_params(config, device)
    
    # 3. 生成所有区域配置点
    print("\n=== 生成区域配置点 ===")
    regions = generate_all_regions(config, region_centers, device)
    
    # 4. 构造全局线性系统
    print("\n=== 构造全局线性系统 ===")
    global_A, global_b = build_global_system(regions, region_centers, weights, biases, config, device)
    
    # 5. 最小二乘求解所有系数
    print("\n=== 求解全局最小二乘问题 ===")
    start_solve = time.perf_counter()
    all_coeffs = qr_least_squares(global_A, global_b, device)
    solve_time = time.perf_counter() - start_solve
    print(f"求解耗时：{solve_time:.6f}秒")
    
    # 6. 分配系数到各区域
    main_coeffs_num = config["main_basis_num"] + 1
    alphas[0] = all_coeffs[:main_coeffs_num]
    sub_coeffs_num = config["sub_basis_num"] + 1
    for k in range(config["num_sub_regions"]):
        start = main_coeffs_num + k * sub_coeffs_num
        end = start + sub_coeffs_num
        alphas[k+1] = all_coeffs[start:end]
    
    # 7. 计算全局L2误差
    print("\n=== 计算全局L2误差 ===")
    # 加载验证数据
    test_point = np.loadtxt("3D_darcy_adam_prediction.txt")
    verify_points = torch.from_numpy(test_point[:, 0:3]).to(device)
    analytical_sol = torch.from_numpy(np.loadtxt("pred_u_FE_300.txt")).to(device)
    # 计算误差
    relative_err, global_approx = compute_global_l2_error(
        verify_points, analytical_sol, region_centers, weights, biases, alphas, config, device
    )
    print(f"全局相对L2误差：{relative_err:.6f}")
    
    # 8. 绘制3D预测结果图
    print("\n=== 绘制3D预测分布图 ===")
    plot_3d_prediction(verify_points, global_approx)
    plot_z01_slice(verify_points, global_approx)
    # 9. 整理结果
    results = {
        "alphas": alphas,
        "region_centers": region_centers,
        "regions": regions,
        "global_relative_err": relative_err,
        "global_approx": global_approx,
        "analytical_sol": analytical_sol
    }
    
    return results

def plot_3d_prediction(points, pred_vals):
    """绘制3D预测值分布图"""
    x = points.cpu().numpy()[:, 0]
    y = points.cpu().numpy()[:, 1]
    z = points.cpu().numpy()[:, 2]
    pred = pred_vals.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=pred, s=5, cmap='viridis', alpha=0.7)
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Predicted Value', fontsize=12)
    
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)
    ax.set_zlabel('Z Coordinate', fontsize=10)
    ax.set_title('3D Distribution of Predicted Values (Fixed Regions)', fontsize=14, pad=20)
    ax.view_init(elev=20, azim=45)
    
    plt.show()

# ============================== 9. 程序入口 ==============================
if __name__ == "__main__":
    # 1. 配置与设备
    config, device = set_global_config()
    # 2. 记录总运行时间
    start_total = time.perf_counter()
    # 3. 求解PDE
    results = solve_pde_fixed_regions(config, device)
    # 4. 输出总结
    total_time = time.perf_counter() - start_total
    print("\n=== 程序执行完成 ===")
    print(f"总运行时间：{total_time:.6f}秒")
    print(f"最终全局相对L2误差：{results['global_relative_err']:.6f}")