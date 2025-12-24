import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def extract_points_from_image(image_path, interactive=True):
    """
    从OCT图像中提取直径两端点的坐标
    
    参数:
        image_path: OCT图像路径
        interactive: 是否使用交互模式选择点
        
    返回:
        两个点的坐标 [x1, y1, x2, y2]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    points = []
    
    if interactive:
        # 交互式选择点
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("图像", img)
                
                if len(points) == 2:
                    cv2.line(img, (points[0][0], points[0][1]), 
                            (points[1][0], points[1][1]), (0, 0, 255), 2)
                    cv2.imshow("图像", img)
        
        cv2.imshow("图像", img)
        cv2.setMouseCallback("图像", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(points) != 2:
            raise ValueError("请选择两个点（直径两端）")
    else:
        # 这里可以添加自动检测点的算法
        # 例如使用霍夫变换检测圆或椭圆
        pass
    
    return np.array(points).flatten()  # [x1, y1, x2, y2]

def fit_plane(points_3d):
    """
    使用SVD方法拟合3D平面
    
    参数:
        points_3d: 形状为(n,3)的点集数组
        
    返回:
        平面参数 (a, b, c, d) 对应 ax + by + cz + d = 0
    """
    # 计算点的质心
    centroid = np.mean(points_3d, axis=0)
    
    # 中心化数据
    centered_points = points_3d - centroid
    
    # 使用SVD分解
    u, s, vh = np.linalg.svd(centered_points)
    
    # 最小奇异值对应的右奇异向量即为平面法向量
    normal = vh[-1]
    
    # 计算d值: ax + by + cz + d = 0 => d = -(ax + by + cz)
    d = -np.dot(normal, centroid)
    
    return np.append(normal, d)

def project_point_to_plane(point, plane_params):
    """
    将点投影到平面上
    
    参数:
        point: 形状为(3,)的点坐标
        plane_params: 平面参数 (a, b, c, d)
        
    返回:
        投影后的点坐标
    """
    a, b, c, d = plane_params
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # 单位化法向量
    
    # 点到平面的距离
    distance = (np.dot(normal, point) + d) / np.linalg.norm(normal)
    
    # 投影点 = 原点 - 距离 * 法向量
    projected_point = point - distance * normal
    
    return projected_point

def fit_circle_2d(points_2d):
    """
    使用最小二乘法拟合2D平面上的圆
    
    参数:
        points_2d: 形状为(n,2)的平面点集
        
    返回:
        圆心坐标和半径 (center_x, center_y, radius)
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # 构造线性方程组的系数矩阵
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    
    # 求解线性方程组
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 提取圆心和半径
    center_x, center_y = solution[0], solution[1]
    radius = np.sqrt(solution[2] + center_x**2 + center_y**2)
    
    return center_x, center_y, radius

def transform_to_3d_coords(points_2d, scan_positions):
    """
    将OCT图像上的2D点转换为3D空间坐标
    
    参数:
        points_2d: 形状为(n,4)的数组，每行包含 [x1, y1, x2, y2]
        scan_positions: OCT扫描的位置信息，长度为n的数组
        
    返回:
        形状为(n*2,3)的3D点数组
    """
    n = len(points_2d)
    points_3d = np.zeros((n*2, 3))
    
    for i in range(n):
        x1, y1, x2, y2 = points_2d[i]
        z = scan_positions[i]  # OCT扫描的Z轴位置
        
        # 将每个图像上的两个点转换为3D坐标
        points_3d[i*2] = [x1, y1, z]
        points_3d[i*2+1] = [x2, y2, z]
    
    return points_3d

def transform_3d_to_plane_coords(points_3d, plane_params):
    """
    将3D点转换到拟合平面上的2D坐标系
    
    参数:
        points_3d: 3D点坐标数组
        plane_params: 平面参数 (a, b, c, d)
        
    返回:
        平面上的2D坐标数组
    """
    # 投影所有点到平面
    projected_points = np.array([project_point_to_plane(p, plane_params) for p in points_3d])
    
    # 使用PCA将3D点转换为平面2D坐标系
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(projected_points)
    
    return points_2d, projected_points, pca

def visualize_results(points_3d, plane_points, circle_center_3d, radius, plane_params):
    """
    可视化结果：原始点、拟合平面和圆
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始点
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', label='原始点')
    
    # 绘制平面上的点
    ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], c='green', label='平面投影点')
    
    # 绘制平面
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    a, b, c, d = plane_params
    Z = (-a * X - b * Y - d) / c if c != 0 else np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan')
    
    # 绘制圆
    theta = np.linspace(0, 2*np.pi, 100)
    a, b, c = plane_params[:3]
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # 创建圆周上的点
    v1 = np.cross(normal, np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    circle_points = circle_center_3d + radius * (v1[:, np.newaxis] * np.cos(theta) + v2[:, np.newaxis] * np.sin(theta))
    circle_points = circle_points.T
    
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'r-', label='拟合圆')
    ax.scatter(circle_center_3d[0], circle_center_3d[1], circle_center_3d[2], c='red', s=100, label='圆心')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('OCT圆孔重建结果')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def measure_oct_hole_diameter(image_paths, scan_positions):
    """
    主函数：从OCT图像测量真实的圆孔直径
    
    参数:
        image_paths: OCT图像路径列表
        scan_positions: 对应的扫描位置(Z坐标)列表
        
    返回:
        圆孔的真实直径
    """
    # 1. 从图像中提取点
    points_2d = []
    for image_path in image_paths:
        points = extract_points_from_image(image_path)
        points_2d.append(points)
    points_2d = np.array(points_2d)
    
    # 2. 转换为3D坐标
    points_3d = transform_to_3d_coords(points_2d, scan_positions)
    
    # 3. 拟合3D平面
    plane_params = fit_plane(points_3d)
    print(f"拟合平面参数 (a, b, c, d): {plane_params}")
    
    # 4. 将点投影到平面上并转换为2D坐标
    points_2d_plane, projected_points, pca = transform_3d_to_plane_coords(points_3d, plane_params)
    
    # 5. 在平面上拟合圆
    center_x, center_y, radius = fit_circle_2d(points_2d_plane)
    print(f"平面坐标系中的圆心: ({center_x:.3f}, {center_y:.3f})")
    print(f"圆的半径: {radius:.3f}")
    
    # 6. 将圆心转换回3D坐标
    circle_center_2d = np.array([center_x, center_y])
    circle_center_3d = pca.inverse_transform(circle_center_2d)
    
    # 7. 可视化结果
    visualize_results(points_3d, projected_points, circle_center_3d, radius, plane_params)
    
    # 8. 计算直径
    diameter = 2 * radius
    print(f"圆孔的真实直径: {diameter:.3f}")
    
    return diameter

if __name__ == "__main__":
    # 示例使用
    print("OCT圆孔直径测量程序")
    print("="*50)
    print("请按提示操作:")
    
    # 获取OCT图像路径
    image_paths = []
    scan_positions = []
    
    num_images = int(input("请输入OCT图像数量 (建议4张以上): "))
    
    for i in range(num_images):
        path = input(f"请输入第{i+1}张OCT图像路径: ")
        position = float(input(f"请输入第{i+1}张OCT图像的扫描位置 (Z坐标): "))
        
        image_paths.append(path)
        scan_positions.append(position)
    
    # 执行测量
    diameter = measure_oct_hole_diameter(image_paths, scan_positions)
    
    print("\n程序完成!")
    print(f"圆孔的真实直径: {diameter:.3f} 像素单位") 