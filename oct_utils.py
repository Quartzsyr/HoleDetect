import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import tempfile
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QFileDialog, QMessageBox, QGroupBox, QFormLayout,
                             QRadioButton, QLabel, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import pandas as pd
from scipy.optimize import least_squares

def fit_circle_algebraic(points_2d):
    """
    使用代数最小二乘法拟合2D平面上的圆（内部函数）
    """
    if len(points_2d) < 3:
        raise ValueError("至少需要3个点才能拟合圆")
    
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    
    try:
        solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        center_x, center_y = solution[0], solution[1]
        radius = np.sqrt(solution[2] + center_x**2 + center_y**2)
        return center_x, center_y, radius
    except Exception as e:
        print(f"代数圆拟合出错: {str(e)}")
        raise

def fit_circle_geometric(points_2d):
    """
    使用迭代法（Levenberg-Marquardt）进行几何拟合，最小化真实几何距离。
    这是拟合圆的黄金标准方法。
    
    参数:
        points_2d: 形状为(n,2)的平面点集
        
    返回:
        圆心坐标和半径 (center_x, center_y, radius)
    """
    if len(points_2d) < 3:
        raise ValueError("至少需要3个点才能拟合圆")

    # 1. 使用代数方法提供一个高质量的初始猜测值
    try:
        center_x_initial, center_y_initial, radius_initial = fit_circle_algebraic(points_2d)
    except Exception as e:
        print(f"几何拟合的初始值计算（代数法）失败: {e}，将使用简单平均值。")
        center_x_initial = np.mean(points_2d[:, 0])
        center_y_initial = np.mean(points_2d[:, 1])
        radius_initial = np.mean(np.sqrt((points_2d[:, 0] - center_x_initial)**2 + (points_2d[:, 1] - center_y_initial)**2))

    initial_guess = [center_x_initial, center_y_initial, radius_initial]

    def residuals(params, x, y):
        """计算每个点到圆的真实几何距离（残差）"""
        center_x, center_y, radius = params
        return np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

    # 2. 使用非线性最小二乘法进行迭代优化
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # 为半径设置下限（不能为负）
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])

    try:
        # 使用Levenberg-Marquardt算法，更精确地最小化几何距离
        res_lsq = least_squares(residuals, initial_guess, args=(x, y), bounds=bounds, method='trf', loss='linear')
        
        center_x, center_y, radius = res_lsq.x
        
        # 计算拟合误差，用于评估拟合质量
        errors = np.abs(residuals([center_x, center_y, radius], x, y))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"迭代几何拟合成功: 中心=({center_x:.1f}, {center_y:.1f}), 半径={radius:.1f}")
        print(f"拟合质量: 平均误差={mean_error:.3f}, 最大误差={max_error:.3f}")
        
        return center_x, center_y, radius
    except Exception as e:
        print(f"迭代几何拟合过程中出错: {str(e)}")
        # 如果迭代失败，则抛出异常
        raise e

def fit_circle_geometric_forced(points_2d):
    """
    强制使用几何拟合，即使结果可能不理想。
    这个函数不会回退到代数法，用于展示真正的几何拟合结果。
    """
    if len(points_2d) < 3:
        raise ValueError("至少需要3个点才能拟合圆")
    
    # 计算质心作为初始圆心猜测
    center_x_initial = np.mean(points_2d[:, 0])
    center_y_initial = np.mean(points_2d[:, 1])
    
    # 计算平均距离作为初始半径猜测
    distances = np.sqrt((points_2d[:, 0] - center_x_initial)**2 + (points_2d[:, 1] - center_y_initial)**2)
    radius_initial = np.mean(distances)
    
    initial_guess = [center_x_initial, center_y_initial, radius_initial]
    
    def residuals(params, x, y):
        """计算每个点到圆的真实几何距离（残差）"""
        center_x, center_y, radius = params
        return np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius
    
    # 使用非线性最小二乘法进行迭代优化
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # 使用更严格的优化参数
    res_lsq = least_squares(residuals, initial_guess, args=(x, y), 
                           method='trf', loss='linear', 
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, 
                           max_nfev=1000)  # 增加最大迭代次数
    
    center_x, center_y, radius = res_lsq.x
    
    # 计算拟合质量指标
    errors = np.abs(residuals([center_x, center_y, radius], x, y))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"强制几何拟合: 中心=({center_x:.1f}, {center_y:.1f}), 半径={radius:.1f}")
    print(f"拟合质量: 平均误差={mean_error:.3f}, 最大误差={max_error:.3f}")
    
    return center_x, center_y, radius

def fit_circle_2d(points_2d, method='algebraic'):
    """
    使用指定方法拟合2D平面上的圆
    
    参数:
        points_2d: 形状为(n,2)的平面点集
        method: 'algebraic'代数法, 'geometric'几何法, 或 'geometric_forced'强制几何法
        
    返回:
        圆心坐标和半径 (center_x, center_y, radius)
    """
    print(f"开始拟合圆，使用方法: {method}，点数: {len(points_2d)}")
    
    if method == 'geometric':
        try:
            return fit_circle_geometric(points_2d)
        except Exception as e:
            print(f"几何拟合失败，回退到代数方法: {str(e)}")
            return fit_circle_algebraic(points_2d)
    elif method == 'geometric_forced':
        return fit_circle_geometric_forced(points_2d)
    else:  # 默认使用代数法
        return fit_circle_algebraic(points_2d)

def transform_to_2d_coords(points_2d, scan_positions, pixel_to_um=1.0):
    """
    将OCT图像上的2D点转换为XY平面坐标（简化版）
    
    参数:
        points_2d: 形状为(n,4)的数组，每行包含 [x1, y1, x2, y2]，这些是在OCT图像上标记的孔洞直径端点
        scan_positions: OCT扫描的位置信息，单位为微米，长度为n的数组，表示每个截面在Y方向上的实际位置
        pixel_to_um: 像素到微米的转换比例
        
    返回:
        形状为(n*2,2)的2D点数组，单位为微米
        每个点的坐标为(X, Y)，其中X是孔洞端点的X坐标，Y是扫描位置
    """
    n = len(points_2d)
    points_xy = np.zeros((n*2, 2))
    
    for i in range(n):
        x1, y1, x2, y2 = points_2d[i]
        scan_y = scan_positions[i]  # Y方向上的扫描位置（微米）
        
        # 存储两个端点的X坐标和扫描位置Y
        points_xy[i*2] = [x1 * pixel_to_um, scan_y]
        points_xy[i*2+1] = [x2 * pixel_to_um, scan_y]
    
    return points_xy

def create_simple_visualization(points_xy, circle_center, radius, pixel_to_um=1.0, fit_method='algebraic', font_prop=None):
    """
    创建简化版OCT重建结果的2D可视化
    
    参数:
        points_xy: 形状为(n*2,2)的2D点数组，单位为微米，表示每个扫描截面上的孔洞端点
        circle_center: 圆心坐标 [x, y]，单位为微米
        radius: 圆半径，单位为微米
        pixel_to_um: 像素到微米的转换比例 (当前未使用，但为保持签名兼容性而保留)
        fit_method: 拟合方法，'algebraic'代数方法或'geometric'几何方法
        font_prop: 用于支持中文显示的字体属性
        
    返回:
        matplotlib Figure 对象
    """
    # 创建图像
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制拟合圆
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = circle_center[0] + radius * np.cos(circle_theta)
    circle_y = circle_center[1] + radius * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='拟合圆')
    
    # 绘制圆心
    ax.scatter(circle_center[0], circle_center[1], c='red', s=100, label='圆心', zorder=5)
    
    # 绘制原始检测点
    ax.scatter(points_xy[:, 0], points_xy[:, 1], c='blue', s=50, label='检测点', zorder=5)
    
    # 打印调试信息
    print(f"可视化: 点数={len(points_xy)}, 圆心=({circle_center[0]:.1f}, {circle_center[1]:.1f}), 半径={radius:.1f}")
    
    # 连接相同扫描位置的两个点（孔洞直径）
    points_count = len(points_xy) // 2
    for i in range(points_count):
        p1 = points_xy[i*2]
        p2 = points_xy[i*2+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.7, linewidth=1.5)
        
        # 计算直径并显示标签
        diameter = np.sqrt((p2[0] - p1[0])**2)
        
        # 每隔几个点显示一个标签，避免拥挤
        if i % 3 == 0:  
            ax.text((p1[0] + p2[0])/2, p1[1] + radius/20, 
                     f"{diameter:.1f} μm", 
                     ha='center', va='bottom', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'),
                     fontproperties=font_prop)
    
    # 设置图表属性
    ax.set_title(f"OCT重建俯视图 (拟合方法: {fit_method})", fontproperties=font_prop)
    ax.set_xlabel("X (μm)", fontproperties=font_prop)
    ax.set_ylabel("Y (μm)", fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', 'box') # 保证X和Y轴等比例，使圆看起来是圆的
    
    # 自动调整布局
    fig.tight_layout()
    
    return fig

# 保留以下函数以保持兼容性，但在新的实现中不会被调用
def fit_plane(points_3d):
    """
    使用PCA拟合3D点云的平面
    返回平面的法向量和中心点
    """
    # 计算点的质心
    centroid = np.mean(points_3d, axis=0)
    
    # 中心化数据
    centered_points = points_3d - centroid
    
    # 使用PCA分解
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # 最小奇异值对应的右奇异向量即为平面法向量
    normal = pca.components_[2]
    
    # 计算d值: ax + by + cz + d = 0 => d = -(ax + by + cz)
    d = -np.dot(normal, centroid)
    
    return np.append(normal, d)

def project_point_to_plane(point, plane_params):
    """
    将点投影到平面上（保留以保持兼容性）
    """
    a, b, c, d = plane_params
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # 单位化法向量
    
    # 点到平面的距离
    distance = (np.dot(normal, point) + d) / np.linalg.norm(normal)
    
    # 投影点 = 原点 - 距离 * 法向量
    projected_point = point - distance * normal
    
    return projected_point

def transform_to_3d_coords(points_2d, scan_positions, pixel_to_um=1.0):
    """
    将OCT图像上的2D点转换为3D空间坐标（保留以保持兼容性）
    """
    # 调用新函数并适配为旧格式
    points_xy = transform_to_2d_coords(points_2d, scan_positions, pixel_to_um)
    # 为了保持与原来3D格式的兼容性，添加一个全零的Z坐标
    points_3d = np.zeros((len(points_xy), 3))
    points_3d[:, 0] = points_xy[:, 0]  # X
    points_3d[:, 2] = points_xy[:, 1]  # Y作为Z
    
    return points_3d

def transform_3d_to_plane_coords(points_3d, plane_params):
    """
    将3D点转换到拟合平面上的2D坐标系（保留以保持兼容性）
    """
    # 简化实现，直接返回XY坐标
    points_2d = points_3d[:, :2]
    return points_2d, points_3d, None

def create_visualization(points_3d, plane_params, circle_center_3d, radius, output_path=None, pixel_to_um=1.0):
    """
    创建OCT重建结果的3D可视化（保留以保持兼容性）
    """
    # 简化为调用2D可视化函数
    circle_center_2d = circle_center_3d[:2]
    return create_simple_visualization(points_3d[:, :2], circle_center_2d, radius, 
                                      pixel_to_um=pixel_to_um)

class OCTHoleReconstructionDialog(QDialog):
    def __init__(self, main_app_instance, image_label_class, parent=None):
        super().__init__(parent)
        self.main_app = main_app_instance
        self.ImageLabel = image_label_class
        self.points_2d = []
        self.scan_positions = []
        self.pixel_to_um = 2.0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("OCT圆孔重建")
        self.setGeometry(150, 150, 1400, 800)
        main_layout = QHBoxLayout(self)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(400)

        file_group = QGroupBox("数据加载")
        file_layout = QFormLayout(file_group)
        load_points_button = QPushButton("加载孔洞边缘数据 (.csv)")
        load_points_button.clicked.connect(self.load_points_data)
        file_layout.addRow(load_points_button)
        load_positions_button = QPushButton("加载扫描位置数据 (.csv)")
        load_positions_button.clicked.connect(self.load_scan_positions)
        file_layout.addRow(load_positions_button)
        control_layout.addWidget(file_group)

        # 替换单选按钮为下拉框
        fit_method_group = QGroupBox("拟合方法")
        fit_method_layout = QVBoxLayout()
        
        from PyQt5.QtWidgets import QComboBox
        self.fit_method_combo = QComboBox()
        self.fit_method_combo.addItem("代数法 (Algebraic) - 速度快，适合噪声数据", "algebraic")
        self.fit_method_combo.addItem("几何法 (Geometric) - 精确，可能回退", "geometric")
        self.fit_method_combo.addItem("强制几何法 (Forced Geometric) - 纯几何拟合", "geometric_forced")
        self.fit_method_combo.setCurrentIndex(0)
        self.fit_method_combo.currentIndexChanged.connect(self.process_and_reconstruct_if_ready)
        
        fit_method_layout.addWidget(self.fit_method_combo)
        fit_method_group.setLayout(fit_method_layout)
        control_layout.addWidget(fit_method_group)

        result_group = QGroupBox("重建结果")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("请先加载数据并开始重建")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        control_layout.addWidget(result_group)

        reconstruct_button = QPushButton("处理并重建")
        reconstruct_button.clicked.connect(self.process_and_reconstruct)
        control_layout.addWidget(reconstruct_button)
        control_layout.addStretch()
        
        self.result_image_label = self.ImageLabel()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.result_image_label)
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def load_points_data(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "加载孔洞边缘数据", "", "CSV文件 (*.csv)")
        if filePath:
            try:
                df = pd.read_csv(filePath)
                self.points_2d = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
                QMessageBox.information(self, "成功", f"成功加载 {len(self.points_2d)} 条边缘数据。")
                self.process_and_reconstruct_if_ready()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载边缘数据失败: {e}")

    def load_scan_positions(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "加载扫描位置数据", "", "CSV文件 (*.csv)")
        if filePath:
            try:
                df = pd.read_csv(filePath)
                self.scan_positions = df.iloc[:, 0].values.tolist()
                QMessageBox.information(self, "成功", f"成功加载 {len(self.scan_positions)} 条位置数据。")
                self.process_and_reconstruct_if_ready()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载位置数据失败: {e}")

    def process_and_reconstruct_if_ready(self):
        if self.points_2d and self.scan_positions:
            self.process_and_reconstruct()

    def process_and_reconstruct(self):
        if not self.points_2d or not self.scan_positions:
            return
        if len(self.points_2d) != len(self.scan_positions):
            QMessageBox.warning(self, "数据不匹配", "边缘数据和位置数据记录数不一致。")
            return
        
        points_xy = transform_to_2d_coords(np.array(self.points_2d), self.scan_positions, self.pixel_to_um)
        
        # 从下拉框获取拟合方法
        fit_method = self.fit_method_combo.currentData()
        fit_method_name = self.fit_method_combo.currentText().split(" ")[0]
        
        try:
            center_x, center_y, radius = fit_circle_2d(points_xy, method=fit_method)
            self.result_label.setText(f"拟合结果: 直径 = {2*radius:.2f} μm\n"
                                      f"圆心 = ({center_x:.2f}, {center_y:.2f}) μm\n"
                                      f"方法 = {fit_method_name}")
            fig = create_simple_visualization(points_xy, [center_x, center_y], radius,
                                                      pixel_to_um=self.pixel_to_um,
                                                      fit_method=fit_method)
            self.result_image_label.setPixmap(QPixmap.fromImage(QImage(fig.canvas.tostring_rgb())))
        except (ValueError, np.linalg.LinAlgError) as e:
            QMessageBox.warning(self, "拟合错误", f"无法拟合圆: {e}")
            self.result_label.setText(f"拟合失败: {e}\n方法 = {fit_method_name}")
        except Exception as e:
            QMessageBox.critical(self, "重建错误", f"未知错误: {e}")
            self.result_label.setText(f"未知错误\n方法 = {fit_method_name}")
            import traceback
            traceback.print_exc() 