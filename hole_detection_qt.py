import sys
import os
import cv2  # OpenCV库
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QSlider, QSpinBox, 
                            QGroupBox, QGridLayout, QCheckBox, QSplitter, QSizePolicy,
                            QMenuBar, QMenu, QAction, QMessageBox, QDialog, QFormLayout,
                            QProgressDialog, QFrame, QToolButton, QScrollArea, QDoubleSpinBox,
                            QDialogButtonBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QKeySequence, QFont, QColor, QPalette, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QPoint, QRect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import csv
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import oct_module
from sklearn.decomposition import PCA
from pixel_calibration import PixelCalibrationApp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 检查字体是否存在，如果不存在则使用系统默认字体
fonts = [f.name for f in fm.fontManager.ttflist]
if 'SimHei' in fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'SimHei')
elif 'Microsoft YaHei' in fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Microsoft YaHei')

# 文件夹路径设置
input_dir = 'input'
output_dir = 'output'
debug_dir = 'debug'

# 没有则创建输出和调试文件夹
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# 像素到实际尺寸的转换比例
PIXEL_TO_UM_X = 2  # X方向（直径）像素到微米的转换比例
PIXEL_TO_UM_Y = 2  # Y方向（深度）像素到微米的转换比例
# 旧的全局转换常量，保留用于兼容性
PIXEL_TO_UM = 2

class ImageLabel(QLabel):
    """自定义QLabel用于显示图像，能够调整大小时保持图像显示"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)  # 设置最小尺寸
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #CCCCCC; border-radius: 5px; background-color: #F5F5F5;")
        self._pixmap = None
        self._scaled_pixmap = None
        self.aspect_ratio = 1.0
        
        # 用于裁剪的变量
        self.crop_active = False
        self.crop_start = None
        self.crop_end = None
        
        # 用于手动测量的变量
        self.manual_measure_active = False
        self.manual_measure_points = []
        self.current_mouse_pos = None  # 跟踪当前鼠标位置
        
        # 用于控制测量模式（深度或直径）
        self.measure_mode = "diameter"  # 默认测量直径，可选值："diameter"或"depth"
        
        # 设置鼠标跟踪
        self.setMouseTracking(True)  # 启用鼠标追踪，鼠标移动时也会触发事件
        
    def createOscilloscopeCursor(self):
        """创建示波器样式的光标，X轴测量直径，Y轴测量深度"""
        # 创建48x48像素的较大位图，以便有更多空间显示细节
        bitmap = QPixmap(48, 48)
        bitmap.fill(Qt.transparent)
        
        # 创建画笔
        painter = QPainter(bitmap)
        
        # 绘制示波器风格的十字光标（黄色线条）
        painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.DashLine))  # 黄色虚线
        painter.drawLine(0, 24, 48, 24)  # 水平线
        painter.drawLine(24, 0, 24, 48)  # 垂直线
        
        # 在十字线交点绘制小圆圈
        painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(22, 22, 4, 4)
        
        # 添加测量标记
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        
        # X轴标记（直径）
        painter.setPen(Qt.yellow)
        painter.drawText(31, 20, "Ø")
        
        # Y轴标记（深度）
        painter.drawText(15, 16, "D")
        
        # 添加小刻度线
        painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.SolidLine))
        # X轴刻度
        for i in range(1, 5):
            painter.drawLine(24 + i*4, 23, 24 + i*4, 25)  # 右侧刻度
            painter.drawLine(24 - i*4, 23, 24 - i*4, 25)  # 左侧刻度
        
        # Y轴刻度
        for i in range(1, 5):
            painter.drawLine(23, 24 + i*4, 25, 24 + i*4)  # 下方刻度
            painter.drawLine(23, 24 - i*4, 25, 24 - i*4)  # 上方刻度
            
        painter.end()
        
        # 设置热点在十字线的中心
        return QCursor(bitmap, 24, 24)
    
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.updatePixmap()
        
    def updatePixmap(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled_pixmap = self._pixmap.scaled(self.size(), 
                                              Qt.KeepAspectRatio, 
                                              Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """当控件大小改变时，重新调整图像大小"""
        self.updatePixmap()
        super().resizeEvent(event)
    
    def setCropActive(self, active):
        """设置是否处于裁剪模式"""
        print(f"设置裁剪模式: {active}")
        self.crop_active = active
        self.crop_start = None
        self.crop_end = None
        
        if active:
            self.manual_measure_active = False  # 关闭手动测量模式
            self.manual_measure_points = []
        
        # 根据裁剪模式更改鼠标样式
        if active:
            self.setCursor(Qt.CrossCursor)  # 设置十字光标
            print("裁剪模式启用，设置为十字光标")
        else:
            self.setCursor(Qt.ArrowCursor)  # 恢复箭头光标
            print("裁剪模式关闭，恢复默认光标")
            
        self.update()  # 刷新显示
    
    def setManualMeasureActive(self, active):
        """设置是否处于手动测量模式"""
        print(f"设置手动测量模式: {active}")
        self.manual_measure_active = active
        
        if active:
            self.crop_active = False  # 关闭裁剪模式
            self.crop_start = None
            self.crop_end = None
            self.manual_measure_points = []  # 清空测量点
            self.measure_mode = "diameter"  # 默认测量直径
            # 不再设置自定义光标，而是使用普通光标
            self.setCursor(Qt.CrossCursor)
            print("手动测量模式启用，使用十字光标配合示波器线")
        else:
            self.setCursor(Qt.ArrowCursor)  # 恢复箭头光标
            self.current_mouse_pos = None  # 清除鼠标位置
            print("手动测量模式关闭，恢复默认光标")
            
        self.update()  # 刷新显示
    
    def getImageRect(self):
        """获取图像在标签中的实际矩形区域"""
        if not self._pixmap or self._pixmap.isNull():
            print("无法获取图像区域：未设置有效图像")
            return QRect()
            
        # 获取标签的尺寸
        label_width = self.width()
        label_height = self.height()
        
        # 获取原始图像的尺寸
        pixmap_width = self._pixmap.width()
        pixmap_height = self._pixmap.height()
        
        # 计算缩放比例，保持宽高比
        scale_width = float(label_width) / pixmap_width
        scale_height = float(label_height) / pixmap_height
        scale = min(scale_width, scale_height)
        
        # 计算缩放后的图像尺寸
        scaled_width = int(pixmap_width * scale)
        scaled_height = int(pixmap_height * scale)
        
        # 计算图像在标签中的位置（居中显示）
        x = (label_width - scaled_width) // 2
        y = (label_height - scaled_height) // 2
        
        # 创建矩形区域
        image_rect = QRect(x, y, scaled_width, scaled_height)
        
        print(f"图像显示区域: ({x}, {y}, {scaled_width}, {scaled_height}), 原图尺寸: {pixmap_width}x{pixmap_height}, 标签尺寸: {label_width}x{label_height}")
        
        return image_rect
        
    def getImageCoordinates(self, pos):
        """将窗口坐标转换为图像坐标"""
        if not self._pixmap or self._pixmap.isNull():
            print("无法获取图像坐标：没有有效的图像")
            return None
            
        # 获取图像在标签中的实际矩形区域
        image_rect = self.getImageRect()
        
        # 检查点击点是否在图像区域内
        if not image_rect.contains(pos):
            print(f"点击位置 ({pos.x()}, {pos.y()}) 不在图像区域内 ({image_rect.left()}, {image_rect.top()}, {image_rect.right()}, {image_rect.bottom()})")
            return None
            
        try:
            # 转换为图像坐标
            x_ratio = float(self._pixmap.width()) / float(image_rect.width())
            y_ratio = float(self._pixmap.height()) / float(image_rect.height())
            
            img_x = int((pos.x() - image_rect.left()) * x_ratio)
            img_y = int((pos.y() - image_rect.top()) * y_ratio)
            
            # 确保坐标在图像范围内
            img_x = max(0, min(img_x, self._pixmap.width() - 1))
            img_y = max(0, min(img_y, self._pixmap.height() - 1))
            
            print(f"窗口坐标 ({pos.x()}, {pos.y()}) 转换为图像坐标 ({img_x}, {img_y})")
            return QPoint(img_x, img_y)
        except Exception as e:
            print(f"坐标转换错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.crop_active and event.button() == Qt.LeftButton:
            # 裁剪模式下，记录起始点
            img_pos = self.getImageCoordinates(event.pos())
            if img_pos:
                self.crop_start = img_pos
                self.crop_end = None
                print(f"裁剪起点: ({self.crop_start.x()}, {self.crop_start.y()})")
                self.update()
        elif self.manual_measure_active:
            if event.button() == Qt.LeftButton:
                # 左键点击确定测量点
                self.handleManualMeasurement(event)
            elif event.button() == Qt.RightButton:
                # 右键切换测量模式（直径/深度）
                if self.measure_mode == "diameter":
                    self.measure_mode = "depth"
                    print("切换到测量深度模式")
                else:
                    self.measure_mode = "diameter"
                    print("切换到测量直径模式")
                self.update()
        
        super().mousePressEvent(event)
    
    def handleManualMeasurement(self, event):
        """处理手动测量的逻辑"""
        image_coords = self.getImageCoordinates(event.pos())
        if not image_coords:
            return

        if len(self.manual_measure_points) >= 2:
            self.manual_measure_points = []
        
        if len(self.manual_measure_points) == 1:
            if self.measure_mode == 'depth':
                image_coords.setX(self.manual_measure_points[0].x())
            elif self.measure_mode == 'diameter':
                image_coords.setY(self.manual_measure_points[0].y())
        
        self.manual_measure_points.append(image_coords)
        print(f"添加测量点: ({image_coords.x()}, {image_coords.y()}), 总点数: {len(self.manual_measure_points)}")
        
        if len(self.manual_measure_points) == 2:
            self.notifyMeasurementFinished()
        
        self.update()

    def notifyMeasurementFinished(self):
        """
        Notifies that manual measurement is finished.
        It first checks for a 'manual_measure_callback' on itself.
        If not found, it traverses up the parent hierarchy to find a 'manualMeasurementFinished' method.
        """
        # 优先: 检查实例上是否有特定的回调
        if hasattr(self, 'manual_measure_callback') and callable(self.manual_measure_callback):
            self.manual_measure_callback(self.manual_measure_points)
            return

        # 备用: 沿父级结构向上查找通用的处理器
        current = self.parent()
        while current:
            if hasattr(current, 'manualMeasurementFinished') and callable(getattr(current, 'manualMeasurementFinished')):
                getattr(current, 'manualMeasurementFinished')(self.manual_measure_points)
                return
            current = current.parent()

        print("Warning: No manualMeasurementFinished method or manual_measure_callback found.")

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.crop_active and self.crop_start is not None:
            # 裁剪模式下更新裁剪终点
            image_coords = self.getImageCoordinates(event.pos())
            if image_coords:
                self.crop_end = image_coords
                print(f"更新裁剪终点: ({self.crop_end.x()}, {self.crop_end.y()})")
                self.update()  # 刷新显示
        elif self.manual_measure_active:
            # 手动测量模式下，更新当前鼠标位置
            self.current_mouse_pos = event.pos()
            self.update()  # 刷新显示
                
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.crop_active and self.crop_start is not None and event.button() == Qt.LeftButton:
            # 确保有选择区域
            if self.crop_end is not None:
                print(f"裁剪选择完成，起点: ({self.crop_start.x()}, {self.crop_start.y()}), 终点: ({self.crop_end.x()}, {self.crop_end.y()})")
            
                # 通知父窗口裁剪选择完成
                parent = self.parent()
                if parent and hasattr(parent, 'cropSelectionFinished'):
                    parent.cropSelectionFinished()
            else:
                print("裁剪选择取消，未捕获到释放点")
        
        super().mouseReleaseEvent(event)
    
    def paintEvent(self, event):
        """绘制事件，用于绘制裁剪框和手动测量线"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制裁剪矩形
        if self.crop_active and self.crop_start and self.crop_end:
            # 将图像坐标转换为窗口坐标
            rect = self.getImageRect()
            if not rect.isValid():
                return
                
            # 计算缩放比例
            x_ratio = float(rect.width()) / float(self._pixmap.width())
            y_ratio = float(rect.height()) / float(self._pixmap.height())
            
            # 转换裁剪点到窗口坐标
            start_x = int(self.crop_start.x() * x_ratio) + rect.left()
            start_y = int(self.crop_start.y() * y_ratio) + rect.top()
            end_x = int(self.crop_end.x() * x_ratio) + rect.left()
            end_y = int(self.crop_end.y() * y_ratio) + rect.top()
            
            # 绘制裁剪矩形
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.setBrush(QColor(255, 0, 0, 50))  # 半透明红色填充
            painter.drawRect(QRect(QPoint(start_x, start_y), QPoint(end_x, end_y)))
            
        # 在手动测量模式下绘制示波器贯穿线
        if self.manual_measure_active and self.current_mouse_pos:
            # 设置不同颜色以区分深度测量和直径测量
            if self.measure_mode == "depth":
                # 深度测量使用蓝色水平线
                painter.setPen(QPen(QColor(0, 160, 255), 1, Qt.DashLine))
                painter.drawLine(0, self.current_mouse_pos.y(), self.width(), self.current_mouse_pos.y())
                # 添加深度指示标签
                painter.setPen(Qt.yellow)
                painter.drawText(10, self.current_mouse_pos.y() - 5, "深度测量 (H)")
            else:
                # 直径测量使用红色垂直线
                painter.setPen(QPen(QColor(255, 160, 0), 1, Qt.DashLine))
                painter.drawLine(self.current_mouse_pos.x(), 0, self.current_mouse_pos.x(), self.height())
                # 添加直径指示标签
                painter.setPen(Qt.yellow)
                painter.drawText(self.current_mouse_pos.x() + 5, 20, "直径测量 (D)")
            
        # 绘制已选择的测量点和临时线
        if self.manual_measure_points:
            rect = self.getImageRect()
            if rect.isValid() and self._pixmap and not self._pixmap.isNull():
                x_ratio = float(rect.width()) / self._pixmap.width()
                y_ratio = float(rect.height()) / self._pixmap.height()
                
                # 绘制点
                painter.setPen(QPen(Qt.yellow, 2))
                for point in self.manual_measure_points:
                    x = int(point.x() * x_ratio) + rect.left()
                    y = int(point.y() * y_ratio) + rect.top()
                    painter.drawEllipse(x - 3, y - 3, 6, 6)

                # 如果有一个点，绘制到鼠标的临时线
                if len(self.manual_measure_points) == 1 and self.current_mouse_pos:
                    p1_img = self.manual_measure_points[0]
                    p1_view = QPoint(int(p1_img.x() * x_ratio) + rect.left(), int(p1_img.y() * y_ratio) + rect.top())
                    
                    p2_view = self.current_mouse_pos
                    
                    # 检查鼠标是否在图像区域内
                    img_coords_p2 = self.getImageCoordinates(p2_view)
                    if img_coords_p2:
                        if self.measure_mode == 'depth':
                            p2_view.setX(p1_view.x())
                        elif self.measure_mode == 'diameter':
                            p2_view.setY(p1_view.y())

                        painter.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                        painter.drawLine(p1_view, p2_view)

                # 如果有两个点，绘制最终的测量线
                elif len(self.manual_measure_points) == 2:
                    p1_img = self.manual_measure_points[0]
                    p2_img = self.manual_measure_points[1]
                    p1_view = QPoint(int(p1_img.x() * x_ratio) + rect.left(), int(p1_img.y() * y_ratio) + rect.top())
                    p2_view = QPoint(int(p2_img.x() * x_ratio) + rect.left(), int(p2_img.y() * y_ratio) + rect.top())
                    
                    painter.setPen(QPen(Qt.yellow, 1, Qt.DashLine))
                    painter.drawLine(p1_view, p2_view)
        
        painter.end()

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if self.manual_measure_active:
            # 手动测量模式下，按D键切换到测量直径，按H键切换到测量深度
            if event.key() == Qt.Key_D:
                self.measure_mode = "diameter"
                self.update()
                print("切换到测量直径模式")
            elif event.key() == Qt.Key_H:
                self.measure_mode = "depth"
                self.update()
                print("切换到测量深度模式")
                
        super().keyPressEvent(event)

class HoleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化参数
        self.original_image = None
        self.binary_image = None
        self.result_image = None
        self.cropped_image = None
        self.current_image_path = None
        self.saved_results = 0
        
        # 图像导航参数
        self.image_files = []
        self.current_image_index = -1
        
        # 测量历史和计数
        self.measurement_count = 0
        self.measurement_history = {}
        self.diameter_history = []  # 添加缺失的属性，用于存储直径历史记录
        self.reference_depth_for_oct = 0.0 # 为OCT模块设置的参考深度
        
        # 裁剪和旋转状态
        self.is_cropping = False
        self.is_image_rotated = False
        self.rotation_angle = 0
        self.is_no_gap_measure_active = False
        
        # 设置状态栏
        self.statusbar = self.statusBar()
        
        # 多次测量结果
        self.multi_hole_measurements = []
        
        # 创建虚拟processBtn以防其他代码引用
        class DummyButton:
            def setEnabled(self, state):
                pass
        self.processBtn = DummyButton()
        
        # 设置应用程序UI
        self.initUI()
        
        # 检查输出目录
        if not os.path.exists(debug_dir):
            try:
                os.makedirs(debug_dir)
            except Exception as e:
                print(f"创建调试输出目录失败: {str(e)}")
        
        # 初始化用于指导用户的类
        class DummyButton:
            def setEnabled(self, state):
                pass
        
        # 设置应用程序UI
        self.initUI()
        
        # 创建菜单栏
        self.createMenuBar()
        
        # 创建文件夹
        for directory in [input_dir, output_dir, debug_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 重置参数
        self.resetParameters()

    def createMenuBar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        fileMenu = menubar.addMenu('文件')
        
        loadAction = QAction('打开图像', self)
        loadAction.setShortcut('Ctrl+O')
        loadAction.triggered.connect(self.loadImage)
        fileMenu.addAction(loadAction)
        
        loadFolderAction = QAction('打开文件夹', self)
        loadFolderAction.triggered.connect(self.loadFolder)
        fileMenu.addAction(loadFolderAction)
        
        saveAction = QAction('保存结果', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.triggered.connect(self.saveResult)
        fileMenu.addAction(saveAction)
        
        saveAllAction = QAction('批量保存结果', self)
        saveAllAction.triggered.connect(self.saveAllResults)
        fileMenu.addAction(saveAllAction)
        
        exportDataAction = QAction('导出测量数据', self)
        exportDataAction.triggered.connect(self.exportMeasurementData)
        fileMenu.addAction(exportDataAction)
        
        exitAction = QAction('退出', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)
        
        # 图像菜单
        imageMenu = menubar.addMenu('图像')
        
        prevImageAction = QAction('上一张图像', self)
        prevImageAction.triggered.connect(self.loadPreviousImage)
        prevImageAction.setShortcut('Left')
        imageMenu.addAction(prevImageAction)
        
        nextImageAction = QAction('下一张图像', self)
        nextImageAction.triggered.connect(self.loadNextImage)
        nextImageAction.setShortcut('Right')
        imageMenu.addAction(nextImageAction)
        
        cropAction = QAction('裁剪图像', self)
        cropAction.triggered.connect(self.toggleCropMode)
        imageMenu.addAction(cropAction)
        # 保存引用以便后续启用/禁用
        self.cropAction = cropAction
        
        rotateAction = QAction('旋转图像', self)
        rotateMenu = QMenu('旋转图像', self)
        imageMenu.addMenu(rotateMenu)
        
        # 添加不同的旋转角度选项
        for angle in [90, 180, 270]:
            rotateAngleAction = QAction(f'旋转 {angle}°', self)
            rotateAngleAction.triggered.connect(lambda checked, a=angle: self._applyRotation(a))
            rotateMenu.addAction(rotateAngleAction)
        
        # 参数菜单
        paramMenu = menubar.addMenu('参数')
        
        exportParamsAction = QAction('导出参数设置', self)
        exportParamsAction.triggered.connect(self.saveParameters)
        paramMenu.addAction(exportParamsAction)
        
        loadParamsAction = QAction('读取参数设置', self)
        loadParamsAction.triggered.connect(self.loadParameters)
        paramMenu.addAction(loadParamsAction)
        
        resetParamsAction = QAction('重置参数设置', self)
        resetParamsAction.triggered.connect(self.resetParameters)
        paramMenu.addAction(resetParamsAction)
        
        # 脚本菜单
        scriptMenu = menubar.addMenu('脚本')
        
        # 手动测量菜单项
        self.manualMeasureAction = QAction('手动测量', self)
        self.manualMeasureAction.triggered.connect(self.toggleManualMeasureMode)
        scriptMenu.addAction(self.manualMeasureAction)
        
        noiseAndMergeAction = QAction('生成噪声图像并合并', self)
        noiseAndMergeAction.triggered.connect(self.processNoiseAndMerge)
        scriptMenu.addAction(noiseAndMergeAction)
        
        analyzeMergedAction = QAction('分析合并图像', self)
        analyzeMergedAction.triggered.connect(self.analyzeMergedImage)
        scriptMenu.addAction(analyzeMergedAction)
        
        # 移除原锥度计算菜单项，只保留直接锥度计算
        directTaperAction = QAction('锥度计算', self)
        directTaperAction.triggered.connect(self.directTaperCalculation)
        scriptMenu.addAction(directTaperAction)
        
        # 添加无缺口测量菜单项
        self.noGapMeasureAction = QAction('无缺口测量', self)
        self.noGapMeasureAction.triggered.connect(self.toggleNoGapMeasureMode)
        scriptMenu.addAction(self.noGapMeasureAction)
        
        # 添加粗糙度分析菜单项
        roughnessAction = QAction('孔洞粗糙度分析', self)
        roughnessAction.triggered.connect(self.analyzeRoughness)
        scriptMenu.addAction(roughnessAction)
        
        # 帮助菜单

        # OCT圆孔重建菜单项
        octAction = QAction('OCT圆孔重建', self)
        octAction.setStatusTip('通过多张OCT图像重建真实圆孔直径')
        octAction.triggered.connect(self.startOCTReconstruction)
        
        # 将OCT选项添加到分析菜单
        if not hasattr(self, 'analyzeMenu'):
            self.analyzeMenu = menubar.addMenu('分析')
        self.analyzeMenu.addAction(octAction)

        # 添加像素校准菜单项
        calibrationAction = QAction('像素校准', self)
        calibrationAction.setStatusTip('打开像素校准工具来确定图像的转换系数')
        calibrationAction.triggered.connect(self.open_pixel_calibrator)
        self.analyzeMenu.addAction(calibrationAction)

        helpMenu = menubar.addMenu('帮助')
        
        aboutAction = QAction('关于', self)
        aboutAction.triggered.connect(self.showAboutDialog)
        helpMenu.addAction(aboutAction)
        
    def resetParameters(self):
        """重置所有参数为默认值"""
        self.params = {
            'gaussian_kernel': 5,
            'adaptive_block_size': 51,
            'adaptive_c': 5,
            'binary_threshold': 128,  # 添加全局二值化阈值参数
            'top_line_index': 1,  # 第几条水平线作为顶部
            'row_projection_threshold': 70,  # 行投影阈值
            'gap_min_width': 50,  # 缺口最小宽度
            'horizontal_kernel_size': 25,  # 水平线检测核大小
            'column_projection_threshold': 30,  # 列投影阈值
            'column_peak_window': 10,  # 峰值检测窗口大小
            'bottom_enhance_contrast': 1.5,  # 底部增强对比度
            'bottom_search_range': 0.8,  # 底部搜索范围（占图像高度的比例）
            'bottom_line_index': 0,  # 第几条水平线作为底部
            'short_line_min_length': 5,  # 短横线最小长度
            'short_line_min_white_ratio': 0.4,  # 短横线区域白色像素最小比例
            'short_line_max_white_ratio': 0.9,  # 短横线区域白色像素最大比例
            'invert_binary': False,  # 是否反转二值图像
            'pixel_to_um_x': 1.60,  # X方向（直径）像素到微米的转换比例
            'pixel_to_um_y': 1.94   # Y方向（深度）像素到微米的转换比例
        }
        
        # 更新全局变量
        global PIXEL_TO_UM_X, PIXEL_TO_UM_Y, PIXEL_TO_UM
        PIXEL_TO_UM_X = self.params['pixel_to_um_x']
        PIXEL_TO_UM_Y = self.params['pixel_to_um_y']
        PIXEL_TO_UM = self.params['pixel_to_um_x']  # 兼容旧代码，使用X方向值
        
        # 更新UI控件的值
        if hasattr(self, 'gaussianSlider'):
            self.gaussianSlider.setValue(self.params['gaussian_kernel'])
        
        if hasattr(self, 'adaptiveBlockSlider'):
            self.adaptiveBlockSlider.setValue(self.params['adaptive_block_size'])
        
        if hasattr(self, 'adaptiveCSlider'):
            self.adaptiveCSlider.setValue(self.params['adaptive_c'])
        
        if hasattr(self, 'topLineIndexSpin'):
            self.topLineIndexSpin.setValue(self.params['top_line_index'])
            
        if hasattr(self, 'rowProjSlider'):
            self.rowProjSlider.setValue(self.params['row_projection_threshold'])
            
        if hasattr(self, 'gapWidthSlider'):
            self.gapWidthSlider.setValue(self.params['gap_min_width'])
            
        if hasattr(self, 'horizontalKernelSlider'):
            self.horizontalKernelSlider.setValue(self.params['horizontal_kernel_size'])
            
        if hasattr(self, 'colProjSlider'):
            self.colProjSlider.setValue(self.params['column_projection_threshold'])
            
        if hasattr(self, 'peakWindowSlider'):
            self.peakWindowSlider.setValue(self.params['column_peak_window'])
            
        if hasattr(self, 'bottomContrastSlider'):
            self.bottomContrastSlider.setValue(int(self.params['bottom_enhance_contrast'] * 10))
            
        if hasattr(self, 'bottomSearchSlider'):
            self.bottomSearchSlider.setValue(int(self.params['bottom_search_range'] * 100))
            
        if hasattr(self, 'bottomLineIndexSpin'):
            self.bottomLineIndexSpin.setValue(self.params['bottom_line_index'])
            
        if hasattr(self, 'shortLineMinLengthSlider'):
            self.shortLineMinLengthSlider.setValue(self.params['short_line_min_length'])
            
        if hasattr(self, 'shortLineMinWhiteSlider'):
            self.shortLineMinWhiteSlider.setValue(int(self.params['short_line_min_white_ratio'] * 100))
            
        if hasattr(self, 'shortLineMaxWhiteSlider'):
            self.shortLineMaxWhiteSlider.setValue(int(self.params['short_line_max_white_ratio'] * 100))
            
        if hasattr(self, 'invertBinaryCheckbox'):
            self.invertBinaryCheckbox.setChecked(self.params['invert_binary'])
            
        # 检查是否有像素转换控件
        if hasattr(self, 'pixelToUmXSpinBox'):
            self.pixelToUmXSpinBox.setValue(self.params['pixel_to_um_x'])
            
        if hasattr(self, 'pixelToUmYSpinBox'):
            self.pixelToUmYSpinBox.setValue(self.params['pixel_to_um_y'])
            
        # 更新全局二值化阈值控件
        if hasattr(self, 'binaryThresholdSlider'):
            self.binaryThresholdSlider.setValue(self.params['binary_threshold'])
            
        # 处理可能不存在的变量名
        if hasattr(self, 'invertCheckbox') and not hasattr(self, 'invertBinaryCheckbox'):
            self.invertCheckbox.setChecked(self.params['invert_binary'])
        
        # 不再显示消息弹窗
        # QMessageBox.information(self, "参数重置", "所有参数已重置为默认值")
    
    def saveParameters(self):
        """保存当前参数到文件"""
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "保存参数", "", 
                                                "参数文件 (*.json);;所有文件 (*)", 
                                                options=options)
        if filePath:
            import json
            with open(filePath, 'w') as f:
                json.dump(self.params, f, indent=4)
            QMessageBox.information(self, "保存参数", f"参数已保存到: {filePath}")
    
    def loadParameters(self):
        """从文件加载参数"""
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "加载参数", "", 
                                                "参数文件 (*.json);;所有文件 (*)", 
                                                options=options)
        if filePath:
            import json
            try:
                with open(filePath, 'r') as f:
                    loaded_params = json.load(f)
                    
                # 更新参数
                self.params.update(loaded_params)
                
                # 更新UI控件的值
                if hasattr(self, 'gaussianSlider'):
                    self.gaussianSlider.setValue(self.params['gaussian_kernel'])
                
                if hasattr(self, 'adaptiveBlockSlider'):
                    self.adaptiveBlockSlider.setValue(self.params['adaptive_block_size'])
                
                if hasattr(self, 'adaptiveCSlider'):
                    self.adaptiveCSlider.setValue(self.params['adaptive_c'])
                
                if hasattr(self, 'topLineIndexSpin'):
                    self.topLineIndexSpin.setValue(self.params['top_line_index'])
                    
                if hasattr(self, 'rowProjSlider'):
                    self.rowProjSlider.setValue(self.params['row_projection_threshold'])
                    
                if hasattr(self, 'gapWidthSlider'):
                    self.gapWidthSlider.setValue(self.params['gap_min_width'])
                    
                if hasattr(self, 'horizontalKernelSlider'):
                    self.horizontalKernelSlider.setValue(self.params['horizontal_kernel_size'])
                    
                if hasattr(self, 'colProjSlider'):
                    self.colProjSlider.setValue(self.params['column_projection_threshold'])
                    
                if hasattr(self, 'peakWindowSlider'):
                    self.peakWindowSlider.setValue(self.params['column_peak_window'])
                    
                if hasattr(self, 'bottomContrastSlider'):
                    self.bottomContrastSlider.setValue(int(self.params['bottom_enhance_contrast'] * 10))
                    
                if hasattr(self, 'bottomSearchSlider'):
                    self.bottomSearchSlider.setValue(int(self.params['bottom_search_range'] * 100))
                    
                if hasattr(self, 'bottomLineIndexSpin'):
                    self.bottomLineIndexSpin.setValue(self.params['bottom_line_index'])
                    
                if hasattr(self, 'shortLineMinLengthSlider'):
                    self.shortLineMinLengthSlider.setValue(self.params['short_line_min_length'])
                    
                if hasattr(self, 'shortLineMinWhiteSlider'):
                    self.shortLineMinWhiteSlider.setValue(int(self.params['short_line_min_white_ratio'] * 100))
                    
                if hasattr(self, 'shortLineMaxWhiteSlider'):
                    self.shortLineMaxWhiteSlider.setValue(int(self.params['short_line_max_white_ratio'] * 100))
                    
                if hasattr(self, 'invertBinaryCheckbox'):
                    self.invertBinaryCheckbox.setChecked(self.params['invert_binary'])
                    
                # 处理可能不存在的变量名
                if hasattr(self, 'invertCheckbox') and not hasattr(self, 'invertBinaryCheckbox'):
                    self.invertCheckbox.setChecked(self.params['invert_binary'])
                
                QMessageBox.information(self, "加载参数", f"参数已从以下文件加载: {filePath}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载参数时出错: {str(e)}")
    
    def showAboutDialog(self):
        """显示关于对话框"""
        aboutDialog = QDialog(self)
        aboutDialog.setWindowTitle("关于")
        aboutDialog.setMinimumSize(400, 300)
        
        layout = QVBoxLayout()
        
        # 标题
        titleLabel = QLabel("孔洞测量应用程序")
        titleLabel.setStyleSheet("font-size: 18pt; font-weight: bold;")
        titleLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(titleLabel)
        
        # 版本和设计者信息
        infoLabel = QLabel("版本 Final Version")
        infoLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(infoLabel)
        
        # 日期和设计者
        designLabel = QLabel("2025.6.2 Design by 石殷睿")
        designLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(designLabel)
        
        # 比赛信息
        competitionLabel = QLabel("【舟行续石】为光电设计大赛开发")
        competitionLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(competitionLabel)
        
        # 描述
        descriptionLabel = QLabel(
            "本应用程序用于分析截面图像，测量孔的直径和深度。\n"
            "提供多种参数调整选项，以适应不同的图像特性。\n"
            "支持参数保存和加载，方便批量处理。"
        )
        descriptionLabel.setAlignment(Qt.AlignCenter)
        descriptionLabel.setWordWrap(True)
        layout.addWidget(descriptionLabel)
        
        # 添加一些空间
        layout.addSpacing(20)
        
        # 功能列表
        featuresLabel = QLabel("主要功能:")
        featuresLabel.setStyleSheet("font-weight: bold;")
        layout.addWidget(featuresLabel)
        
        featuresText = QLabel(
            "• 测量孔直径和深度\n"
            "• 可视化处理过程\n"
            "• 参数实时调整\n"
            "• 保存分析结果\n"
            "• 保存和加载处理参数\n"
            "• 支持OCT圆孔重建。\n"
            "• 支持手动测量。\n"
            "• 支持锥度计算与粗糙度分析。\n"
        )
        layout.addWidget(featuresText)
        
        # 确认按钮
        okButton = QPushButton("确定")
        okButton.clicked.connect(aboutDialog.accept)
        layout.addWidget(okButton)
        
        aboutDialog.setLayout(layout)
        aboutDialog.exec_()
        
    def initUI(self):
        self.setWindowTitle('孔洞检测程序 Final Version')
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F8F9FA;
                color: #333333;
                font-family: 'Microsoft YaHei', 'SimHei', 'Sans-serif';
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #DDDDDD;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4A86E8;
            }
            QPushButton {
                background-color: #4A86E8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #366ED8;
            }
            QPushButton:pressed {
                background-color: #2855B7;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
            QSlider::groove:horizontal {
                border: 1px solid #DDDDDD;
                height: 8px;
                background: #FFFFFF;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4A86E8;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::add-page:horizontal {
                background: #EEEEEE;
            }
            QLabel {
                color: #333333;
            }
            QSplitter::handle {
                background-color: #DDDDDD;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }
        """)
        
        # 设置窗口图标
        try:
            self.setWindowIcon(QIcon('icon.ico'))
        except:
            print("无法加载图标文件")
        
        # 创建主窗口布局
        mainWidget = QWidget()
        mainLayout = QHBoxLayout()
        mainLayout.setSpacing(10)
        mainLayout.setContentsMargins(10, 10, 10, 10)
        
        # 创建左侧控制面板
        controlPanel = QWidget()
        controlLayout = QVBoxLayout()
        controlLayout.setSpacing(10)
        
        # 文件操作区域
        fileGroup = QGroupBox('文件操作')
        fileLayout = QVBoxLayout()
        fileLayout.setSpacing(8)
        
        # 创建按钮样式
        btnLayout = QHBoxLayout()
        
        self.loadImageBtn = QPushButton('加载图像')
        self.loadImageBtn.setIcon(QIcon.fromTheme("document-open"))
        self.loadImageBtn.clicked.connect(self.loadImage)
        btnLayout.addWidget(self.loadImageBtn)
        
        self.loadFolderBtn = QPushButton('加载文件夹')
        self.loadFolderBtn.setIcon(QIcon.fromTheme("folder-open"))
        self.loadFolderBtn.clicked.connect(self.loadFolder)
        btnLayout.addWidget(self.loadFolderBtn)
        
        fileLayout.addLayout(btnLayout)
        
        # 添加图像导航控件并美化
        navLayout = QHBoxLayout()
        
        self.prevImageBtn = QToolButton()
        self.prevImageBtn.setText('◀')
        self.prevImageBtn.setToolTip('上一张')
        self.prevImageBtn.clicked.connect(self.loadPreviousImage)
        self.prevImageBtn.setEnabled(False)
        self.prevImageBtn.setStyleSheet("""
            QToolButton {
                background-color: #4A86E8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QToolButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        navLayout.addWidget(self.prevImageBtn)
        
        self.imageCountLabel = QLabel('0/0')
        self.imageCountLabel.setAlignment(Qt.AlignCenter)
        self.imageCountLabel.setStyleSheet("font-weight: bold;")
        navLayout.addWidget(self.imageCountLabel, 1)
        
        self.nextImageBtn = QToolButton()
        self.nextImageBtn.setText('▶')
        self.nextImageBtn.setToolTip('下一张')
        self.nextImageBtn.clicked.connect(self.loadNextImage)
        self.nextImageBtn.setEnabled(False)
        self.nextImageBtn.setStyleSheet("""
            QToolButton {
                background-color: #4A86E8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QToolButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        navLayout.addWidget(self.nextImageBtn)
        
        fileLayout.addLayout(navLayout)
        
        # 添加一个分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #CCCCCC;")
        fileLayout.addWidget(separator)
        
        btnSaveLayout = QHBoxLayout()
        
        self.saveResultBtn = QPushButton('保存结果')
        self.saveResultBtn.setIcon(QIcon.fromTheme("document-save"))
        self.saveResultBtn.clicked.connect(self.saveResult)
        self.saveResultBtn.setEnabled(False)
        btnSaveLayout.addWidget(self.saveResultBtn)
        
        self.saveAllBtn = QPushButton('批量保存')
        self.saveAllBtn.setIcon(QIcon.fromTheme("document-save-all"))
        self.saveAllBtn.clicked.connect(self.saveAllResults)
        self.saveAllBtn.setEnabled(False)
        btnSaveLayout.addWidget(self.saveAllBtn)
        
        fileLayout.addLayout(btnSaveLayout)
        
        self.exportDataBtn = QPushButton('导出测量数据')
        self.exportDataBtn.setIcon(QIcon.fromTheme("x-office-spreadsheet"))
        self.exportDataBtn.clicked.connect(self.exportMeasurementData)
        self.exportDataBtn.setEnabled(False)
        fileLayout.addWidget(self.exportDataBtn)
        
        fileGroup.setLayout(fileLayout)
        controlLayout.addWidget(fileGroup)
        
        # 结果显示放在参数前面
        resultGroup = QGroupBox('测量结果')
        resultLayout = QVBoxLayout()
        
        self.diameterLabel = QLabel("直径: -- μm")
        self.diameterLabel.setStyleSheet("font-size: 14px; font-weight: bold; color: #4A86E8;")
        self.diameterLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.diameterLabel)
        
        self.depthLabel = QLabel("深度: -- μm")
        self.depthLabel.setStyleSheet("font-size: 14px; font-weight: bold; color: #4A86E8;")
        self.depthLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.depthLabel)
        
        # 添加一个按钮来设置OCT的参考深度
        self.setDepthForOctBtn = QPushButton("设为OCT参考深度")
        self.setDepthForOctBtn.clicked.connect(self.set_reference_depth)
        self.setDepthForOctBtn.setEnabled(False)
        self.setDepthForOctBtn.setStyleSheet("font-size: 10px; padding: 4px 8px; margin: 0 20px;")
        self.setDepthForOctBtn.setToolTip("将当前计算出的深度值，作为OCT模块计算深径比的参考")
        resultLayout.addWidget(self.setDepthForOctBtn)
        
        # 添加新的微孔直径标准测量结果标签
        self.upperDiameterLabel = QLabel("上方0.1mm处直径: -- μm")
        self.upperDiameterLabel.setStyleSheet("font-size: 12px; color: #4A86E8;")
        self.upperDiameterLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.upperDiameterLabel)
        
        self.lowerDiameterLabel = QLabel("下方0.1mm处直径: -- μm")
        self.lowerDiameterLabel.setStyleSheet("font-size: 12px; color: #4A86E8;")
        self.lowerDiameterLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.lowerDiameterLabel)
        
        self.standardDiameterLabel = QLabel("标准直径: -- μm")
        self.standardDiameterLabel.setStyleSheet("font-size: 13px; font-weight: bold; color: #E8574A;")
        self.standardDiameterLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.standardDiameterLabel)
        
        self.ratioLabel = QLabel("深径比: --")
        self.ratioLabel.setStyleSheet("font-size: 13px; font-weight: bold; color: #E8574A;")
        self.ratioLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.ratioLabel)
        
        self.measureCountLabel = QLabel("测量次数: 0/3")
        self.measureCountLabel.setStyleSheet("font-size: 12px; color: #4A86E8;")
        self.measureCountLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.measureCountLabel)
        
        resultGroup.setLayout(resultLayout)
        controlLayout.addWidget(resultGroup)
        
        # 参数调整区域
        paramGroup = QGroupBox('参数调整')
        paramLayout = QGridLayout()
        paramLayout.setVerticalSpacing(8)
        paramLayout.setHorizontalSpacing(10)
        
        # 创建一个滚动区域，使得参数调整可以在小屏幕上滚动显示
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setStyleSheet("QScrollArea { border: none; }")
        
        paramWidget = QWidget()
        paramWidget.setLayout(paramLayout)
        scrollArea.setWidget(paramWidget)
        
        # 统一标签样式
        labelStyle = "min-width: 120px;"
        
        # 高斯核大小
        gaussianLabel = QLabel("高斯滤波核大小:")
        gaussianLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(gaussianLabel, 0, 0)
        self.gaussianSlider = QSlider(Qt.Horizontal)
        self.gaussianSlider.setMinimum(1)
        self.gaussianSlider.setMaximum(21)
        self.gaussianSlider.setValue(5)
        self.gaussianSlider.setSingleStep(2)
        self.gaussianSlider.valueChanged.connect(self.updateGaussianKernel)
        paramLayout.addWidget(self.gaussianSlider, 0, 1)
        self.gaussianLabel = QLabel("5")
        self.gaussianLabel.setMinimumWidth(30)
        self.gaussianLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.gaussianLabel, 0, 2)
        
        # 自适应二值化块大小
        adaptiveBlockLabel = QLabel("二值化块大小:")
        adaptiveBlockLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(adaptiveBlockLabel, 1, 0)
        self.adaptiveBlockSlider = QSlider(Qt.Horizontal)
        self.adaptiveBlockSlider.setMinimum(3)
        self.adaptiveBlockSlider.setMaximum(101)
        self.adaptiveBlockSlider.setValue(51)
        self.adaptiveBlockSlider.setSingleStep(2)
        self.adaptiveBlockSlider.valueChanged.connect(self.updateAdaptiveBlockSize)
        paramLayout.addWidget(self.adaptiveBlockSlider, 1, 1)
        self.adaptiveBlockLabel = QLabel("51")
        self.adaptiveBlockLabel.setMinimumWidth(30)
        self.adaptiveBlockLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.adaptiveBlockLabel, 1, 2)
        
        # 自适应二值化常数C
        adaptiveCLabel = QLabel("二值化常数C:")
        adaptiveCLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(adaptiveCLabel, 2, 0)
        self.adaptiveCSlider = QSlider(Qt.Horizontal)
        self.adaptiveCSlider.setMinimum(1)
        self.adaptiveCSlider.setMaximum(30)
        self.adaptiveCSlider.setValue(5)
        self.adaptiveCSlider.valueChanged.connect(self.updateAdaptiveC)
        paramLayout.addWidget(self.adaptiveCSlider, 2, 1)
        self.adaptiveCLabel = QLabel("5")
        self.adaptiveCLabel.setMinimumWidth(30)
        self.adaptiveCLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.adaptiveCLabel, 2, 2)
        
        # 全局二值化阈值
        binaryThresholdLabel = QLabel("全局二值化阈值:")
        binaryThresholdLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(binaryThresholdLabel, 3, 0)
        self.binaryThresholdSlider = QSlider(Qt.Horizontal)
        self.binaryThresholdSlider.setMinimum(0)
        self.binaryThresholdSlider.setMaximum(255)
        self.binaryThresholdSlider.setValue(128)
        self.binaryThresholdSlider.valueChanged.connect(self.updateBinaryThreshold)
        paramLayout.addWidget(self.binaryThresholdSlider, 3, 1)
        self.binaryThresholdLabel = QLabel("128")
        self.binaryThresholdLabel.setMinimumWidth(30)
        self.binaryThresholdLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.binaryThresholdLabel, 3, 2)
        
        # 顶部线索引
        topLineLabel = QLabel("顶部线索引:")
        topLineLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(topLineLabel, 4, 0)
        self.topLineIndexSpin = QSpinBox()
        self.topLineIndexSpin.setMinimum(0)
        self.topLineIndexSpin.setMaximum(10)
        self.topLineIndexSpin.setValue(1)
        self.topLineIndexSpin.valueChanged.connect(self.updateTopLineIndex)
        self.topLineIndexSpin.setStyleSheet("""
            QSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                background: white;
            }
        """)
        paramLayout.addWidget(self.topLineIndexSpin, 4, 1)
        
        # 行投影阈值
        rowProjLabel = QLabel("行投影阈值(%):")
        rowProjLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(rowProjLabel, 5, 0)
        self.rowProjSlider = QSlider(Qt.Horizontal)
        self.rowProjSlider.setMinimum(10)
        self.rowProjSlider.setMaximum(90)
        self.rowProjSlider.setValue(70)
        self.rowProjSlider.valueChanged.connect(self.updateRowProjThreshold)
        paramLayout.addWidget(self.rowProjSlider, 5, 1)
        self.rowProjLabel = QLabel("70%")
        self.rowProjLabel.setMinimumWidth(30)
        self.rowProjLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.rowProjLabel, 5, 2)
        
        # 缺口最小宽度
        gapWidthLabel = QLabel("缺口最小宽度:")
        gapWidthLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(gapWidthLabel, 6, 0)
        self.gapWidthSlider = QSlider(Qt.Horizontal)
        self.gapWidthSlider.setMinimum(10)
        self.gapWidthSlider.setMaximum(200)
        self.gapWidthSlider.setValue(50)
        self.gapWidthSlider.valueChanged.connect(self.updateGapMinWidth)
        paramLayout.addWidget(self.gapWidthSlider, 6, 1)
        self.gapWidthLabel = QLabel("50")
        self.gapWidthLabel.setMinimumWidth(30)
        self.gapWidthLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.gapWidthLabel, 6, 2)
        
        # 水平线检测核大小
        horizontalKernelLabel = QLabel("水平线检测核:")
        horizontalKernelLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(horizontalKernelLabel, 7, 0)
        self.horizontalKernelSlider = QSlider(Qt.Horizontal)
        self.horizontalKernelSlider.setMinimum(5)
        self.horizontalKernelSlider.setMaximum(51)
        self.horizontalKernelSlider.setValue(25)
        self.horizontalKernelSlider.setSingleStep(2)
        self.horizontalKernelSlider.valueChanged.connect(self.updateHorizontalKernelSize)
        paramLayout.addWidget(self.horizontalKernelSlider, 7, 1)
        self.horizontalKernelLabel = QLabel("25")
        self.horizontalKernelLabel.setMinimumWidth(30)
        self.horizontalKernelLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.horizontalKernelLabel, 7, 2)
        
        # 列投影阈值
        colProjLabel = QLabel("列投影阈值(%):")
        colProjLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(colProjLabel, 8, 0)
        self.colProjSlider = QSlider(Qt.Horizontal)
        self.colProjSlider.setMinimum(10)
        self.colProjSlider.setMaximum(90)
        self.colProjSlider.setValue(30)
        self.colProjSlider.valueChanged.connect(self.updateColProjThreshold)
        paramLayout.addWidget(self.colProjSlider, 8, 1)
        self.colProjLabel = QLabel("30%")
        self.colProjLabel.setMinimumWidth(30)
        self.colProjLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.colProjLabel, 8, 2)
        
        # 创建分隔线
        separatorParam = QFrame()
        separatorParam.setFrameShape(QFrame.HLine)
        separatorParam.setFrameShadow(QFrame.Sunken)
        separatorParam.setStyleSheet("background-color: #DDDDDD; margin: 5px 0;")
        paramLayout.addWidget(separatorParam, 9, 0, 1, 3)
        
        # 峰值检测窗口大小
        peakWindowLabel = QLabel("峰值检测窗口:")
        peakWindowLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(peakWindowLabel, 10, 0)
        self.peakWindowSlider = QSlider(Qt.Horizontal)
        self.peakWindowSlider.setMinimum(3)
        self.peakWindowSlider.setMaximum(31)
        self.peakWindowSlider.setValue(10)
        self.peakWindowSlider.setSingleStep(2)
        self.peakWindowSlider.valueChanged.connect(self.updatePeakWindow)
        paramLayout.addWidget(self.peakWindowSlider, 10, 1)
        self.peakWindowLabel = QLabel("10")
        self.peakWindowLabel.setMinimumWidth(30)
        self.peakWindowLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.peakWindowLabel, 10, 2)
        
        # 底部增强对比度
        bottomContrastLabel = QLabel("底部增强对比度:")
        bottomContrastLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(bottomContrastLabel, 11, 0)
        self.bottomContrastSlider = QSlider(Qt.Horizontal)
        self.bottomContrastSlider.setMinimum(10)
        self.bottomContrastSlider.setMaximum(40)
        self.bottomContrastSlider.setValue(15)
        self.bottomContrastSlider.setSingleStep(1)
        self.bottomContrastSlider.valueChanged.connect(self.updateBottomContrast)
        paramLayout.addWidget(self.bottomContrastSlider, 11, 1)
        self.bottomContrastLabel = QLabel("1.5")
        self.bottomContrastLabel.setMinimumWidth(30)
        self.bottomContrastLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.bottomContrastLabel, 11, 2)
        
        # 底部搜索范围
        bottomSearchLabel = QLabel("底部搜索范围:")
        bottomSearchLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(bottomSearchLabel, 12, 0)
        self.bottomSearchSlider = QSlider(Qt.Horizontal)
        self.bottomSearchSlider.setMinimum(10)
        self.bottomSearchSlider.setMaximum(100)
        self.bottomSearchSlider.setValue(80)
        self.bottomSearchSlider.setSingleStep(5)
        self.bottomSearchSlider.valueChanged.connect(self.updateBottomSearchRange)
        paramLayout.addWidget(self.bottomSearchSlider, 12, 1)
        self.bottomSearchLabel = QLabel("80%")
        self.bottomSearchLabel.setMinimumWidth(30)
        self.bottomSearchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.bottomSearchLabel, 12, 2)
        
        # 底部线索引
        bottomLineLabel = QLabel("底部线索引:")
        bottomLineLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(bottomLineLabel, 13, 0)
        self.bottomLineIndexSpin = QSpinBox()
        self.bottomLineIndexSpin.setMinimum(0)
        self.bottomLineIndexSpin.setMaximum(10)
        self.bottomLineIndexSpin.setValue(0)
        self.bottomLineIndexSpin.valueChanged.connect(self.updateBottomLineIndex)
        self.bottomLineIndexSpin.setStyleSheet("""
            QSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                background: white;
            }
        """)
        paramLayout.addWidget(self.bottomLineIndexSpin, 13, 1)
        
        # 创建另一个分隔线
        separatorParam2 = QFrame()
        separatorParam2.setFrameShape(QFrame.HLine)
        separatorParam2.setFrameShadow(QFrame.Sunken)
        separatorParam2.setStyleSheet("background-color: #DDDDDD; margin: 5px 0;")
        paramLayout.addWidget(separatorParam2, 14, 0, 1, 3)
        
        # 图像旋转角度滑动条
        rotationLabel = QLabel("旋转角度:")
        rotationLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(rotationLabel, 15, 0)
        
        # 创建一个水平布局来组合滑动条和输入框
        rotationLayout = QHBoxLayout()
        
        self.rotationSlider = QSlider(Qt.Horizontal)
        self.rotationSlider.setMinimum(-180)
        self.rotationSlider.setMaximum(180)
        self.rotationSlider.setValue(0)
        self.rotationSlider.setSingleStep(5)
        self.rotationSlider.setTickInterval(45)
        self.rotationSlider.setTickPosition(QSlider.TicksBelow)
        self.rotationSlider.valueChanged.connect(self.updateRotationAngle)
        rotationLayout.addWidget(self.rotationSlider)
        
        # 添加角度输入框
        self.rotationInput = QSpinBox()
        self.rotationInput.setMinimum(-180)
        self.rotationInput.setMaximum(180)
        self.rotationInput.setValue(0)
        self.rotationInput.setSuffix("°")
        self.rotationInput.setAlignment(Qt.AlignRight)
        self.rotationInput.setFixedWidth(60)
        self.rotationInput.setStyleSheet("""
            QSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                background: white;
            }
        """)
        self.rotationInput.valueChanged.connect(self.applyInputRotation)
        rotationLayout.addWidget(self.rotationInput)
        
        # 将布局添加到参数网格
        paramLayout.addLayout(rotationLayout, 15, 1, 1, 2)
        
        # 短横线最小长度
        shortLineMinLengthLabel = QLabel("短横线最小长度:")
        shortLineMinLengthLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(shortLineMinLengthLabel, 17, 0)
        self.shortLineMinLengthSlider = QSlider(Qt.Horizontal)
        self.shortLineMinLengthSlider.setMinimum(1)
        self.shortLineMinLengthSlider.setMaximum(20)
        self.shortLineMinLengthSlider.setValue(5)
        self.shortLineMinLengthSlider.valueChanged.connect(self.updateShortLineMinLength)
        paramLayout.addWidget(self.shortLineMinLengthSlider, 17, 1)
        self.shortLineMinLengthLabel = QLabel("5")
        self.shortLineMinLengthLabel.setMinimumWidth(30)
        self.shortLineMinLengthLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.shortLineMinLengthLabel, 17, 2)
        
        # 短横线白色像素比例下限
        shortLineMinWhiteLabel = QLabel("白色比例下限:")
        shortLineMinWhiteLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(shortLineMinWhiteLabel, 18, 0)
        self.shortLineMinWhiteSlider = QSlider(Qt.Horizontal)
        self.shortLineMinWhiteSlider.setMinimum(10)
        self.shortLineMinWhiteSlider.setMaximum(60)
        self.shortLineMinWhiteSlider.setValue(40)
        self.shortLineMinWhiteSlider.valueChanged.connect(self.updateShortLineMinWhite)
        paramLayout.addWidget(self.shortLineMinWhiteSlider, 18, 1)
        self.shortLineMinWhiteLabel = QLabel("0.4")
        self.shortLineMinWhiteLabel.setMinimumWidth(30)
        self.shortLineMinWhiteLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.shortLineMinWhiteLabel, 18, 2)
        
        # 短横线白色像素比例上限
        shortLineMaxWhiteLabel = QLabel("白色比例上限:")
        shortLineMaxWhiteLabel.setStyleSheet(labelStyle)
        paramLayout.addWidget(shortLineMaxWhiteLabel, 19, 0)
        self.shortLineMaxWhiteSlider = QSlider(Qt.Horizontal)
        self.shortLineMaxWhiteSlider.setMinimum(60)
        self.shortLineMaxWhiteSlider.setMaximum(100)
        self.shortLineMaxWhiteSlider.setValue(90)
        self.shortLineMaxWhiteSlider.valueChanged.connect(self.updateShortLineMaxWhite)
        paramLayout.addWidget(self.shortLineMaxWhiteSlider, 19, 1)
        self.shortLineMaxWhiteLabel = QLabel("0.9")
        self.shortLineMaxWhiteLabel.setMinimumWidth(30)
        self.shortLineMaxWhiteLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        paramLayout.addWidget(self.shortLineMaxWhiteLabel, 19, 2)
        
        # 添加像素转换比例设置
        pixelConversionGroup = QGroupBox("像素转换设置")
        pixelConversionGroup.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #4A86E8;
                margin-top: 12px;
            }
        """)
        pixelConversionLayout = QGridLayout()
        pixelConversionLayout.setVerticalSpacing(8)
        
        # X方向（直径）转换比例
        pixelToUmXLabel = QLabel("X方向（直径）转换比例:")
        pixelToUmXLabel.setToolTip("水平方向上每像素对应的微米数")
        pixelConversionLayout.addWidget(pixelToUmXLabel, 0, 0)
        
        self.pixelToUmXSpinBox = QDoubleSpinBox()
        self.pixelToUmXSpinBox.setMinimum(0.1)
        self.pixelToUmXSpinBox.setMaximum(10.0)
        self.pixelToUmXSpinBox.setSingleStep(0.1)
        self.pixelToUmXSpinBox.setValue(2.0)
        self.pixelToUmXSpinBox.setSuffix(" μm/px")
        self.pixelToUmXSpinBox.setAlignment(Qt.AlignRight)
        self.pixelToUmXSpinBox.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                background: white;
            }
        """)
        self.pixelToUmXSpinBox.valueChanged.connect(self.updatePixelToUmX)
        pixelConversionLayout.addWidget(self.pixelToUmXSpinBox, 0, 1)
        
        # Y方向（深度）转换比例
        pixelToUmYLabel = QLabel("Y方向（深度）转换比例:")
        pixelToUmYLabel.setToolTip("垂直方向上每像素对应的微米数")
        pixelConversionLayout.addWidget(pixelToUmYLabel, 1, 0)
        
        self.pixelToUmYSpinBox = QDoubleSpinBox()
        self.pixelToUmYSpinBox.setMinimum(0.1)
        self.pixelToUmYSpinBox.setMaximum(10.0)
        self.pixelToUmYSpinBox.setSingleStep(0.1)
        self.pixelToUmYSpinBox.setValue(2.0)
        self.pixelToUmYSpinBox.setSuffix(" μm/px")
        self.pixelToUmYSpinBox.setAlignment(Qt.AlignRight)
        self.pixelToUmYSpinBox.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                background: white;
            }
        """)
        self.pixelToUmYSpinBox.valueChanged.connect(self.updatePixelToUmY)
        pixelConversionLayout.addWidget(self.pixelToUmYSpinBox, 1, 1)
        
        pixelConversionGroup.setLayout(pixelConversionLayout)
        paramLayout.addWidget(pixelConversionGroup, 20, 0, 1, 3)
        
        # 反转二值图像复选框
        self.invertBinaryCheckbox = QCheckBox("反转二值图像")
        self.invertBinaryCheckbox.setChecked(False)
        self.invertBinaryCheckbox.stateChanged.connect(self.updateInvertBinary)
        self.invertBinaryCheckbox.setStyleSheet("""
            QCheckBox {
                spacing: 5px;
                font-weight: normal;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #CCCCCC;
                background: white;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #4A86E8;
                background: #4A86E8;
                border-radius: 3px;
            }
        """)
        paramLayout.addWidget(self.invertBinaryCheckbox, 21, 0, 1, 3)
        
        # 添加重置参数按钮
        resetParamsBtn = QPushButton("重置参数")
        resetParamsBtn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)
        resetParamsBtn.clicked.connect(self.resetParameters)
        paramLayout.addWidget(resetParamsBtn, 22, 0, 1, 3)
        
        # 把滚动区域添加到参数分组中
        paramGroupLayout = QVBoxLayout()
        paramGroupLayout.addWidget(scrollArea)
        paramGroup.setLayout(paramGroupLayout)
        
        controlLayout.addWidget(paramGroup)
        
        # 处理按钮（已移除）
        # self.processBtn = QPushButton('处理图像')
        # self.processBtn.setStyleSheet("""
        #     QPushButton {
        #         background-color: #28A745;
        #         color: white;
        #         padding: 8px 15px;
        #         font-weight: bold;
        #         font-size: 14px;
        #     }
        #     QPushButton:hover {
        #         background-color: #218838;
        #     }
        #     QPushButton:pressed {
        #         background-color: #1E7E34;
        #     }
        #     QPushButton:disabled {
        #         background-color: #CCCCCC;
        #     }
        # """)
        # self.processBtn.clicked.connect(self.processImage)
        # self.processBtn.setEnabled(False)
        # controlLayout.addWidget(self.processBtn)
        
        # 添加弹性空间
        controlLayout.addStretch(1)
        
        controlPanel.setLayout(controlLayout)
        controlPanel.setFixedWidth(350)
        
        # 创建右侧图像显示区域
        displayPanel = QSplitter(Qt.Vertical)
        displayPanel.setStyleSheet("""
            QSplitter {
                background-color: transparent;
            }
            QWidget {
                background-color: white;
                border-radius: 6px;
            }
        """)
        
        # 标题样式
        titleStyle = "color: #4A86E8; font-weight: bold; font-size: 14px; padding: 8px; background-color: transparent;"
        
        # 原始图像区域
        originalContainer = QWidget()
        originalContainer.setStyleSheet("border: 2px solid #DDDDDD; border-radius: 6px; background-color: white;")
        originalLayout = QVBoxLayout(originalContainer)
        originalLayout.setContentsMargins(10, 10, 10, 10)
        
        originalTitle = QLabel("原始图像")
        originalTitle.setAlignment(Qt.AlignCenter)
        originalTitle.setStyleSheet(titleStyle)
        originalLayout.addWidget(originalTitle)
        
        self.originalImageLabel = ImageLabel()
        originalLayout.addWidget(self.originalImageLabel)
        displayPanel.addWidget(originalContainer)
        
        # 创建二值图像和处理结果的水平分割器
        lowerPanel = QSplitter(Qt.Horizontal)
        lowerPanel.setStyleSheet("background-color: transparent;")
        
        # 二值图像显示
        binaryContainer = QWidget()
        binaryContainer.setStyleSheet("border: 2px solid #DDDDDD; border-radius: 6px; background-color: white;")
        binaryLayout = QVBoxLayout(binaryContainer)
        binaryLayout.setContentsMargins(10, 10, 10, 10)
        
        binaryTitle = QLabel("二值图像")
        binaryTitle.setAlignment(Qt.AlignCenter)
        binaryTitle.setStyleSheet(titleStyle)
        binaryLayout.addWidget(binaryTitle)
        
        self.binaryImageLabel = ImageLabel()
        binaryLayout.addWidget(self.binaryImageLabel)
        lowerPanel.addWidget(binaryContainer)
        
        # 处理结果显示
        resultContainer = QWidget()
        resultContainer.setStyleSheet("border: 2px solid #DDDDDD; border-radius: 6px; background-color: white;")
        resultLayout = QVBoxLayout(resultContainer)
        resultLayout.setContentsMargins(10, 10, 10, 10)
        
        resultTitle = QLabel("处理结果")
        resultTitle.setAlignment(Qt.AlignCenter)
        resultTitle.setStyleSheet(titleStyle)
        resultLayout.addWidget(resultTitle)
        
        self.resultImageLabel = ImageLabel()
        resultLayout.addWidget(self.resultImageLabel)
        lowerPanel.addWidget(resultContainer)
        
        displayPanel.addWidget(lowerPanel)
        
        # 设置分割器的初始大小
        displayPanel.setSizes([400, 400])
        lowerPanel.setSizes([500, 500])
        
        # 将控制面板和图像显示区域添加到主布局
        mainLayout.addWidget(controlPanel)
        mainLayout.addWidget(displayPanel, 1)  # 图像显示区域应该占据更多空间
        
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
    
    @pyqtSlot()
    def loadImage(self):
        """加载单个图像"""
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", 
                                                "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
                                                options=options)
        if filePath:
            self.image_files = [filePath]
            self.current_image_index = 0
            self.loadImageAtCurrentIndex()
            self.updateNavigationButtons()
            
    @pyqtSlot()
    def loadFolder(self):
        """加载文件夹中的所有图像"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder_path:
            # 获取所有支持的图像文件
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            self.image_files = []
            
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                    self.image_files.append(file_path)
            
            if self.image_files:
                # 按文件名排序
                self.image_files.sort()
                self.current_image_index = 0
                self.loadImageAtCurrentIndex()
                self.updateNavigationButtons()
                QMessageBox.information(self, "文件夹加载", f"已找到 {len(self.image_files)} 张图像")
            else:
                QMessageBox.warning(self, "文件夹加载", "所选文件夹中没有找到支持的图像文件")
    
    def resetMeasurements(self):
        """重置测量计数和历史记录"""
        self.measurement_count = 0
        self.diameter_history = []
        self.upper_diameter_at_01mm = 0
        self.lower_diameter_at_01mm = 0
        self.standard_diameter = 0
        self.depth_diameter_ratio = 0
        self.measureCountLabel.setText("测量次数: 0/3")
    
    def loadImageAtCurrentIndex(self):
        """加载当前索引位置的图像"""
        if not self.image_files:
            return
            
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            self.current_image_index = 0
            
        file_path = self.image_files[self.current_image_index]
        
        try:
            # 使用PIL库加载图像，解决中文路径问题
            from PIL import Image
            pil_image = Image.open(file_path)
            # 转换为灰度图
            pil_image_gray = pil_image.convert('L')
            # 转换为numpy数组供OpenCV使用
            img = np.array(pil_image_gray)
            
            if img is None or img.size == 0:
                QMessageBox.warning(self, "图像加载错误", f"无法加载图像: {file_path}")
                return
                
            # 更新图像和路径
            self.original_image = img.copy()
            self.current_image_path = file_path
            self.result_image = None
            self.binary_image = None
            
            # 重置测量结果
            self.resetMeasurements()
            
            # 更新UI
            self.displayImage(img, self.originalImageLabel, f"原图 - {os.path.basename(file_path)}")
            
            # 更新状态栏
            self.statusbar.showMessage(f"已加载图像: {file_path}, 尺寸: {img.shape}")
            print(f"已加载图像: {file_path}, 尺寸: {img.shape}")
            
            # 启用裁剪菜单项
            if hasattr(self, 'cropAction'):
                self.cropAction.setEnabled(True)
            
            # 更新导航按钮状态
            self.updateNavigationButtons()
            
            # 自动处理图像，不需要用户点击处理按钮
            self.processImage()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def loadPreviousImage(self):
        """加载上一张图像"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.loadImageAtCurrentIndex()
            self.updateNavigationButtons()
    
    def loadNextImage(self):
        """加载下一张图像"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.loadImageAtCurrentIndex()
            self.updateNavigationButtons()
    
    def updateNavigationButtons(self):
        """更新导航按钮状态和图像计数器"""
        if self.image_files:
            self.prevImageBtn.setEnabled(self.current_image_index > 0)
            self.nextImageBtn.setEnabled(self.current_image_index < len(self.image_files) - 1)
            self.imageCountLabel.setText(f"{self.current_image_index + 1}/{len(self.image_files)}")
        else:
            self.prevImageBtn.setEnabled(False)
            self.nextImageBtn.setEnabled(False)
            self.imageCountLabel.setText("0/0")
    
    @pyqtSlot()
    def saveResult(self):
        """保存处理结果"""
        if self.result_image is None:
            return
        
        # 获取当前图像的文件名
        if self.current_image_path:
            base_filename = os.path.basename(self.current_image_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            
            # 如果图像已被裁剪，添加标识
            if hasattr(self, 'original_image_before_crop') and self.original_image_before_crop is not None:
                filename_no_ext += "_cropped"
                
            # 创建默认的保存路径
            default_save_path = os.path.join(output_dir, f"{filename_no_ext}_result.png")
            
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "保存结果", default_save_path, 
                                                    "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)", 
                                                    options=options)
            if filePath:
                self.saveMeasurementResult(filePath)
                print(f"结果已保存到: {filePath}")
    
    def saveMeasurementResult(self, filePath):
        """保存带测量结果的图像"""
        if self.result_image is None:
            return False
            
        # 将结果图像转换为RGB
        rgb_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
        
        # 使用matplotlib添加标注
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title("孔测量结果", fontsize=14)
        
        # 使用支持中文的字体
        font_props = fm.FontProperties(family=plt.rcParams['font.sans-serif'][0])
        
        hole_center_x = (self.hole_start + self.hole_end) // 2
        plt.text(hole_center_x - 100, self.upper_surface_row - 30, 
                 f"直径: {self.hole_diameter:.2f} μm", color='red', fontsize=12,
                 fontproperties=font_props)
        plt.text(hole_center_x + 30, (self.upper_surface_row + self.bottom_surface_row) // 2, 
                 f"深度: {self.hole_depth:.2f} μm", color='red', fontsize=12,
                 fontproperties=font_props)
        
        plt.tight_layout()
        plt.savefig(filePath, dpi=200)
        plt.close()
        
        return True
    
    def displayImage(self, img, label, title=None):
        """在Qt标签中显示图像"""
        if img is None:
            print("无法显示：图像为空")
            return
        
        try:
            # 如果是灰度图像，转换为RGB
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建QImage
            h, w, c = display_img.shape
            bytesPerLine = 3 * w
            qImg = QImage(display_img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
            
            # 创建QPixmap并设置到标签
            pixmap = QPixmap.fromImage(qImg)
            if pixmap.isNull():
                print("错误：创建的QPixmap为空")
                return
                
            # 设置标题和图像
            if title:
                label.setText(title)
            
            # 设置图像到标签
            label.setPixmap(pixmap)
            
            print(f"图像显示成功，尺寸: {w}x{h}")
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    @pyqtSlot()
    def processImage(self):
        """处理图像并更新显示"""
        if self.original_image is None:
            return
        
        try:
            # 应用高斯滤波减少噪声
            gaussian_kernel_size = self.params['gaussian_kernel']
            blurred = cv2.GaussianBlur(self.original_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
            
            # 应用自适应二值化
            adaptive_block_size = self.params['adaptive_block_size']
            adaptive_c = self.params['adaptive_c']
            binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
            
            # 全局OTSU二值化
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 使用自定义全局阈值进行二值化
            _, binary_global = cv2.threshold(blurred, self.params['binary_threshold'], 255, cv2.THRESH_BINARY)
            
            # 合并三种二值化结果
            binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
            binary_combined = cv2.bitwise_and(binary_combined, binary_global)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            binary_opened = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel)
            binary_final = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel)
            
            # 如果需要反转二值图像
            if self.params['invert_binary']:
                binary_final = 255 - binary_final
            
            self.binary_image = binary_final
            # 确保二值图像显示正确
            self.displayImage(self.binary_image, self.binaryImageLabel, "二值图像")
            
            # 根据当前激活的模式选择合适的测量算法
            if self.is_no_gap_measure_active:
                self.measureWithoutGap()
            elif hasattr(self, 'is_image_rotated') and self.is_image_rotated:
                # 使用专为旋转图像设计的算法
                self.detect_hole_in_rotated_image()
            else:
                # 使用标准算法
                self.detect_hole_dimensions()
            
            # 确保结果图像显示正确
            if self.result_image is not None:
                self.displayImage(self.result_image, self.resultImageLabel, "测量结果")
            
            # 启用保存按钮和导出数据按钮
            self.saveResultBtn.setEnabled(True)
            self.exportDataBtn.setEnabled(True)
            
            # 更新状态栏
            self.statusbar.showMessage("图像处理完成")
            
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusbar.showMessage(f"处理图像出错: {str(e)}")
            QMessageBox.critical(self, "处理错误", f"处理图像时出错: {str(e)}")
    
    def detect_hole_dimensions(self):
        """检测孔的尺寸"""
        if self.original_image is None or self.binary_image is None:
            return
        
        # 创建结果图像
        result_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        
        # 计算行投影以找到水平线
        row_projection = np.sum(self.binary_image, axis=1)
        row_projection_smooth = np.convolve(row_projection, np.ones(5)/5, mode='same')
        
        # 找到所有显著的水平线
        threshold = np.max(row_projection_smooth) * (self.params['row_projection_threshold'] / 100.0)
        significant_rows = np.where(row_projection_smooth > threshold)[0]
        
        # 对找到的行进行分组
        horizontal_lines = []
        current_group = []
        
        for i in range(len(significant_rows)):
            if i == 0 or significant_rows[i] - significant_rows[i-1] <= 5:
                current_group.append(significant_rows[i])
            else:
                if current_group:
                    horizontal_lines.append(int(np.mean(current_group)))
                    current_group = [significant_rows[i]]
        
        # 添加最后一组
        if current_group:
            horizontal_lines.append(int(np.mean(current_group)))
        
        # 按从上到下排序水平线
        horizontal_lines.sort()
        
        # 在图像上标记所有检测到的水平线，用不同颜色标记顶部和底部
        for i, line_pos in enumerate(horizontal_lines):
            color = (0, 255, 0)  # 默认绿色
            if i == self.params['top_line_index']:
                color = (0, 255, 255)  # 黄色标记顶部线
            elif i == self.params['bottom_line_index'] and self.params['bottom_line_index'] < len(horizontal_lines):
                color = (255, 0, 255)  # 品红色标记底部线
            cv2.line(result_img, (0, line_pos), (self.original_image.shape[1], line_pos), color, 1)
        
        # 确定孔的顶部位置
        top_line_index = self.params['top_line_index']
        if len(horizontal_lines) > top_line_index:
            self.upper_surface_row = horizontal_lines[top_line_index]
        else:
            self.upper_surface_row = self.original_image.shape[0] // 4
        
        # 在顶部位置找缺口 - 重新实现缺口检测算法
        # 首先检查二值化图像中选定行的像素值
        search_range = 10
        best_row = self.upper_surface_row
        max_gap_width = 0
        self.hole_start = 0
        self.hole_end = 0
        
        # 在选定行附近搜索最明显的缺口
        for i in range(max(0, self.upper_surface_row - search_range), 
                      min(self.original_image.shape[0], self.upper_surface_row + search_range)):
            # 获取当前行的像素值
            line = self.binary_image[i, :]
            
            # 寻找黑白转换点
            transitions = np.where(np.diff(line) != 0)[0]
            # 如果检测到的转换点不少于2个
            if len(transitions) >= 2:
                # 查找连续的白-黑-白模式（白线上的黑色缺口）
                for j in range(0, len(transitions) - 1):
                    # 确认这是白到黑的转换（缺口开始）
                    if line[transitions[j]] == 255 and transitions[j] + 1 < len(line) and line[transitions[j] + 1] == 0:
                        # 查找黑到白的转换点（缺口结束）
                        for k in range(j + 1, len(transitions)):
                            if line[transitions[k]] == 0 and transitions[k] + 1 < len(line) and line[transitions[k] + 1] == 255:
                                # 计算缺口宽度
                                gap_width = transitions[k] - transitions[j]
                                
                                # 检查缺口是否符合最小宽度要求，并且是否是目前找到的最宽的缺口
                                if gap_width > self.params['gap_min_width'] and gap_width > max_gap_width:
                                    max_gap_width = gap_width
                                    best_row = i
                                    self.hole_start = transitions[j]
                                    self.hole_end = transitions[k]
                                break
        
        # 如果找到了有效的缺口
        if max_gap_width > 0:
            self.upper_surface_row = best_row
            self.hole_diameter = (self.hole_end - self.hole_start) * PIXEL_TO_UM_X
        else:
            # 备选检测方法：寻找行中最长的黑色区域
            for i in range(max(0, self.upper_surface_row - search_range), 
                          min(self.original_image.shape[0], self.upper_surface_row + search_range)):
                line = self.binary_image[i, :]
                black_regions = []
                start = None
                
                # 寻找所有黑色区域
                for j in range(len(line)):
                    if line[j] == 0 and start is None:  # 黑色区域开始
                        start = j
                    elif line[j] == 255 and start is not None:  # 黑色区域结束
                        black_regions.append((start, j))
                        start = None
                
                # 处理最后一个黑色区域（如果存在）
                if start is not None:
                    black_regions.append((start, len(line)))
                
                # 在黑色区域中找最宽的一个
                for region in black_regions:
                    region_width = region[1] - region[0]
                    # 检查该区域是否在一条明显的白线上
                    if region[0] > 0 and region[1] < len(line) - 1 and line[region[0] - 1] == 255 and line[region[1]] == 255:
                        if region_width > self.params['gap_min_width'] and region_width > max_gap_width:
                            max_gap_width = region_width
                            best_row = i
                            self.hole_start = region[0]
                            self.hole_end = region[1]
            
            if max_gap_width > 0:
                self.upper_surface_row = best_row
                self.hole_diameter = (self.hole_end - self.hole_start) * PIXEL_TO_UM_X
            else:
                # 最后的备选方案
                self.hole_start = self.original_image.shape[1] // 3
                self.hole_end = self.original_image.shape[1] * 2 // 3
                self.hole_diameter = (self.hole_end - self.hole_start) * PIXEL_TO_UM_X
        
        # 计算孔的中心x坐标，确保在后续代码中可用
        hole_center_x = (self.hole_start + self.hole_end) // 2
        
        # 在结果图像中标记找到的缺口
        # 使用红色椭圆标记缺口位置
        cv2.ellipse(result_img, 
                   (hole_center_x, self.upper_surface_row),
                   (int(max_gap_width / 2), 10), 
                   0, 0, 360, (0, 0, 255), 2)
        
        # 为寻找底部做准备 - 改进底部检测算法
        found_bottom = False
        
        # 计算搜索范围
        max_search_depth = int(self.original_image.shape[0] * self.params['bottom_search_range'])
        search_end_row = min(self.original_image.shape[0], self.upper_surface_row + max_search_depth)
        
        # 首先尝试使用底部线索引
        bottom_line_index = self.params['bottom_line_index']
        if bottom_line_index < len(horizontal_lines) and bottom_line_index >= 0:
            # 确保选择的底部线在顶部线之下
            if horizontal_lines[bottom_line_index] > self.upper_surface_row + 20:
                self.bottom_surface_row = horizontal_lines[bottom_line_index]
                found_bottom = True
        
        # 如果未找到底部，设置一个默认值
        if not found_bottom:
            # 默认将底部设置为顶部行的一定距离下方
            self.bottom_surface_row = min(self.original_image.shape[0] - 1, self.upper_surface_row + 100)
        
        # 如果未找到底部或索引无效，使用改进的方法继续查找
        if not found_bottom:
            # 1. 首先尝试特别关注孔洞中心区域的底部短横线
            # hole_center_x已经在上面计算过了，这里不需要重复
            hole_width = self.hole_end - self.hole_start
            search_width = max(hole_width // 2, 30)  # 孔宽度的一半或至少30像素
            
            # 在原始二值图像中在孔中心区域搜索短横线
            hole_center_min_x = max(0, hole_center_x - search_width)
            hole_center_max_x = min(self.original_image.shape[1], hole_center_x + search_width)
            
            # 创建调试图像以显示短横线检测过程
            debug_img = cv2.cvtColor(self.binary_image.copy(), cv2.COLOR_GRAY2BGR)
            
            # 在调试图像中标记孔中心搜索区域
            cv2.rectangle(debug_img, 
                        (hole_center_min_x, self.upper_surface_row + 50), 
                        (hole_center_max_x, search_end_row), 
                        (0, 255, 255), 1)
            
            # 计算孔中心区域的列投影，用于确定底部位置
            column_sum = np.sum(self.binary_image[self.upper_surface_row + 50:search_end_row, 
                                                hole_center_min_x:hole_center_max_x], axis=1)
            
            # 为列投影创建调试图像
            plt.figure(figsize=(10, 4))
            plt.title("孔中心区域垂直投影")
            plt.plot(column_sum)
            plt.savefig(os.path.join(debug_dir, "hole_center_column_projection.png"))
            plt.close()
            
            # 自适应确定短横线的阈值，避免受噪声影响
            # 计算投影的平均值和标准差
            projection_mean = np.mean(column_sum)
            projection_std = np.std(column_sum)
            
            # 检测峰值 - 峰值表示水平线（高于平均值+标准差的区域）
            potential_lines = []
            min_peak_height = projection_mean + projection_std * 1.5  # 调整这个系数可以控制灵敏度
            
            for i in range(len(column_sum)):
                if column_sum[i] > min_peak_height:
                    # 检查是否是局部最大值
                    window = 5  # 窗口大小
                    start_idx = max(0, i - window)
                    end_idx = min(len(column_sum), i + window + 1)
                    if column_sum[i] == np.max(column_sum[start_idx:end_idx]):
                        row_position = self.upper_surface_row + 50 + i
                        potential_lines.append((row_position, column_sum[i]))
            
            # 去除过近的峰值
            filtered_lines = []
            if potential_lines:
                filtered_lines.append(potential_lines[0])
                for i in range(1, len(potential_lines)):
                    # 如果与之前添加的峰值距离足够远，则添加
                    if potential_lines[i][0] - filtered_lines[-1][0] > 20:  # 至少20像素的距离
                        filtered_lines.append(potential_lines[i])
            
            # 在底部附近寻找最合适的短横线
            # 1. 首先查看二值化图像中是否有明显的短横线
            bottom_found = False
            
            # 将潜在的水平线按照从上到下排序
            filtered_lines.sort(key=lambda x: x[0])
            
            # 在调试图像中标记所有检测到的水平线位置
            for row_pos, strength in filtered_lines:
                cv2.line(debug_img, (0, row_pos), (self.original_image.shape[1], row_pos), 
                        (0, 255, 0), 1)  # 绿色标记所有潜在的水平线
            
            # 保存标记了潜在水平线的调试图像
            cv2.imwrite(os.path.join(debug_dir, "potential_horizontal_lines.jpg"), debug_img)
            
            # 特别关注孔洞底部区域，避免错误检测上方的噪声
            # 设置最小搜索深度，确保在孔洞足够深的位置寻找底部
            min_valid_depth = 100  # 至少需要100像素的深度
            
            for row_pos, strength in filtered_lines:
                # 只考虑距离上表面足够远的水平线
                if row_pos < self.upper_surface_row + min_valid_depth:
                    continue
                
                # 获取当前行在孔中心区域的像素
                line_segment = self.binary_image[row_pos, hole_center_min_x:hole_center_max_x]
                
                if len(line_segment) == 0:
                    continue
                
                # 计算这一段中白色像素的比例
                white_ratio = np.sum(line_segment == 255) / len(line_segment)
                
                # 寻找短横线特征：白色像素比例适中
                if white_ratio > self.params['short_line_min_white_ratio'] and white_ratio < self.params['short_line_max_white_ratio']:
                    # 在这一行中查找最长的连续白色区域
                    max_run_length = 0
                    current_run = 0
                    
                    for pixel in line_segment:
                        if pixel == 255:  # 白色像素
                            current_run += 1
                        else:
                            max_run_length = max(max_run_length, current_run)
                            current_run = 0
                    
                    # 检查最后一个连续区域
                    max_run_length = max(max_run_length, current_run)
                    
                    # 如果有足够长的连续白色区域，认为这是短横线
                    if max_run_length > self.params['short_line_min_length']:
                        # 在调试图像和结果图像上标记找到的短横线
                        cv2.line(debug_img, (hole_center_min_x, row_pos), 
                                (hole_center_max_x, row_pos), (0, 0, 255), 2)  # 红色
                        cv2.line(result_img, (hole_center_min_x, row_pos), 
                                (hole_center_max_x, row_pos), (0, 0, 255), 3)
                        
                        # 找到底部短横线
                        self.bottom_surface_row = row_pos
                        bottom_found = True
                        break
            
            # 如果还是没找到底部，尝试在可能的峰值中选择最下面的一个
            if not bottom_found and filtered_lines:
                # 排除上表面附近的峰值
                valid_lines = [line for line in filtered_lines 
                              if line[0] > self.upper_surface_row + min_valid_depth]
                
                if valid_lines:
                    # 找出强度最大的底部线
                    max_strength_line = max(valid_lines, key=lambda x: x[1])
                    self.bottom_surface_row = max_strength_line[0]
                    
                    # 在结果图像上标记
                    cv2.line(result_img, (hole_center_min_x, self.bottom_surface_row), 
                            (hole_center_max_x, self.bottom_surface_row), (0, 0, 255), 3)
                    bottom_found = True
            
            # 保存最终的调试图像
            if bottom_found:
                cv2.imwrite(os.path.join(debug_dir, "bottom_line_detected.jpg"), debug_img)
                found_bottom = True
            else:
                cv2.imwrite(os.path.join(debug_dir, "bottom_line_search_failed.jpg"), debug_img)
        
        # 计算深度
        self.hole_depth = (self.bottom_surface_row - self.upper_surface_row) * PIXEL_TO_UM_Y
        
        # === 新增：计算指定位置的直径（上表面下0.1mm和底部上0.1mm处） ===
        
        # 从像素转换到0.1mm的距离
        distance_01mm_pixels = int(0.1 * 1000 / PIXEL_TO_UM_Y)  # 0.1mm转换为像素
        
        # 计算上表面下0.1mm处的行位置
        upper_measure_row = self.upper_surface_row + distance_01mm_pixels
        
        # 计算底部上0.1mm处的行位置
        lower_measure_row = self.bottom_surface_row - distance_01mm_pixels
        
        # 确保行位置在有效范围内
        upper_measure_row = min(max(0, upper_measure_row), self.original_image.shape[0]-1)
        lower_measure_row = min(max(0, lower_measure_row), self.original_image.shape[0]-1)
        
        # 标记测量位置
        cv2.line(result_img, (0, upper_measure_row), (self.original_image.shape[1], upper_measure_row), (0, 128, 255), 1)
        cv2.line(result_img, (0, lower_measure_row), (self.original_image.shape[1], lower_measure_row), (0, 128, 255), 1)
        
        # 根据宽度查找这两个位置的直径
        upper_diameter = 0
        lower_diameter = 0
        
        # 检测上测量点处的直径
        upper_found, upper_left, upper_right = self.find_hole_edges_at_row(upper_measure_row)
        if upper_found:
            upper_diameter = (upper_right - upper_left) * PIXEL_TO_UM_X
            cv2.line(result_img, (upper_left, upper_measure_row), (upper_right, upper_measure_row), (255, 128, 0), 2)
            cv2.circle(result_img, (upper_left, upper_measure_row), 4, (255, 0, 0), -1)
            cv2.circle(result_img, (upper_right, upper_measure_row), 4, (255, 0, 0), -1)
        
        # 检测下测量点处的直径
        lower_found, lower_left, lower_right = self.find_hole_edges_at_row(lower_measure_row)
        if lower_found:
            lower_diameter = (lower_right - lower_left) * PIXEL_TO_UM_X
            cv2.line(result_img, (lower_left, lower_measure_row), (lower_right, lower_measure_row), (255, 128, 0), 2)
            cv2.circle(result_img, (lower_left, lower_measure_row), 4, (255, 0, 0), -1)
            cv2.circle(result_img, (lower_right, lower_measure_row), 4, (255, 0, 0), -1)
        
        # 更新测量结果
        self.upper_diameter_at_01mm = upper_diameter
        self.lower_diameter_at_01mm = lower_diameter
        
        # 计算标准直径（上下两处的平均值）
        if upper_diameter > 0 and lower_diameter > 0:
            standard_diameter = (upper_diameter + lower_diameter) / 2
        elif upper_diameter > 0:
            standard_diameter = upper_diameter
        elif lower_diameter > 0:
            standard_diameter = lower_diameter
        else:
            standard_diameter = self.hole_diameter  # 如果两个位置都无法测量，使用原始直径
        
        # 累加测量次数和历史测量值
        if standard_diameter > 0:
            self.measurement_count += 1
            if self.measurement_count <= 3:
                self.diameter_history.append(standard_diameter)
            if self.measurement_count > 3:
                # 保持最近3次测量记录
                self.diameter_history = self.diameter_history[1:] + [standard_diameter]
        
        # 计算3次测量的平均值作为最终标准直径
        if len(self.diameter_history) > 0:
            self.standard_diameter = sum(self.diameter_history) / len(self.diameter_history)
        else:
            self.standard_diameter = standard_diameter
        
        # 计算深径比
        if self.standard_diameter > 0:
            self.depth_diameter_ratio = self.hole_depth / self.standard_diameter
        else:
            self.depth_diameter_ratio = 0
        
        # 在图像上绘制测量结果
        # 突出显示上表面直线
        cv2.line(result_img, (0, self.upper_surface_row), 
                (self.original_image.shape[1], self.upper_surface_row), (0, 255, 0), 2)
        
        # 突出显示孔的边界
        cv2.line(result_img, (self.hole_start, self.upper_surface_row), 
                (self.hole_start, self.bottom_surface_row), (255, 0, 0), 2)
        cv2.line(result_img, (self.hole_end, self.upper_surface_row), 
                (self.hole_end, self.bottom_surface_row), (255, 0, 0), 2)
        
        # 突出显示底部横线
        cv2.line(result_img, (0, self.bottom_surface_row), 
                (self.original_image.shape[1], self.bottom_surface_row), (0, 255, 0), 2)
        
        # 标记测量点
        cv2.circle(result_img, (self.hole_start, self.upper_surface_row), 5, (0, 0, 255), -1)
        cv2.circle(result_img, (self.hole_end, self.upper_surface_row), 5, (0, 0, 255), -1)
        cv2.circle(result_img, (hole_center_x, self.bottom_surface_row), 5, (0, 0, 255), -1)
        
        # 绘制测量线
        cv2.line(result_img, (self.hole_start, self.upper_surface_row - 20), 
                (self.hole_end, self.upper_surface_row - 20), (0, 0, 255), 2)
        cv2.line(result_img, (self.hole_start, self.upper_surface_row - 20), 
                (self.hole_start, self.upper_surface_row - 15), (0, 0, 255), 2)
        cv2.line(result_img, (self.hole_end, self.upper_surface_row - 20), 
                (self.hole_end, self.upper_surface_row - 15), (0, 0, 255), 2)
        
        cv2.line(result_img, (hole_center_x + 20, self.upper_surface_row), 
                (hole_center_x + 20, self.bottom_surface_row), (0, 0, 255), 2)
        cv2.line(result_img, (hole_center_x + 20, self.upper_surface_row), 
                (hole_center_x + 15, self.upper_surface_row), (0, 0, 255), 2)
        cv2.line(result_img, (hole_center_x + 20, self.bottom_surface_row), 
                (hole_center_x + 15, self.bottom_surface_row), (0, 0, 255), 2)
        
        # 在图像上添加文本信息
        self.addChineseText(result_img, f"标准直径: {self.standard_diameter:.2f}μm", 
                          (10, 30), (255, 255, 255), (0, 0, 0))
        self.addChineseText(result_img, f"深径比: {self.depth_diameter_ratio:.2f}", 
                          (10, 60), (255, 255, 255), (0, 0, 0))
        
        # 更新结果图像
        self.result_image = result_img
        # 不在这里调用displayImage，因为processImage中已经统一处理
        # self.displayImage(self.result_image, self.resultImageLabel)
        
        # 更新测量结果标签
        self.diameterLabel.setText(f"直径: {self.hole_diameter:.2f} μm")
        self.depthLabel.setText(f"深度: {self.hole_depth:.2f} μm")
        self.upperDiameterLabel.setText(f"上方0.1mm处直径: {self.upper_diameter_at_01mm:.2f} μm")
        self.lowerDiameterLabel.setText(f"下方0.1mm处直径: {self.lower_diameter_at_01mm:.2f} μm")
        self.standardDiameterLabel.setText(f"标准直径: {self.standard_diameter:.2f} μm")
        self.ratioLabel.setText(f"深径比: {self.depth_diameter_ratio:.2f}")
        self.measureCountLabel.setText(f"测量次数: {min(self.measurement_count, 3)}/3")
        
        # 启用保存按钮和导出数据按钮
        self.saveResultBtn.setEnabled(True)
        self.exportDataBtn.setEnabled(True)

        # 启用设置参考深度按钮
        if self.hole_depth > 0:
            self.setDepthForOctBtn.setEnabled(True)
    
    def find_hole_edges_at_row(self, row):
        """在指定行查找孔洞的左右边缘，优先使用已知的孔洞边界（蓝色竖线）"""
        if self.binary_image is None:
            return False, 0, 0
            
        if row < 0 or row >= self.binary_image.shape[0]:
            return False, 0, 0
            
        # 计算在当前行的位置应与原始检测到的孔洞边界保持一致
        # 这样可以确保始终测量的是蓝色竖线之间的区域
        
        # 蓝色竖线通常是垂直的，所以x坐标在不同行应该相同
        left_x = self.hole_start
        right_x = self.hole_end
        
        # 但如果图像经过旋转，或者孔洞呈现一定角度，
        # 可能需要根据行的位置调整x坐标
        if hasattr(self, 'is_image_rotated') and self.is_image_rotated:
            # 对于旋转的图像，可能需要投影蓝色线到当前行
            # 这部分代码在detect_hole_in_rotated_image方法中已经考虑
            pass
        
        # 在确定的位置附近搜索实际的边缘，以提高精度
        search_range = 10  # 每边搜索10个像素的范围
        
        # 获取指定行
        line = self.binary_image[row, :]
        
        # 在左边缘附近搜索实际边界
        left_search_start = max(0, left_x - search_range)
        left_search_end = min(self.binary_image.shape[1]-1, left_x + search_range)
        
        for x in range(left_search_start, left_search_end):
            # 寻找从白到黑的过渡（边缘）
            if x < len(line)-1 and line[x] == 255 and line[x+1] == 0:
                left_x = x
                break
        
        # 在右边缘附近搜索实际边界
        right_search_start = max(0, right_x - search_range)
        right_search_end = min(self.binary_image.shape[1]-1, right_x + search_range)
        
        for x in range(right_search_start, right_search_end):
            # 寻找从黑到白的过渡（边缘）
            if x < len(line)-1 and line[x] == 0 and line[x+1] == 255:
                right_x = x
                break
        
        # 确保左边缘在右边缘之前
        if left_x >= right_x:
            # 如果搜索失败，使用原始的边界
            left_x = self.hole_start
            right_x = self.hole_end
            
        return True, left_x, right_x
    
    # 参数更新回调函数
    def updateGaussianKernel(self, value):
        if value % 2 == 0:
            value += 1  # 确保是奇数
        self.params['gaussian_kernel'] = value
        self.gaussianLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateAdaptiveBlockSize(self, value):
        if value % 2 == 0:
            value += 1  # 确保是奇数
        self.params['adaptive_block_size'] = value
        self.adaptiveBlockLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateAdaptiveC(self, value):
        self.params['adaptive_c'] = value
        self.adaptiveCLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateTopLineIndex(self, value):
        self.params['top_line_index'] = value
        if self.original_image is not None:
            self.processImage()
    
    def updateRowProjThreshold(self, value):
        self.params['row_projection_threshold'] = value
        self.rowProjLabel.setText(f"{value}%")
        if self.original_image is not None:
            self.processImage()
    
    def updateGapMinWidth(self, value):
        self.params['gap_min_width'] = value
        self.gapWidthLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateHorizontalKernelSize(self, value):
        if value % 2 == 0:
            value += 1  # 确保是奇数
        self.params['horizontal_kernel_size'] = value
        self.horizontalKernelLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateColProjThreshold(self, value):
        self.params['column_projection_threshold'] = value
        self.colProjLabel.setText(f"{value}%")
        if self.original_image is not None:
            self.processImage()
    
    def updatePeakWindow(self, value):
        if value % 2 == 0:
            value += 1  # 确保是奇数
        self.params['column_peak_window'] = value
        self.peakWindowLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateBottomContrast(self, value):
        self.params['bottom_enhance_contrast'] = value / 10.0
        self.bottomContrastLabel.setText(f"{self.params['bottom_enhance_contrast']:.1f}")
        if self.original_image is not None:
            self.processImage()
    
    def updateBottomSearchRange(self, value):
        self.params['bottom_search_range'] = value / 100.0
        self.bottomSearchLabel.setText(f"{self.params['bottom_search_range']:.2f}")
        if self.original_image is not None:
            self.processImage()
    
    def updateBottomLineIndex(self, value):
        self.params['bottom_line_index'] = value
        if self.original_image is not None:
            self.processImage()
    
    def updateShortLineMinLength(self, value):
        self.params['short_line_min_length'] = value
        self.shortLineMinLengthLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateShortLineMinWhite(self, value):
        self.params['short_line_min_white_ratio'] = value / 100.0
        self.shortLineMinWhiteLabel.setText(f"{self.params['short_line_min_white_ratio']:.2f}")
        if self.original_image is not None:
            self.processImage()
    
    def updateShortLineMaxWhite(self, value):
        self.params['short_line_max_white_ratio'] = value / 100.0
        self.shortLineMaxWhiteLabel.setText(f"{value/100:.1f}")
        if self.original_image is not None:
            self.processImage()
    
    def updateInvertBinary(self, state):
        self.params['invert_binary'] = state == Qt.Checked
        if self.original_image is not None:
            self.processImage()
            
    def updatePixelToUmX(self, value):
        """更新X方向（直径）像素转换比例"""
        self.params['pixel_to_um_x'] = value
        global PIXEL_TO_UM_X, PIXEL_TO_UM
        PIXEL_TO_UM_X = value
        PIXEL_TO_UM = value  # 保持兼容
        if self.original_image is not None:
            self.processImage()
        self.updateRatioLabel()
    
    def updatePixelToUmY(self, value):
        """更新Y方向（深度）像素转换比例"""
        self.params['pixel_to_um_y'] = value
        global PIXEL_TO_UM_Y
        PIXEL_TO_UM_Y = value
        if self.original_image is not None:
            self.processImage()
        self.updateRatioLabel()
    
    def updateRatioLabel(self):
        if hasattr(self, 'standard_diameter') and self.standard_diameter > 0 and hasattr(self, 'hole_depth') and self.hole_depth > 0:
            ratio = self.hole_depth / self.standard_diameter
            self.ratioLabel.setText(f"深径比: {ratio:.2f}")
        else:
            self.ratioLabel.setText("深径比: --")

    @pyqtSlot()
    def saveAllResults(self):
        """保存所有已处理图像的结果到输出文件夹"""
        if not self.image_files:
            QMessageBox.warning(self, "无法保存", "未加载任何图像")
            return
            
        # 备份当前图像索引
        current_idx = self.current_image_index
        
        # 保存当前参数设置
        current_params = self.params.copy()
        
        # 进度对话框
        progress = QMessageBox()
        progress.setIcon(QMessageBox.Information)
        progress.setText("正在处理并保存所有图像...")
        progress.setStandardButtons(QMessageBox.NoButton)
        progress.show()
        QApplication.processEvents()
        
        success_count = 0
        error_count = 0
        
        try:
            # 为每张图像创建结果
            for i, image_path in enumerate(self.image_files):
                # 加载图像
                self.current_image_index = i
                self.loadImageAtCurrentIndex()
                QApplication.processEvents()
                
                # 获取文件名
                base_filename = os.path.basename(image_path)
                filename_no_ext = os.path.splitext(base_filename)[0]
                save_path = os.path.join(output_dir, f"{filename_no_ext}_result.png")
                
                # 保存结果
                if self.result_image is not None and self.saveMeasurementResult(save_path):
                    success_count += 1
                else:
                    error_count += 1
        
        except Exception as e:
            QMessageBox.critical(self, "批量处理错误", f"处理过程中出现错误: {str(e)}")
        
        finally:
            # 恢复到原始图像
            self.current_image_index = current_idx
            self.loadImageAtCurrentIndex()
            
            progress.close()
            
            # 显示结果
            QMessageBox.information(self, "批量处理完成", 
                                  f"成功处理并保存: {success_count}个文件\n"
                                  f"处理失败: {error_count}个文件\n"
                                  f"结果保存在: {os.path.abspath(output_dir)}")

    @pyqtSlot()
    def exportMeasurementData(self):
        """导出测量数据到CSV文件"""
        try:
            options = QFileDialog.Options()
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output_files"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            default_filename = os.path.join(output_dir, f"hole_measurements_{current_time}.xlsx")
            filePath, _ = QFileDialog.getSaveFileName(
                self, "导出测量数据", default_filename,
                "Excel文件 (*.xlsx);;CSV文件 (*.csv);;所有文件 (*)", options=options)

            if filePath:
                if not hasattr(self, 'diameter_history') or not self.diameter_history:
                    QMessageBox.warning(self, "无数据", "没有可导出的测量数据。")
                    return

                data = {
                    "文件名": [], "上表面 (像素)": [], "孔洞起点 (像素)": [], "孔洞终点 (像素)": [],
                    "直径 (像素)": [], "直径 (μm)": [], "底面 (像素)": [], "深度 (像素)": [],
                    "深度 (μm)": [], "测量时间": []
                }
                
                for record in self.diameter_history:
                    if isinstance(record, dict):
                        data["文件名"].append(record.get("filename", "N/A"))
                        data["上表面 (像素)"].append(record.get("upper_surface_row", "N/A"))
                        data["孔洞起点 (像素)"].append(record.get("hole_start", "N/A"))
                        data["孔洞终点 (像素)"].append(record.get("hole_end", "N/A"))
                        diameter_px = record.get("hole_end", 0) - record.get("hole_start", 0)
                        depth_px = record.get("bottom_surface_row", 0) - record.get("upper_surface_row", 0)
                        data["直径 (像素)"].append(diameter_px)
                        data["直径 (μm)"].append(diameter_px * PIXEL_TO_UM_X)
                        data["底面 (像素)"].append(record.get("bottom_surface_row", "N/A"))
                        data["深度 (像素)"].append(depth_px)
                        data["深度 (μm)"].append(depth_px * PIXEL_TO_UM_Y)
                        data["测量时间"].append(record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                df = pd.DataFrame(data)
                if filePath.endswith('.xlsx'):
                    df.to_excel(filePath, index=False, engine='openpyxl')
                else:
                    df.to_csv(filePath, index=False, encoding='utf-8-sig')
                
                QMessageBox.information(self, "导出成功", f"测量数据已导出至: {filePath}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出过程中出现错误: {str(e)}")
            print(f"导出错误详情: {str(e)}")
            import traceback
            traceback.print_exc()

    def addChineseText(self, img, text, position, textColor=(255, 255, 255), bgColor=None, fontSize=20):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/msyh.ttc'
        
        font = None
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, fontSize)
            except Exception as e:
                print(f"无法加载字体 {font_path}: {e}")
        
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception as e:
                print(f"无法加载默认字体: {e}")
                return img

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        
        if bgColor:
            x, y = position
            draw.rectangle([(x, y), (x + text_width, y + text_height)], fill=bgColor)
        
        draw.text(position, text, font=font, fill=textColor)
        return np.array(pil_img)

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if self.manual_measure_active:
            # 手动测量模式下，按D键切换到测量直径，按H键切换到测量深度
            if event.key() == Qt.Key_D:
                self.measure_mode = "diameter"
                self.update()
                print("切换到测量直径模式")
            elif event.key() == Qt.Key_H:
                self.measure_mode = "depth"
                self.update()
                print("切换到测量深度模式")
                
        super().keyPressEvent(event)

    def processNoiseAndMerge(self):
        """将一张图片添加随机噪声生成三张图片，合并后进行测量"""
        if self.original_image is None:
            QMessageBox.warning(self, "无法处理", "请先加载一张图像")
            return
            
        # 创建进度对话框
        progress = QProgressDialog("正在处理图像...", "取消", 0, 5, self)
        progress.setWindowTitle("生成噪声图像")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            # 创建临时文件夹存储生成的图像
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成带噪声的图像
            noise_images = []
            for i in range(3):
                progress.setValue(i)
                if progress.wasCanceled():
                    return
                    
                # 添加不同程度的随机噪声
                noise_type = i  # 0=高斯噪声, 1=椒盐噪声, 2=泊松噪声
                noise_img = self.addNoise(self.original_image, noise_type, i)
                noise_images.append(noise_img)
                
                # 保存噪声图像
                temp_path = os.path.join(temp_dir, f"noise_{i}.jpg")
                cv2.imwrite(temp_path, noise_img)
            
            progress.setValue(3)
            if progress.wasCanceled():
                return
            
            # 合并三张图像
            merged_image = self.mergeImages(noise_images)
            
            # 保存合并后的图像
            merged_path = os.path.join(output_dir, "merged_noise_image.jpg")
            cv2.imwrite(merged_path, merged_image)
            
            progress.setValue(4)
            if progress.wasCanceled():
                return
                
            # 加载并处理合并后的图像
            self.original_image = merged_image
            self.displayImage(self.original_image, self.originalImageLabel)
            
            # 直接调用多孔洞分析而不是常规处理
            self.analyzeMergedImage()
            
            progress.setValue(5)
            QMessageBox.information(self, "处理完成", 
                                  f"已生成3张带噪声的图像并合并\n"
                                  f"合并图像已保存至: {merged_path}\n"
                                  f"已对合并图像进行多孔洞测量处理")
            
        except Exception as e:
            QMessageBox.critical(self, "处理错误", f"处理过程中出现错误: {str(e)}")
        
        finally:
            progress.close()
    
    def addNoise(self, image, noise_type, seed=0):
        """添加不同类型的噪声"""
        np.random.seed(seed)  # 设置随机种子以获得可重现的结果
        noisy_image = image.copy().astype(np.float32)
        
        if noise_type == 0:  # 高斯噪声
            mean = 0
            sigma = 25.0 + seed * 5
            gauss = np.random.normal(mean, sigma, image.shape)
            noisy_image = noisy_image + gauss
        
        elif noise_type == 1:  # 椒盐噪声
            s_vs_p = 0.5
            amount = 0.01 + seed * 0.005
            # 添加盐噪声
            salt_mask = np.random.random(image.shape) < (amount * s_vs_p)
            noisy_image[salt_mask] = 255
            # 添加椒噪声
            pepper_mask = np.random.random(image.shape) < (amount * (1.0 - s_vs_p))
            noisy_image[pepper_mask] = 0
        
        elif noise_type == 2:  # 泊松噪声
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            factor = 5.0 - seed * 0.5
            noisy_image = np.random.poisson(noisy_image / factor) * factor
        
        # 确保值在[0, 255]范围内
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def mergeImages(self, images):
        """将三张图像合并为一张"""
        if len(images) != 3:
            raise ValueError("需要3张图像进行合并")
        
        # 确定每个图像的大小
        h, w = images[0].shape
        
        # 创建1x3布局的合并图像
        merged = np.zeros((h, w*3), dtype=np.uint8)
        
        # 添加标题文本
        noise_types = ["高斯噪声", "椒盐噪声", "泊松噪声"]
        
        # 填充图像
        for j in range(3):
            img_with_text = images[j].copy()
            
            # 添加文本标签
            cv2.putText(img_with_text, 
                       noise_types[j], 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       255, 
                       2)
            
            merged[0:h, j*w:(j+1)*w] = img_with_text
        
        return merged

    def analyzeMergedImage(self):
        """分析合并图像中的三个孔洞 - 完善版：先准确分割，再单独处理，最后合并结果"""
        if self.original_image is None:
            QMessageBox.warning(self, "无法处理", "请先加载一张合并图像")
            return
        
        # 检测是否是合并图像
        height, width = self.original_image.shape
        if width % 3 != 0:
            QMessageBox.warning(self, "图像格式错误", "当前图像不是标准的1x3合并图像格式")
            return
        
        # 创建进度对话框
        progress = QProgressDialog("正在分析合并图像...", "取消", 0, 4, self)
        progress.setWindowTitle("孔洞分析中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            # 准备结果图像和数据
            merged_result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
            
            # 准确分割图像为3个子图像
            sub_width = width // 3
            sub_images = []
            
            # 保存分割图像用于调试
            debug_dir = 'debug'
            os.makedirs(debug_dir, exist_ok=True)
            
            for j in range(3):  # 3列
                x_start = j * sub_width
                sub_img = self.original_image[:, x_start:x_start+sub_width].copy()
                sub_images.append(sub_img)
                
                # 保存分割图像(可选)
                cv2.imwrite(os.path.join(debug_dir, f"sub_image_{j}.jpg"), sub_img)
            
            progress.setValue(1)
            if progress.wasCanceled():
                return
            
            # 噪声类型标签
            noise_types = ["高斯噪声", "椒盐噪声", "泊松噪声"]
            
            # 准备存储测量结果
            hole_measurements = []
            
            # 设置参考值 - 基于第一张图像的测量结果
            reference_params = self.params.copy()  # 备份当前参数
            
            # 先处理第一张图像获取参考值
            first_diameter, first_depth, first_hole_data = self.processSingleImage(
                sub_images[0], 
                ref_diameter=200.0,
                ref_depth=1000.0,
                is_noisy=False
            )
            
            # 对每个子图像单独处理
            for idx, sub_img in enumerate(sub_images):
                if progress.wasCanceled():
                    return
                
                # 更新进度
                progress.setValue(1 + idx)
                progress.setLabelText(f"正在分析第 {idx+1}/3 个孔洞...")
                
                # 计算当前子图像的位置偏移
                x_offset = idx * sub_width
                
                # 使用第一张图像作为参考
                ref_diameter = first_diameter
                ref_depth = first_depth
                
                # 对当前子图像进行处理
                diameter, depth, hole_data = self.processSingleImage(
                    sub_img, 
                    ref_diameter=ref_diameter,
                    ref_depth=ref_depth,
                    is_noisy=(idx > 0)  # 第一张图像噪声较少
                )
                
                # 处理测量结果并绘制到合并图像上
                if hole_data:
                    # 调整坐标以适应合并图像
                    upper_row = hole_data['upper_surface_row']
                    bottom_row = hole_data['bottom_surface_row']
                    start_col = hole_data['hole_start'] + x_offset
                    end_col = hole_data['hole_end'] + x_offset
                    
                    # 绘制上表面直线
                    cv2.line(merged_result_img, 
                            (x_offset, upper_row), 
                            (x_offset + sub_width, upper_row), 
                            (0, 255, 0), 2)
                    
                    # 绘制孔的侧边界
                    cv2.line(merged_result_img, 
                            (start_col, upper_row), 
                            (start_col, bottom_row), 
                            (255, 0, 0), 2)
                    cv2.line(merged_result_img, 
                            (end_col, upper_row), 
                            (end_col, bottom_row), 
                            (255, 0, 0), 2)
                    
                    # 绘制底表面直线
                    cv2.line(merged_result_img, 
                            (x_offset, bottom_row), 
                            (x_offset + sub_width, bottom_row), 
                            (0, 255, 0), 2)
                    
                    # 标记测量点
                    cv2.circle(merged_result_img, (start_col, upper_row), 5, (0, 0, 255), -1)
                    cv2.circle(merged_result_img, (end_col, upper_row), 5, (0, 0, 255), -1)
                    
                    hole_center_x = (start_col + end_col) // 2
                    cv2.circle(merged_result_img, (hole_center_x, bottom_row), 5, (0, 0, 255), -1)
                    
                    # 添加文本标注
                    self.addChineseText(
                        merged_result_img,
                        f"{idx+1}: {noise_types[idx]}",
                        (x_offset + 10, 30),
                        textColor=(255, 255, 255),
                        bgColor=(0, 0, 0)
                    )
                    
                    self.addChineseText(
                        merged_result_img,
                        f"直径: {diameter:.1f}μm",
                        (x_offset + 10, 60),
                        textColor=(255, 255, 255),
                        bgColor=(0, 0, 0)
                    )
                    
                    self.addChineseText(
                        merged_result_img,
                        f"深度: {depth:.1f}μm",
                        (x_offset + 10, 90),
                        textColor=(255, 255, 255),
                        bgColor=(0, 0, 0)
                    )
                
                # 保存测量数据
                hole_measurements.append({
                    "编号": idx + 1,
                    "噪声类型": noise_types[idx],
                    "直径(μm)": round(diameter, 1),
                    "深度(μm)": round(depth, 1)
                })
            
            # 保存识别结果图像
            detection_result_path = os.path.join(output_dir, "hole_detection_result.jpg")
            cv2.imwrite(detection_result_path, merged_result_img)
            
            # 显示识别结果图像
            self.result_image = merged_result_img
            self.displayImage(self.result_image, self.resultImageLabel, "合并图像分析结果")
            
            progress.setValue(4)
            
            # 创建并显示测量结果表格
            self.showMeasurementTable(hole_measurements)
            
        except Exception as e:
            QMessageBox.critical(self, "处理错误", f"分析过程中出现错误: {str(e)}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            progress.close()
            
    def processSingleImage(self, image, ref_diameter=200.0, ref_depth=1000.0, is_noisy=False):
        """处理单个子图像，返回测量结果"""
        # 备份原始图像和当前参数
        original_image_backup = self.original_image
        binary_image_backup = self.binary_image
        
        try:
            # 设置当前图像
            self.original_image = image
            
            # 保存原始子图像用于调试
            debug_dir = 'debug'
            cv2.imwrite(os.path.join(debug_dir, f"process_original_{int(time.time())}.jpg"), image)
            
            # 增强图像预处理，针对噪声图像
            # 1. 高斯滤波
            gaussian_kernel_size = max(3, self.params['gaussian_kernel'])
            if gaussian_kernel_size % 2 == 0:
                gaussian_kernel_size += 1  # 确保是奇数
            blurred = cv2.GaussianBlur(image, (gaussian_kernel_size, gaussian_kernel_size), 0)
            
            # 2. 对于噪声较大的图像，应用额外的滤波
            if is_noisy:
                # 中值滤波去除椒盐噪声
                blurred = cv2.medianBlur(blurred, 5)
                
                # 双边滤波保留边缘
                blurred = cv2.bilateralFilter(blurred, 9, 75, 75)
                
                # 非局部均值去噪处理高斯噪声
                blurred = cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
            
            # 保存预处理图像用于调试
            cv2.imwrite(os.path.join(debug_dir, f"process_blurred_{int(time.time())}.jpg"), blurred)
            
            # 二值化处理
            # 1. 自适应二值化
            adaptive_block_size = self.params['adaptive_block_size']
            if adaptive_block_size % 2 == 0:
                adaptive_block_size += 1  # 确保是奇数
            adaptive_c = self.params['adaptive_c']
            binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
            
            # 2. OTSU二值化
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. 全局二值化
            _, binary_global = cv2.threshold(blurred, self.params['binary_threshold'], 255, cv2.THRESH_BINARY)
            
            # 4. 合并三种二值化结果
            binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
            binary_combined = cv2.bitwise_and(binary_combined, binary_global)
            
            # 形态学操作优化二值图像
            kernel = np.ones((3, 3), np.uint8)
            binary_opened = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel)
            binary_final = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel)
            
            # 如果需要反转二值图像
            if self.params['invert_binary']:
                binary_final = 255 - binary_final
            
            self.binary_image = binary_final
            
            # 保存二值化图像用于调试
            cv2.imwrite(os.path.join(debug_dir, f"process_binary_{int(time.time())}.jpg"), binary_final)
            
            # 检测孔的尺寸 - 以下采用与单图像处理相同的逻辑
            
            # 计算行投影找到水平线
            row_projection = np.sum(self.binary_image, axis=1)
            row_projection_smooth = np.convolve(row_projection, np.ones(5)/5, mode='same')
            
            # 找到显著水平线
            threshold = np.max(row_projection_smooth) * (self.params['row_projection_threshold'] / 100.0)
            significant_rows = np.where(row_projection_smooth > threshold)[0]
            
            # 对找到的行分组
            horizontal_lines = []
            current_group = []
            
            for i in range(len(significant_rows)):
                if i == 0 or significant_rows[i] - significant_rows[i-1] <= 5:
                    current_group.append(significant_rows[i])
                else:
                    if current_group:
                        horizontal_lines.append(int(np.mean(current_group)))
                        current_group = [significant_rows[i]]
            
            # 添加最后一组
            if current_group:
                horizontal_lines.append(int(np.mean(current_group)))
            
            # 按从上到下排序
            horizontal_lines.sort()
            
            # 在调试图像上标记找到的水平线
            debug_img = cv2.cvtColor(self.binary_image.copy(), cv2.COLOR_GRAY2BGR)
            for i, line_pos in enumerate(horizontal_lines):
                cv2.line(debug_img, (0, line_pos), (image.shape[1], line_pos), (0, 255, 0), 1)
            
            # 确定上表面位置 - 使用多种方法结合
            upper_surface_row = 0
            
            # 1. 使用水平线检测结果
            if len(horizontal_lines) > 0:
                upper_surface_row = horizontal_lines[0]  # 初步选择最上方的线
            
            # 2. 查找图像上部的亮区域
            upper_region = image[:image.shape[0]//3, :]
            if upper_region.size > 0:
                # 找到最亮的行
                row_brightness = np.sum(upper_region, axis=1)
                if len(row_brightness) > 0:
                    brightest_row = np.argmax(row_brightness)
                    # 如果最亮行位置合理，使用它
                    if brightest_row > 10 and brightest_row < image.shape[0]//4:
                        upper_surface_row = brightest_row
            
            # 在上表面附近寻找孔洞边缘（缺口）
            search_range = 20
            max_gap_width = 0
            hole_start = 0
            hole_end = 0
            best_row = upper_surface_row
            
            # 在选定行附近搜索最明显的缺口
            for i in range(max(0, upper_surface_row - search_range), 
                          min(image.shape[0], upper_surface_row + search_range)):
                # 获取当前行
                line = self.binary_image[i, :]
                
                # 寻找转换点
                transitions = np.where(np.diff(line) != 0)[0]
                if len(transitions) >= 2:
                    # 寻找白-黑-白模式
                    for j in range(0, len(transitions) - 1):
                        # 确认白到黑转换
                        if j < len(transitions) - 1 and line[transitions[j]] == 255 and transitions[j] + 1 < len(line) and line[transitions[j] + 1] == 0:
                            # 寻找黑到白转换
                            for k in range(j + 1, len(transitions)):
                                if line[transitions[k]] == 0 and transitions[k] + 1 < len(line) and line[transitions[k] + 1] == 255:
                                    # 计算缺口宽度
                                    gap_width = transitions[k] - transitions[j]
                                    
                                    # 检查是否符合最小宽度且是最宽的
                                    if gap_width > self.params['gap_min_width'] and gap_width > max_gap_width:
                                        max_gap_width = gap_width
                                        best_row = i
                                        hole_start = transitions[j]
                                        hole_end = transitions[k]
                                    break
            
            # 如果找到了缺口
            if max_gap_width > 0:
                upper_surface_row = best_row
                hole_diameter = (hole_end - hole_start) * PIXEL_TO_UM_X
                
                # 在调试图像上标记缺口
                cv2.line(debug_img, (hole_start, upper_surface_row), 
                         (hole_end, upper_surface_row), (0, 0, 255), 2)
            else:
                # 备选方法：使用图像分析
                # 查找图像中间区域的亮度分布
                center_row = upper_surface_row + 20  # 稍微往下一点
                if center_row < image.shape[0]:
                    center_line = image[center_row, :]
                    
                    # 平滑处理
                    center_line_smooth = np.convolve(center_line, np.ones(7)/7, mode='same')
                    
                    # 查找亮度变化
                    # 计算平均值作为阈值
                    avg_brightness = np.mean(center_line_smooth)
                    
                    # 查找低于平均亮度的连续区域
                    in_dark = False
                    dark_regions = []
                    dark_start = 0
                    
                    for x in range(len(center_line_smooth)):
                        if center_line_smooth[x] < avg_brightness - 20 and not in_dark:
                            in_dark = True
                            dark_start = x
                        elif center_line_smooth[x] >= avg_brightness - 20 and in_dark:
                            in_dark = False
                            dark_regions.append((dark_start, x))
                    
                    # 如果还在暗区域，添加最后一个区域
                    if in_dark:
                        dark_regions.append((dark_start, len(center_line_smooth)-1))
                    
                    # 找出最宽的暗区域
                    if dark_regions:
                        widest_region = max(dark_regions, key=lambda x: x[1]-x[0])
                        
                        # 如果宽度合理，使用它作为孔洞
                        if widest_region[1] - widest_region[0] > 30:
                            hole_start = widest_region[0]
                            hole_end = widest_region[1]
                            hole_diameter = (hole_end - hole_start) * PIXEL_TO_UM_X
                            
                            # 标记在调试图像上
                            cv2.line(debug_img, (hole_start, center_row), 
                                     (hole_end, center_row), (255, 0, 255), 2)
                        else:
                            # 使用参考直径
                            width_center = image.shape[1] // 2
                            hole_start = width_center - int(ref_diameter / PIXEL_TO_UM_X / 2)
                            hole_end = width_center + int(ref_diameter / PIXEL_TO_UM_X / 2)
                            hole_diameter = ref_diameter
                else:
                    # 使用参考直径
                    width_center = image.shape[1] // 2
                    hole_start = width_center - int(ref_diameter / PIXEL_TO_UM_X / 2)
                    hole_end = width_center + int(ref_diameter / PIXEL_TO_UM_X / 2)
                    hole_diameter = ref_diameter
            
            # ===== 改进底部检测算法 =====
            # 不再依赖固定的参考深度，扩大搜索范围到整个图像下半部分
            
            # 保存所有检测到的可能底部位置
            potential_bottoms = []
            
            # 1. 首先在整个图像下半部分搜索水平线
            bottom_region_start = upper_surface_row + 100  # 从上表面向下至少100像素开始搜索
            bottom_region_end = image.shape[0] - 5  # 搜索到接近图像底部
            
            # 获取孔洞中心区域
            hole_center = (hole_start + hole_end) // 2
            search_width = max((hole_end - hole_start) // 2, 30)
            center_min_x = max(0, hole_center - search_width)
            center_max_x = min(image.shape[1], hole_center + search_width)
            
            # 查找图像下半部分的所有水平线 - 使用行投影方法
            if bottom_region_start < bottom_region_end:
                # 针对全图下半部分
                lower_half_proj = row_projection_smooth[bottom_region_start:bottom_region_end]
                
                # 平滑处理
                if len(lower_half_proj) > 5:
                    lower_half_smooth = np.convolve(lower_half_proj, np.ones(5)/5, mode='same')
                    
                    # 计算局部峰值 - 明显高于周围区域的点
                    for i in range(2, len(lower_half_smooth)-2):
                        # 如果当前点是局部最大值，并且值足够大
                        if (lower_half_smooth[i] > lower_half_smooth[i-1] and 
                            lower_half_smooth[i] > lower_half_smooth[i-2] and
                            lower_half_smooth[i] > lower_half_smooth[i+1] and
                            lower_half_smooth[i] > lower_half_smooth[i+2] and
                            lower_half_smooth[i] > np.mean(lower_half_smooth) * 1.2):
                            
                            # 计算实际行位置
                            row_pos = bottom_region_start + i
                            strength = lower_half_smooth[i]
                            potential_bottoms.append((row_pos, strength, 'projection_peak'))
                            
                            # 在调试图像中标记
                            cv2.line(debug_img, (0, row_pos), (image.shape[1], row_pos), 
                                     (255, 255, 0), 1)  # 黄色
            
            # 2. 特别查找孔洞中心列的水平亮线
            center_column = image[:, hole_center]
            if len(center_column) > 0:
                # 平滑处理
                center_col_smooth = np.convolve(center_column, np.ones(5)/5, mode='same')
                
                # 只关注上表面以下区域
                col_lower_half = center_col_smooth[upper_surface_row:]
                
                # 查找亮度变化 - 暗区域后的亮区域可能是底部
                if len(col_lower_half) > 10:
                    # 计算梯度
                    gradient = np.gradient(col_lower_half)
                    
                    # 查找正梯度（从暗到亮的转变）
                    for i in range(10, len(gradient)-10):
                        # 大于0表示亮度增加
                        if gradient[i] > 5 and gradient[i+1] > 5:  # 连续正梯度
                            row_pos = upper_surface_row + i
                            strength = col_lower_half[i]
                            potential_bottoms.append((row_pos, strength, 'column_gradient'))
                            
                            # 在调试图像中标记
                            cv2.line(debug_img, (hole_center-20, row_pos), (hole_center+20, row_pos), 
                                     (0, 255, 255), 1)  # 青色
            
            # 3. 针对性搜索孔洞区域可能的短横线
            # 提取孔洞区域
            hole_region = image[upper_surface_row:, center_min_x:center_max_x]
            
            if hole_region.size > 0:
                # 二值化处理，增强短横线的可见性
                _, hole_binary = cv2.threshold(hole_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 水平结构元素，用于检测水平线
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
                horizontal_lines_img = cv2.morphologyEx(hole_binary, cv2.MORPH_OPEN, horizontal_kernel)
                
                # 保存处理结果用于调试
                cv2.imwrite(os.path.join(debug_dir, f"horizontal_lines_{int(time.time())}.jpg"), horizontal_lines_img)
                
                # 查找短横线位置
                horizontal_proj = np.sum(horizontal_lines_img, axis=1)
                
                # 平滑处理
                horizontal_proj_smooth = np.convolve(horizontal_proj, np.ones(3)/3, mode='same')
                
                # 在距离上表面至少100像素处查找短横线
                min_search_row = 100
                
                for i in range(min_search_row, len(horizontal_proj_smooth)):
                    # 如果当前行水平投影值较大，可能是短横线
                    if (horizontal_proj_smooth[i] > np.mean(horizontal_proj_smooth) * 1.5 and
                        horizontal_proj_smooth[i] > 500):  # 阈值可调整
                        
                        # 计算实际行位置
                        row_pos = upper_surface_row + i
                        strength = horizontal_proj_smooth[i]
                        potential_bottoms.append((row_pos, strength * 2, 'horizontal_line'))  # 增加权重
                        
                        # 在调试图像中标记
                        cv2.line(debug_img, (center_min_x, row_pos), (center_max_x, row_pos), 
                                 (255, 0, 255), 2)  # 紫色
            
            # 4. 选择最可能的底部位置
            bottom_surface_row = 0
            bottom_found = False
            
            if potential_bottoms:
                # 按照行位置排序（从上到下）
                potential_bottoms.sort(key=lambda x: x[0])
                
                # 筛选出可能的底部位置 - 需要在合理深度范围内
                min_depth = 50  # 最小深度（像素）
                
                valid_bottoms = [b for b in potential_bottoms 
                               if b[0] > upper_surface_row + min_depth]
                
                if valid_bottoms:
                    # 查找所有类型为'horizontal_line'的候选，这些是最可能的底部
                    horizontal_line_bottoms = [b for b in valid_bottoms if b[2] == 'horizontal_line']
                    
                    if horizontal_line_bottoms:
                        # 选择最低的短横线位置
                        bottom_candidate = max(horizontal_line_bottoms, key=lambda x: x[0])
                        bottom_surface_row = bottom_candidate[0]
                        bottom_found = True
                        
                        # 在调试图像中标记找到的底部
                        cv2.line(debug_img, (0, bottom_surface_row), (image.shape[1], bottom_surface_row), 
                                 (0, 0, 255), 3)  # 红色粗线
                    else:
                        # 如果没有检测到短横线，则使用其他方法找到的最强位置
                        bottom_candidate = max(valid_bottoms, key=lambda x: x[1])
                        bottom_surface_row = bottom_candidate[0]
                        bottom_found = True
                        
                        # 在调试图像中标记找到的底部
                        cv2.line(debug_img, (0, bottom_surface_row), (image.shape[1], bottom_surface_row), 
                                 (0, 255, 0), 2)  # 绿色
            
            # 如果没找到有效的底部位置，使用参考深度计算
            if not bottom_found:
                # 使用参考深度，但增加安全余量
                expected_bottom_row = int(upper_surface_row + ref_depth / PIXEL_TO_UM_X * 1.2)  # 增加20%余量
                bottom_surface_row = min(expected_bottom_row, image.shape[0] - 10)  # 确保在图像范围内
            
            # 计算深度
            hole_depth = (bottom_surface_row - upper_surface_row) * PIXEL_TO_UM_Y
            
            # 基于参考值调整异常结果 - 放宽标准
            if abs(hole_diameter - ref_diameter) > ref_diameter * 0.7:  # 放宽直径差异容忍度
                hole_diameter = ref_diameter
            
            # 保存最终调试图像
            cv2.imwrite(os.path.join(debug_dir, f"process_debug_{int(time.time())}.jpg"), debug_img)
            
            # 返回结果
            hole_data = {
                'upper_surface_row': upper_surface_row,
                'bottom_surface_row': bottom_surface_row,
                'hole_start': hole_start,
                'hole_end': hole_end,
                'diameter': hole_diameter,
                'depth': hole_depth
            }
            
            return hole_diameter, hole_depth, hole_data
            
        finally:
            # 恢复原始图像
            self.original_image = original_image_backup
            self.binary_image = binary_image_backup
    
    def cropSelectionFinished(self):
        """完成裁剪区域选择"""
        # 重置裁剪模式
        self.is_cropping = False
        
        # 更新菜单项文本
        if hasattr(self, 'cropAction'):
            self.cropAction.setText("裁剪图像")
        
        # 获取选择矩形
        if not self.originalImageLabel.crop_start or not self.originalImageLabel.crop_end:
            QMessageBox.warning(self, "裁剪", "未选择有效区域")
            return
            
        # 获取裁剪区域
        crop_start = self.originalImageLabel.crop_start
        crop_end = self.originalImageLabel.crop_end
        
        if crop_start and crop_end:
            try:
                print(f"开始执行裁剪操作...")
                print(f"原始裁剪坐标: start=({crop_start.x()}, {crop_start.y()}), end=({crop_end.x()}, {crop_end.y()})")
                
                # 确保裁剪区域有效
                x_min = min(crop_start.x(), crop_end.x())
                y_min = min(crop_start.y(), crop_end.y())
                x_max = max(crop_start.x(), crop_end.x())
                y_max = max(crop_start.y(), crop_end.y())
                
                print(f"裁剪区域坐标: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                
                # 确保有最小尺寸
                if x_max - x_min < 10 or y_max - y_min < 10:
                    QMessageBox.warning(self, "裁剪失败", "裁剪区域太小，请重新选择")
                    print("裁剪失败：区域太小")
                    # 保持裁剪模式，让用户重新选择
                    return
                
                # 确保裁剪区域在图像范围内
                height, width = self.original_image.shape
                print(f"原始图像尺寸: {width}x{height}")
                
                # 强制将坐标限制在图像范围内
                x_min = max(0, min(x_min, width-1))
                y_min = max(0, min(y_min, height-1))
                x_max = max(0, min(x_max, width-1))
                y_max = max(0, min(y_max, height-1))
                
                print(f"调整后裁剪坐标: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                
                # 再次检查裁剪区域是否有效
                if x_max <= x_min or y_max <= y_min or x_max - x_min < 5 or y_max - y_min < 5:
                    QMessageBox.warning(self, "裁剪失败", "调整后的裁剪区域无效或过小")
                    print("裁剪失败：调整后区域无效或过小")
                    # 保持裁剪模式，让用户重新选择
                    return
                
                # 执行裁剪
                try:
                    # 保存原始图像备份
                    if not hasattr(self, 'original_image_before_crop') or self.original_image_before_crop is None:
                        self.original_image_before_crop = self.original_image.copy()
                    
                    # 确保裁剪区域正确，使用整数坐标，不偏移，直接获取子区域
                    print(f"开始裁剪子区域: y={y_min}:{y_max+1}, x={x_min}:{x_max+1}")
                    cropped_image = self.original_image[y_min:y_max+1, x_min:x_max+1]
                    
                    print(f"裁剪完成，结果尺寸: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
                    
                    # 检查裁剪结果是否有效
                    if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                        QMessageBox.warning(self, "裁剪失败", "裁剪区域无效，请重新选择")
                        print("裁剪失败：裁剪结果为空")
                        # 保持裁剪模式，让用户重新选择
                        return
                    
                    # 更新图像
                    self.original_image = cropped_image.copy()  # 使用copy()避免引用问题
                    print(f"更新后图像尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
                    self.displayImage(self.original_image, self.originalImageLabel)
                    
                    # 如果有其他图像处理结果，清空它们
                    self.binary_image = None
                    self.result_image = None
                    self.binaryImageLabel.clear()
                    self.resultImageLabel.clear()
                    
                    # 启用恢复原图按钮
                    if hasattr(self, 'restoreAction'):
                        self.restoreAction.setEnabled(True)
                    
                    # 关闭裁剪模式
                    self.is_cropping = False
                    self.cropAction.setText("裁剪图像")
                    self.originalImageLabel.setCropActive(False)
                    
                    # 恢复按钮状态 - 我们已经移除了processBtn，所以不需要启用它
                    # self.processBtn.setEnabled(True)
                    if hasattr(self, 'saveAllBtn') and len(self.image_files) > 1:
                        self.saveAllBtn.setEnabled(True)
                    
                    # 处理新图像
                    self.processImage()
                    
                    # 显示成功信息
                    print(f"裁剪成功，新图像尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
                    QMessageBox.information(self, "裁剪成功", f"图像已裁剪至尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
                    
                except Exception as e:
                    QMessageBox.warning(self, "裁剪失败", f"裁剪处理过程中出错: {str(e)}")
                    print(f"裁剪错误详情: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 退出裁剪模式
                    self.toggleCropMode()
            except Exception as e:
                QMessageBox.warning(self, "裁剪失败", f"裁剪处理过程中出错: {str(e)}")
                print(f"裁剪错误详情: {str(e)}")
                import traceback
                traceback.print_exc()
                # 退出裁剪模式
                self.toggleCropMode()
        else:
            QMessageBox.warning(self, "裁剪失败", "未选择有效的裁剪区域")
            print("裁剪失败：未选择有效区域")
            # 不退出裁剪模式，让用户可以重新选择

    def estimate_hole_width_at_point(self, center_x, center_y, direction_x, direction_y, expected_width):
        """
        在指定点估计孔洞宽度，沿指定方向搜索边缘
        返回：宽度像素值，左边缘点坐标，右边缘点坐标
        """
        # 使用已知的孔洞直径作为搜索范围的参考
        search_distance = int(expected_width / 2 * 1.2)  # 搜索距离略大于预期宽度的一半
        
        # 左右两边的边缘点
        left_edge = None
        right_edge = None
        
        # 沿负方向搜索左边缘
        for dist in range(1, search_distance):
            x = int(center_x - direction_x * dist)
            y = int(center_y - direction_y * dist)
            
            # 确保坐标在图像范围内
            if x < 0 or x >= self.binary_image.shape[1] or y < 0 or y >= self.binary_image.shape[0]:
                continue
            
            # 如果从黑色区域(0)到白色区域(255)，找到边缘
            if self.binary_image[y, x] > 0:  # 白色像素
                left_edge = (x, y)
                break
        
        # 沿正方向搜索右边缘
        for dist in range(1, search_distance):
            x = int(center_x + direction_x * dist)
            y = int(center_y + direction_y * dist)
            
            # 确保坐标在图像范围内
            if x < 0 or x >= self.binary_image.shape[1] or y < 0 or y >= self.binary_image.shape[0]:
                continue
            
            # 如果从黑色区域(0)到白色区域(255)，找到边缘
            if self.binary_image[y, x] > 0:  # 白色像素
                right_edge = (x, y)
                break
        
        # 如果找到两个边缘点，计算宽度
        if left_edge and right_edge:
            width = np.sqrt((right_edge[0] - left_edge[0])**2 + (right_edge[1] - left_edge[1])**2)
            
            # 检查测量到的宽度是否合理（与预期宽度接近）
            if abs(width - expected_width) > expected_width * 0.5:
                # 宽度偏差过大，可能测量不准确，使用预期宽度作为回退
                # 计算预期的边缘位置
                left_x = int(center_x - direction_x * expected_width / 2)
                left_y = int(center_y - direction_y * expected_width / 2)
                right_x = int(center_x + direction_x * expected_width / 2)
                right_y = int(center_y + direction_y * expected_width / 2)
                
                left_edge = (left_x, left_y)
                right_edge = (right_x, right_y)
                width = expected_width
            
            return width, left_edge, right_edge
        
        # 如果边缘检测失败，使用预期宽度计算边缘位置
        left_x = int(center_x - direction_x * expected_width / 2)
        left_y = int(center_y - direction_y * expected_width / 2)
        right_x = int(center_x + direction_x * expected_width / 2)
        right_y = int(center_y + direction_y * expected_width / 2)
        
        left_edge = (left_x, left_y)
        right_edge = (right_x, right_y)
        
        return expected_width, left_edge, right_edge

    def restoreOriginalImage(self):
        """恢复原始图像"""
        if hasattr(self, 'original_image_before_crop') and self.original_image_before_crop is not None:
            # 恢复原始图像
            self.original_image = self.original_image_before_crop.copy()
            self.displayImage(self.original_image, self.originalImageLabel)
            
            # 清空旧的处理结果
            self.binary_image = None
            self.result_image = None
            self.binaryImageLabel.clear()
            self.resultImageLabel.clear()
            
            # 禁用恢复按钮
            self.restoreAction.setEnabled(False)
            
            # 清除备份
            self.original_image_before_crop = None
            
            # 处理恢复后的图像
            self.processImage()
            
            QMessageBox.information(self, "恢复原图", "原始图像已恢复")
        else:
            QMessageBox.warning(self, "恢复原图", "没有可用的原始图像备份")

    def showCropConfirmation(self):
        """显示裁剪确认对话框"""
        print("进入showCropConfirmation方法")
        if not self.is_cropping:
            print("未处于裁剪模式，退出")
            return
            
        if self.original_image is None:
            print("原始图像为空，退出")
            return
        
        print(f"裁剪模式: {self.is_cropping}")
            
        # 获取裁剪区域
        crop_start = self.originalImageLabel.crop_start
        crop_end = self.originalImageLabel.crop_end
        
        print(f"裁剪起点: {crop_start}")
        print(f"裁剪终点: {crop_end}")
        
        if crop_start and crop_end:
            try:
                print("裁剪点有效，准备计算裁剪区域")
                # 确保裁剪区域有效
                x_min = min(crop_start.x(), crop_end.x())
                y_min = min(crop_start.y(), crop_end.y())
                x_max = max(crop_start.x(), crop_end.x())
                y_max = max(crop_start.y(), crop_end.y())
                
                print(f"裁剪区域原始坐标: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                
                # 获取原始图像尺寸
                height, width = self.original_image.shape
                print(f"原始图像尺寸: {width}x{height}")
                
                # 确保坐标在图像范围内
                x_min = max(0, min(x_min, width-1))
                y_min = max(0, min(y_min, height-1))
                x_max = max(0, min(x_max, width-1))
                y_max = max(0, min(y_max, height-1))
                
                print(f"调整后裁剪区域坐标: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                
                # 计算裁剪区域大小
                crop_width = x_max - x_min + 1
                crop_height = y_max - y_min + 1
                
                # 确保有最小尺寸
                if crop_width < 10 or crop_height < 10:
                    print("裁剪区域太小，通知用户重新选择")
                    QMessageBox.warning(self, "裁剪区域过小", "选择的裁剪区域太小，请重新选择更大的区域")
                    return
                    
                # 显示确认对话框，包含更详细的信息
                message = f"""确定裁剪所选区域?

原始图像尺寸: {width} × {height} 像素
裁剪区域尺寸: {crop_width} × {crop_height} 像素
裁剪区域坐标: ({x_min}, {y_min}) - ({x_max}, {y_max})

点击"是"执行裁剪，点击"否"取消并重新选择。"""
                
                print("显示确认对话框")
                reply = QMessageBox.question(
                    self, 
                    "确认裁剪", 
                    message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                # 如果用户确认，执行裁剪
                if reply == QMessageBox.Yes:
                    print("用户确认裁剪，开始执行...")
                    self.cropSelectionFinished()
                else:
                    print("用户取消了裁剪操作")
                    # 保持裁剪模式，用户可以重新选择
                
            except Exception as e:
                QMessageBox.warning(self, "确认裁剪失败", f"处理裁剪区域时出错: {str(e)}")
                print(f"确认裁剪错误: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("裁剪点无效，无法显示确认对话框")
            QMessageBox.warning(self, "裁剪失败", "未获取到有效的裁剪区域，请重新选择")
            return

    def calculateTaper(self):
        """计算孔洞锥度并显示结果"""
        try:
            if self.original_image is None:
                QMessageBox.warning(self, "警告", "请先加载图像")
                return
                
            # 创建结果图像 - 使用原始图像的深拷贝以避免修改原图
            result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
            
            # 确保已经处理过图像
            if not hasattr(self, 'upper_surface_row') or not hasattr(self, 'hole_start') or not hasattr(self, 'hole_end'):
                QMessageBox.warning(self, "警告", "请先点击处理按钮以检测孔洞")
                return
            
            # 创建调试目录
            debug_dir = 'debug'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
                
            # 改进图像增强以便更好地检测边缘
            enhanced_image = self.original_image.copy()
            # 应用CLAHE（对比度受限自适应直方图均衡化）增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(enhanced_image)
            
            # 结合Sobel和Canny边缘增强可靠性
            sobelx = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            
            # 使用已知的顶部位置的洞口宽度
            top_left = (self.hole_start, self.upper_surface_row)
            top_right = (self.hole_end, self.upper_surface_row)
            
            # 计算搜索范围
            bottom_half_start = self.upper_surface_row + 50  # 从顶部下方50像素开始搜索
            search_end_row = min(self.original_image.shape[0], 
                              int(self.original_image.shape[0] * self.params['bottom_search_range']))
            
            # 计算中心区域和期望宽度
            hole_center_x = (self.hole_start + self.hole_end) // 2
            hole_width = self.hole_end - self.hole_start
            expected_bottom_width = hole_width * 0.85  # 假设底部略窄于顶部
            
            # ======= 改进的底部短横线检测 =======
            # 1. 创建感兴趣区域ROI，集中在孔洞中心区域
            search_width = max(hole_width, 60)  # 保证搜索宽度足够
            roi_x_start = max(0, hole_center_x - search_width // 2)
            roi_x_end = min(self.original_image.shape[1], hole_center_x + search_width // 2)
            
            # 选择合适的ROI区域
            roi_height = search_end_row - bottom_half_start
            roi = self.original_image[bottom_half_start:search_end_row, roi_x_start:roi_x_end]
            
            # 保存ROI用于调试
            cv2.imwrite(os.path.join(debug_dir, "taper_roi.jpg"), roi)
            
            # 2. 使用不同的预处理方法增强ROI中的短横线
            # 使用自适应阈值处理
            roi_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 5)
            
            # 使用边缘增强
            roi_enhanced = cv2.GaussianBlur(roi, (5, 5), 0)
            roi_edges = cv2.Canny(roi_enhanced, 30, 100)
            
            # 形态学操作增强水平结构
            kernel_h = np.ones((1, 15), np.uint8)  # 水平核
            roi_horizontal = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_h)
            
            # 保存处理后的ROI用于调试
            cv2.imwrite(os.path.join(debug_dir, "taper_roi_binary.jpg"), roi_binary)
            cv2.imwrite(os.path.join(debug_dir, "taper_roi_edges.jpg"), roi_edges)
            cv2.imwrite(os.path.join(debug_dir, "taper_roi_horizontal.jpg"), roi_horizontal)
            
            # 3. 计算行投影，查找短横线的位置
            row_projection = np.sum(roi_horizontal, axis=1)
            
            # 平滑投影曲线
            row_projection_smooth = np.convolve(row_projection, np.ones(5)/5, mode='same')
            
            # 保存投影曲线用于调试
            plt.figure(figsize=(10, 4))
            plt.plot(row_projection_smooth)
            plt.title("底部短横线行投影")
            plt.savefig(os.path.join(debug_dir, "taper_row_projection.png"))
            plt.close()
            
            # 4. 找到投影中的峰值，对应短横线位置
            peaks = []
            for i in range(1, len(row_projection_smooth) - 1):
                if (row_projection_smooth[i] > row_projection_smooth[i-1] and 
                    row_projection_smooth[i] > row_projection_smooth[i+1]):
                    intensity = row_projection_smooth[i]
                    # 筛选明显的峰值，避免噪声
                    if intensity > np.mean(row_projection_smooth) + np.std(row_projection_smooth):
                        peaks.append((i, intensity))
            
            # 按照投影强度降序排序峰值
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # 记录最佳横线位置和点
            best_short_line = None
            bottom_left = None
            bottom_right = None
            best_short_line_row = None
            best_short_line_score = 0
            
            # 5. 分析每个可能的短横线
            for peak_idx, _ in peaks[:5]:  # 仅考虑前5个最强峰值
                # 将峰值映射回原始图像坐标
                peak_row = bottom_half_start + peak_idx
                
                # 获取该行像素
                line_roi = roi_horizontal[peak_idx, :]
                line_original = self.original_image[peak_row, roi_x_start:roi_x_end]
                
                # 寻找该行中的所有连通区域
                transitions = np.where(np.diff(line_roi) != 0)[0]
                if len(transitions) < 2:
                    continue
                
                # 分析横线的特征
                segments = []
                for i in range(0, len(transitions) - 1, 2):
                    if i + 1 < len(transitions):
                        start_pos = transitions[i]
                        end_pos = transitions[i + 1]
                        length = end_pos - start_pos
                        
                        # 检查是否足够长且位于中心区域
                        segment_center = roi_x_start + (start_pos + end_pos) // 2
                        dist_from_center = abs(segment_center - hole_center_x)
                        
                        # 计算段的评分
                        length_score = min(length / expected_bottom_width, 1.0)
                        center_score = 1.0 - min(dist_from_center / (search_width / 2), 1.0)
                        
                        # 检查颜色对比度（短横线应该较亮）
                        if start_pos > 0 and end_pos < len(line_original):
                            segment_avg = np.mean(line_original[start_pos:end_pos])
                            surrounding_avg = np.mean([np.mean(line_original[:start_pos]) if start_pos > 0 else 0, 
                                                    np.mean(line_original[end_pos:]) if end_pos < len(line_original) else 0])
                            contrast_score = max(0, min((segment_avg - surrounding_avg) / 50, 1.0))
                        else:
                            contrast_score = 0
                        
                        # 综合评分
                        total_score = length_score * 0.4 + center_score * 0.4 + contrast_score * 0.2
                        
                        if length >= 5:  # 最小长度阈值
                            segments.append({
                                'start': roi_x_start + start_pos,
                                'end': roi_x_start + end_pos,
                                'length': length,
                                'score': total_score,
                                'row': peak_row
                            })
                
                # 寻找最佳段
                if segments:
                    best_segment = max(segments, key=lambda s: s['score'])
                    
                    # 如果这个段比之前找到的更好，则更新
                    if best_segment['score'] > best_short_line_score:
                        best_short_line_score = best_segment['score']
                        bottom_left = (best_segment['start'], best_segment['row'])
                        bottom_right = (best_segment['end'], best_segment['row'])
                        best_short_line_row = best_segment['row']
                        best_short_line = best_segment
            
            # 如果没有找到好的短横线，使用传统方法尝试查找
            if bottom_left is None or bottom_right is None:
                # 尝试使用水平线霍夫变换
                roi_edges = cv2.Canny(roi, 50, 150)
                lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 20, 
                                     minLineLength=expected_bottom_width*0.3, 
                                     maxLineGap=20)
                
                if lines is not None:
                    best_line = None
                    best_score = 0
                    
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # 转换回原始图像坐标
                        x1 += roi_x_start
                        x2 += roi_x_start
                        y1 += bottom_half_start
                        y2 += bottom_half_start
                        
                        # 检查是否为水平线
                        if abs(y2 - y1) <= 3:  # 允许轻微倾斜
                            # 计算评分
                            length = abs(x2 - x1)
                            length_score = min(length / expected_bottom_width, 1.0)
                            
                            center_pos = (x1 + x2) // 2
                            center_score = 1.0 - min(abs(center_pos - hole_center_x) / (hole_width / 2), 1.0)
                            
                            # 检查亮度对比度
                            y_mid = (y1 + y2) // 2
                            if y_mid < self.original_image.shape[0]:
                                line_segment = self.original_image[y_mid, min(x1, x2):max(x1, x2)]
                                if len(line_segment) > 0:
                                    segment_avg = np.mean(line_segment)
                                    above_row = max(0, y_mid - 3)
                                    below_row = min(self.original_image.shape[0] - 1, y_mid + 3)
                                    surrounding_avg = (np.mean(self.original_image[above_row, min(x1, x2):max(x1, x2)]) + 
                                                      np.mean(self.original_image[below_row, min(x1, x2):max(x1, x2)])) / 2
                                    contrast_score = max(0, min((segment_avg - surrounding_avg) / 30, 1.0))
                                else:
                                    contrast_score = 0
                            else:
                                contrast_score = 0
                                
                            total_score = length_score * 0.4 + center_score * 0.4 + contrast_score * 0.2
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_line = (x1, y1, x2, y2)
                    
                    if best_line and best_score > 0.5:
                        x1, y1, x2, y2 = best_line
                        bottom_left = (min(x1, x2), (y1 + y2) // 2)
                        bottom_right = (max(x1, x2), (y1 + y2) // 2)
                        best_short_line_row = (y1 + y2) // 2
            
            # 如果仍然没有找到，尝试使用二值图像中的转换点
            if bottom_left is None or bottom_right is None:
                # 在下半部分中搜索黑白转换模式
                for row in range(int((bottom_half_start + search_end_row) * 0.7), bottom_half_start, -2):  # 从下往上搜索
                    line = self.binary_image[row, :]
                    transitions = np.where(np.diff(line) != 0)[0]
                    
                    if len(transitions) >= 2:
                        # 分析所有可能的段
                        candidates = []
                        for i in range(0, len(transitions) - 1):
                            start_idx = transitions[i]
                            end_idx = transitions[i + 1]
                            length = end_idx - start_idx
                            
                            # 检查是白到黑再到白的模式（黑色短横线）
                            if (i < len(transitions) - 1 and 
                                line[max(0, start_idx-1)] == 255 and 
                                line[min(start_idx+1, len(line)-1)] == 0 and
                                line[min(end_idx+1, len(line)-1)] == 255):
                                
                                # 计算评分
                                center_pos = (start_idx + end_idx) // 2
                                dist_from_hole_center = abs(center_pos - hole_center_x)
                                
                                if dist_from_hole_center < hole_width * 0.7:
                                    length_score = min(length / expected_bottom_width, 1.0)
                                    center_score = 1.0 - min(dist_from_hole_center / (hole_width / 2), 1.0)
                                    total_score = length_score * 0.6 + center_score * 0.4
                                    
                                    candidates.append({
                                        'start': start_idx,
                                        'end': end_idx,
                                        'score': total_score
                                    })
                        
                        if candidates:
                            best_candidate = max(candidates, key=lambda c: c['score'])
                            if best_candidate['score'] > 0.5:
                                bottom_left = (best_candidate['start'], row)
                                bottom_right = (best_candidate['end'], row)
                                best_short_line_row = row
                                break
            
            # 如果仍然没有找到，回退到图像下部位置估计
            if bottom_left is None or bottom_right is None:
                # 估计底部位置
                estimated_bottom_row = int(self.upper_surface_row + (self.original_image.shape[0] - self.upper_surface_row) * 0.7)
                bottom_width = (self.hole_end - self.hole_start) * 0.8  # 假设底部宽度为顶部的80%
                bottom_left = (int(hole_center_x - bottom_width / 2), estimated_bottom_row)
                bottom_right = (int(hole_center_x + bottom_width / 2), estimated_bottom_row)
                best_short_line_row = estimated_bottom_row
            
            # 确保所有点的坐标都是整数
            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            # 绘制辅助线以便更好地可视化锥度
            cv2.line(result_img, (0, top_left[1]), (result_img.shape[1], top_left[1]), (0, 200, 0), 1)  # 淡绿色顶部辅助线
            
            # 使用红色显著标记底部短横线
            if best_short_line_row is not None:
                cv2.line(result_img, (0, best_short_line_row), (result_img.shape[1], best_short_line_row), (0, 200, 0), 1)  # 淡绿色底部辅助线
            
            # 在图像上绘制连接线
            cv2.line(result_img, top_left, bottom_left, (255, 0, 0), 2)  # 蓝色
            cv2.line(result_img, top_right, bottom_right, (255, 0, 0), 2)  # 蓝色
            
            # 标记四个关键点
            cv2.circle(result_img, top_left, 5, (255, 0, 0), -1)  # 蓝色
            cv2.circle(result_img, top_right, 5, (255, 0, 0), -1)  # 蓝色
            cv2.circle(result_img, bottom_left, 5, (0, 255, 0), -1)  # 绿色
            cv2.circle(result_img, bottom_right, 5, (0, 255, 0), -1)  # 绿色
            
            # 如果找到了短横线，显著标记它
            if best_short_line:
                # 使用红色标记短横线
                short_line_start = bottom_left[0]
                short_line_end = bottom_right[0]
                short_line_row = bottom_left[1]
                cv2.line(result_img, (short_line_start, short_line_row), 
                       (short_line_end, short_line_row), (0, 0, 255), 2)  # 红色标记短横线
            
            # 计算锥度
            top_width = top_right[0] - top_left[0]
            bottom_width = bottom_right[0] - bottom_left[0]
            height = bottom_left[1] - top_left[1]  # 垂直高度
            
            if height > 0:
                # 锥度计算 = (顶部宽度 - 底部宽度) / (2 * 高度)
                taper = (top_width - bottom_width) / (2 * height)
                taper_angle = np.arctan(taper) * 180 / np.pi
                
                # 创建结果图像的副本以避免修改原图
                display_img = result_img.copy()
                
                # 直接使用OpenCV绘制文本
                y_offset = 30
                for i, text in enumerate([
                    f"顶部宽度: {top_width:.2f} 像素",
                    f"底部宽度: {bottom_width:.2f} 像素",
                    f"孔洞高度: {height:.2f} 像素",
                    f"锥度: {taper:.6f}",
                    f"锥度角度: {taper_angle:.2f}°"
                ]):
                    cv2.putText(display_img, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
                
                # 显示锥度结果
                self.showTaperResult(display_img, top_width, bottom_width, height, taper, taper_angle)
            else:
                QMessageBox.warning(self, "警告", "无法计算锥度：未找到有效的顶部和底部")
        except Exception as e:
            print(f"计算锥度时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"计算锥度时出错: {str(e)}")

    def showTaperResult(self, result_img, top_width, bottom_width, height, taper, taper_angle):
        """显示锥度计算结果"""
        try:
            # 创建结果对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("锥度计算结果")
            dialog.setMinimumSize(900, 700)  # 增加窗口大小
            
            # 创建布局
            layout = QVBoxLayout()
            
            # 创建滚动区域以确保大图片可以完整显示
            scrollArea = QScrollArea()
            scrollArea.setWidgetResizable(True)
            
            # 创建图像显示区域
            imageWidget = QWidget()
            imageLayout = QVBoxLayout(imageWidget)
            
            # 确保result_img不为None
            if result_img is None:
                print("警告: 结果图像为空，使用原始图像替代")
                result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
                
            imgLabel = QLabel()
            h, w, c = result_img.shape
            bytesPerLine = 3 * w
            qImg = QImage(result_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            
            # 设置合适的图像大小
            orig_width = pixmap.width()
            if orig_width > 800:
                pixmap = pixmap.scaled(800, int(h * 800 / w), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            imgLabel.setPixmap(pixmap)
            imgLabel.setAlignment(Qt.AlignCenter)
            imageLayout.addWidget(imgLabel)
            
            scrollArea.setWidget(imageWidget)
            
            # 创建结果显示区域
            resultGroupBox = QGroupBox("测量结果")
            resultLayout = QFormLayout()
            
            # 像素测量值
            topWidthLabel = QLabel(f"{top_width:.2f} 像素")
            bottomWidthLabel = QLabel(f"{bottom_width:.2f} 像素")
            heightLabel = QLabel(f"{height:.2f} 像素")
            
            # 转换为物理尺寸 (假设PIXEL_TO_UM是定义的像素到微米的转换系数)
            if 'PIXEL_TO_UM' in globals():
                topWidthPhysical = top_width * PIXEL_TO_UM_X
                bottomWidthPhysical = bottom_width * PIXEL_TO_UM_X
                heightPhysical = height * PIXEL_TO_UM_Y
                
                topWidthLabel.setText(f"{top_width:.2f} 像素 ({topWidthPhysical:.2f} 微米)")
                bottomWidthLabel.setText(f"{bottom_width:.2f} 像素 ({bottomWidthPhysical:.2f} 微米)")
                heightLabel.setText(f"{height:.2f} 像素 ({heightPhysical:.2f} 微米)")
            
            taperLabel = QLabel(f"{taper:.6f}")
            taperAngleLabel = QLabel(f"{taper_angle:.2f}°")
            
            resultLayout.addRow("顶部宽度:", topWidthLabel)
            resultLayout.addRow("底部宽度:", bottomWidthLabel)
            resultLayout.addRow("孔洞高度:", heightLabel)
            resultLayout.addRow("锥度 (倾斜率):", taperLabel)
            resultLayout.addRow("锥度角度:", taperAngleLabel)
            
            resultGroupBox.setLayout(resultLayout)
            
            # 添加到主布局
            layout.addWidget(scrollArea, 3)  # 图像区域占据更多空间
            layout.addWidget(resultGroupBox, 1)
            
            # 添加按钮
            buttonBox = QHBoxLayout()
            saveButton = QPushButton("保存结果")
            closeButton = QPushButton("关闭")
            
            saveButton.clicked.connect(lambda: self.saveTaperResult(result_img, top_width, bottom_width, height, taper, taper_angle))
            closeButton.clicked.connect(dialog.accept)
            
            buttonBox.addWidget(saveButton)
            buttonBox.addWidget(closeButton)
            layout.addLayout(buttonBox)
            
            dialog.setLayout(layout)
            dialog.exec_()
        except Exception as e:
            print(f"显示锥度结果时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"显示锥度结果时出错: {str(e)}")

    def saveTaperResult(self, result_img, top_width, bottom_width, height, taper, taper_angle):
        """保存锥度计算结果"""
        # 创建保存文件对话框
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "保存锥度结果", "", 
                                               "图像文件 (*.png);;所有文件 (*)", options=options)
        if fileName:
            # 确保文件名有扩展名
            if not fileName.endswith('.png'):
                fileName += '.png'
            
            # 在图像上添加测量结果文本
            result_copy = result_img.copy()
            
            # 添加测量结果到图像
            info_text = [
                f"顶部宽度: {top_width:.2f} 像素",
                f"底部宽度: {bottom_width:.2f} 像素",
                f"孔洞高度: {height:.2f} 像素",
                f"锥度: {taper:.6f}",
                f"锥度角度: {taper_angle:.2f}°"
            ]
            
            # 如果存在物理尺寸转换，添加物理尺寸信息
            info_text[0] += f" ({top_width * PIXEL_TO_UM_X:.2f} 微米)"
            info_text[1] += f" ({bottom_width * PIXEL_TO_UM_X:.2f} 微米)"
            info_text[2] += f" ({height * PIXEL_TO_UM_Y:.2f} 微米)"
            
            # 在图像上添加文本
            y_offset = 30
            for text in info_text:
                result_copy = self.addChineseText(result_copy, text, (20, y_offset), 
                                                textColor=(0, 0, 255), bgColor=(255, 255, 255), fontSize=30)
                y_offset += 40
            
            try:
                # 使用numpy处理中文路径问题
                is_success, im_buf_arr = cv2.imencode(".png", result_copy)
                if is_success:
                    with open(fileName, 'wb') as f:
                        im_buf_arr.tofile(f)
                else:
                    raise Exception("无法编码图像")
                
                # 创建CSV文件名
                csv_fileName = fileName.rsplit('.', 1)[0] + '.csv'
                
                # 保存CSV数据
                with open(csv_fileName, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['参数', '数值'])
                    writer.writerow(['顶部宽度 (像素)', f'{top_width:.2f}'])
                    writer.writerow(['底部宽度 (像素)', f'{bottom_width:.2f}'])
                    writer.writerow(['孔洞高度 (像素)', f'{height:.2f}'])
                    writer.writerow(['锥度', f'{taper:.6f}'])
                    writer.writerow(['锥度角度 (度)', f'{taper_angle:.2f}'])
                    
                    # 添加物理尺寸
                    if 'PIXEL_TO_UM' in globals():
                        writer.writerow(['顶部宽度 (微米)', f'{top_width * PIXEL_TO_UM:.2f}'])
                        writer.writerow(['底部宽度 (微米)', f'{bottom_width * PIXEL_TO_UM:.2f}'])
                        writer.writerow(['孔洞高度 (微米)', f'{height * PIXEL_TO_UM:.2f}'])
                
                QMessageBox.information(self, "保存成功", f"锥度结果已保存到:\n{fileName}\n{csv_fileName}")
            
            except Exception as e:
                print(f"保存锥度结果时发生错误: {str(e)}")
                QMessageBox.warning(self, "保存错误", f"保存文件时出错: {str(e)}")

    def directTaperCalculation(self):
        """直接计算锥度，通过让用户选择顶部和底部的行"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("直接锥度计算")
        mainLayout = QVBoxLayout()
        
        topSpin = QSpinBox()
        bottomSpin = QSpinBox()
        rangeLabel = QLabel("有效范围: -- 像素")

        # 图像显示
        imgLabel = QLabel()
        pixmap = QPixmap.fromImage(QImage(cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB).data, 
                                        self.original_image.shape[1], self.original_image.shape[0], 
                                        QImage.Format_RGB888))
        imgLabel.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        mainLayout.addWidget(imgLabel)

        topLayout = QHBoxLayout()
        topLayout.addWidget(QLabel("上表面行:"))
        topLayout.addWidget(topSpin)
        topSpin.setRange(0, self.original_image.shape[0] - 1)
        topSpin.setValue(int(self.original_image.shape[0] * 0.2))  # 默认在图像1/5位置
        mainLayout.addLayout(topLayout)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(QLabel("底表面行:"))
        bottomLayout.addWidget(bottomSpin)
        bottomSpin.setRange(0, self.original_image.shape[0] - 1)
        bottomSpin.setValue(int(self.original_image.shape[0] * 0.8))  # 默认在图像4/5位置
        mainLayout.addLayout(bottomLayout)
        mainLayout.addWidget(rangeLabel)

        def updateRangeLabel(value):
            top_row = topSpin.value()
            bottom_row = bottomSpin.value()
            if bottom_row > top_row:
                rangeLabel.setText(f"有效范围: {bottom_row - top_row} 像素")
            else:
                rangeLabel.setText("底部必须在顶部下方")

        topSpin.valueChanged.connect(updateRangeLabel)
        bottomSpin.valueChanged.connect(updateRangeLabel)

        # 确认按钮
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        mainLayout.addWidget(buttonBox)

        dialog.setLayout(mainLayout)

        if dialog.exec_():
            # 计算锥度
            top_row = topSpin.value()
            bottom_row = bottomSpin.value()

            # 确保底部在顶部之下
            if bottom_row <= top_row:
                QMessageBox.warning(self, "警告", "底部必须在顶部下方")
                return

            # 计算锥度和锥度角
            top_left = self.find_hole_edge_at_row_from_left(top_row)
            top_right = self.find_hole_edge_at_row_from_right(top_row)
            bottom_left = self.find_hole_edge_at_row_from_left(bottom_row)
            bottom_right = self.find_hole_edge_at_row_from_right(bottom_row)

            if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
                QMessageBox.warning(self, "警告", "无法检测到孔洞边缘")
                return

            # 计算直径和高度
            top_width = (top_right - top_left) * PIXEL_TO_UM_X
            bottom_width = (bottom_right - bottom_left) * PIXEL_TO_UM_X
            height = (bottom_row - top_row) * PIXEL_TO_UM_Y

            # 计算锥度和锥度角
            taper = ((top_width - bottom_width) / height) * 100  # 百分比表示
            taper_angle = np.degrees(np.arctan2((top_width - bottom_width) / 2, height))  # 角度

            # 在结果图像上标记测量位置和结果
            # 绘制顶部和底部测量线
            cv2.line(self.result_image, (top_left, top_row), (top_right, top_row), (0, 255, 0), 2)
            cv2.line(self.result_image, (bottom_left, bottom_row), (bottom_right, bottom_row), (0, 0, 255), 2)
            
            # 绘制连接线以显示锥度
            cv2.line(self.result_image, (top_left, top_row), (bottom_left, bottom_row), (255, 0, 0), 1)
            cv2.line(self.result_image, (top_right, top_row), (bottom_right, bottom_row), (255, 0, 0), 1)
            
            # 在图像上添加测量结果标签
            self.addChineseText(self.result_image, f"顶部宽度: {top_width:.2f} μm", (10, 30), 
                              (0, 255, 0), (0, 0, 0))
            self.addChineseText(self.result_image, f"底部宽度: {bottom_width:.2f} μm", (10, 60), 
                              (0, 0, 255), (0, 0, 0))
            self.addChineseText(self.result_image, f"高度: {height:.2f} μm", (10, 90), 
                              (255, 255, 0), (0, 0, 0))
            self.addChineseText(self.result_image, f"锥度: {taper:.2f}%", (10, 120), 
                              (255, 0, 0), (0, 0, 0))
            self.addChineseText(self.result_image, f"锥度角: {taper_angle:.2f}°", (10, 150), 
                              (255, 0, 255), (0, 0, 0))
            
            # 显示结果对话框
            self.showEnhancedTaperResult(self.result_image, top_width, bottom_width, height, taper, taper_angle)

    def analyzeRoughness(self):
        if self.binary_image is None:
            QMessageBox.warning(self, "警告", "请先加载并处理图像")
            return
            
        try:
            if not hasattr(self, 'upper_surface_row') or not hasattr(self, 'bottom_surface_row'):
                QMessageBox.warning(self, "警告", "无法进行粗糙度分析：缺少孔洞尺寸信息")
                return
            
            hole_mask = np.zeros_like(self.binary_image)
            cv2.rectangle(hole_mask, (self.hole_start, self.upper_surface_row), 
                          (self.hole_end, self.bottom_surface_row), 255, -1)
            
            inner_wall_roi = cv2.bitwise_and(self.binary_image, hole_mask)
            edges = cv2.Canny(inner_wall_roi, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            long_contours = []
            min_contour_length = (self.bottom_surface_row - self.upper_surface_row) * 0.3
            for contour in contours:
                if len(contour) > min_contour_length:
                    _, _, w, h = cv2.boundingRect(contour)
                    if h > w * 2:
                        long_contours.append(contour)
            
            result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result_img, long_contours, -1, (0, 255, 0), 1)
            
            if len(long_contours) > 0:
                all_points = np.vstack(long_contours)
                pca = PCA(n_components=1)
                pca.fit(all_points.reshape(-1, 2))
                main_axis = pca.components_[0]
                mean_point = np.mean(all_points.reshape(-1, 2), axis=0)
                deviations = np.abs(np.dot(all_points.reshape(-1, 2) - mean_point, 
                                          np.array([-main_axis[1], main_axis[0]])))
                avg_smoothness = np.std(deviations)
                roughness_score = 100 / (1 + avg_smoothness)
                
                if roughness_score > 80:
                    roughness_grade = "非常光滑"
                elif roughness_score > 60:
                    roughness_grade = "光滑"
                elif roughness_score > 40:
                    roughness_grade = "一般"
                else:
                    roughness_grade = "粗糙"
                
                curvatures = []
                for contour in long_contours:
                    arc_length = cv2.arcLength(contour, closed=False)
                    start_point = contour[0][0]
                    end_point = contour[-1][0]
                    chord_length = np.linalg.norm(start_point - end_point)
                    if chord_length > 1e-6:
                        curvatures.append(arc_length / chord_length)
                
                avg_curvature = np.mean(curvatures) if curvatures else 1.0
                self.showRoughnessResult(result_img, roughness_score, roughness_grade, avg_smoothness, 
                                         avg_curvature, len(long_contours))
            else:
                QMessageBox.warning(self, "警告", "无法计算粗糙度：未找到有效的轮廓")
                
        except Exception as e:
            print(f"粗糙度分析时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"在粗糙度分析过程中发生错误: {e}")

    def showRoughnessResult(self, result_img, roughness_score, roughness_grade, 
                           smoothness, curvature, contour_count):
        """显示粗糙度分析结果"""
        try:
            # 创建结果对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("孔洞粗糙度分析结果")
            dialog.setMinimumSize(900, 700)
            
            # 创建布局
            layout = QVBoxLayout()
            
            # 创建滚动区域以确保大图片可以完整显示
            scrollArea = QScrollArea()
            scrollArea.setWidgetResizable(True)
            
            # 创建图像显示区域
            imageWidget = QWidget()
            imageLayout = QVBoxLayout(imageWidget)
            
            # 确保result_img不为None
            if result_img is None:
                print("警告: 结果图像为空，使用原始图像替代")
                result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
                
            imgLabel = QLabel()
            h, w, c = result_img.shape
            bytesPerLine = 3 * w
            qImg = QImage(result_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            
            # 设置合适的图像大小
            orig_width = pixmap.width()
            if orig_width > 800:
                pixmap = pixmap.scaled(800, int(h * 800 / w), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            imgLabel.setPixmap(pixmap)
            imgLabel.setAlignment(Qt.AlignCenter)
            imageLayout.addWidget(imgLabel)
            
            scrollArea.setWidget(imageWidget)
            
            # 创建结果显示区域
            resultGroupBox = QGroupBox("粗糙度测量结果")
            resultLayout = QFormLayout()
            
            # 粗糙度分数和等级
            roughnessScoreText = f"{roughness_score:.2f}/10.00"
            
            # 添加彩色评级标签
            roughnessGradeLabel = QLabel(roughness_grade)
            if roughness_grade == "极光滑":
                roughnessGradeLabel.setStyleSheet("color: green; font-weight: bold;")
            elif roughness_grade == "光滑":
                roughnessGradeLabel.setStyleSheet("color: #00AA00; font-weight: bold;")
            elif roughness_grade == "中等":
                roughnessGradeLabel.setStyleSheet("color: #AAAA00; font-weight: bold;")
            elif roughness_grade == "粗糙":
                roughnessGradeLabel.setStyleSheet("color: #AA5500; font-weight: bold;")
            else:
                roughnessGradeLabel.setStyleSheet("color: red; font-weight: bold;")
            
            # 添加技术指标
            smoothnessText = f"{smoothness:.4f}"
            curvatureText = f"{curvature:.4f}"
            contourText = f"{contour_count}"
            
            resultLayout.addRow("粗糙度评分:", QLabel(roughnessScoreText))
            resultLayout.addRow("粗糙度等级:", roughnessGradeLabel)
            resultLayout.addRow("平滑度指数:", QLabel(smoothnessText))
            resultLayout.addRow("曲率变化指数:", QLabel(curvatureText))
            resultLayout.addRow("检测到的轮廓数:", QLabel(contourText))
            
            # 添加说明文本
            explainLabel = QLabel("说明: 粗糙度评分越低表示孔洞边缘越光滑，评分范围为0-10")
            explainLabel.setStyleSheet("font-style: italic; color: #666666;")
            resultLayout.addRow("", explainLabel)
            
            resultGroupBox.setLayout(resultLayout)
            
            # 添加到主布局
            layout.addWidget(scrollArea, 3)
            layout.addWidget(resultGroupBox, 1)
            
            # 添加按钮
            buttonBox = QHBoxLayout()
            saveButton = QPushButton("保存结果")
            closeButton = QPushButton("关闭")
            
            # 创建保存结果方法
            def saveRoughnessResult():
                self.saveRoughnessResult(result_img, roughness_score, roughness_grade, 
                                       smoothness, curvature, contour_count)
            
            saveButton.clicked.connect(saveRoughnessResult)
            closeButton.clicked.connect(dialog.accept)
            
            buttonBox.addWidget(saveButton)
            buttonBox.addWidget(closeButton)
            layout.addLayout(buttonBox)
            
            dialog.setLayout(layout)
            dialog.exec_()
        except Exception as e:
            print(f"显示粗糙度结果时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"显示粗糙度结果时出错: {str(e)}")
    
    def saveRoughnessResult(self, result_img, roughness_score, roughness_grade, 
                           smoothness, curvature, contour_count):
        """保存粗糙度计算结果"""
        # 创建保存文件对话框
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "保存粗糙度结果", "", 
                                               "图像文件 (*.png);;所有文件 (*)", options=options)
        if fileName:
            # 确保文件名有扩展名
            if not fileName.endswith('.png'):
                fileName += '.png'
            
            # 在图像上添加测量结果文本
            result_copy = result_img.copy()
            
            # 添加测量结果到图像
            info_text = [
                f"粗糙度评分: {roughness_score:.2f}/10.00",
                f"粗糙度等级: {roughness_grade}",
                f"平滑度指数: {smoothness:.4f}",
                f"曲率变化指数: {curvature:.4f}",
                f"检测到的轮廓数: {contour_count}"
            ]
            
            # 在图像上添加文本
            y_offset = 30
            for text in info_text:
                result_copy = self.addChineseText(result_copy, text, (20, y_offset), 
                                                textColor=(0, 0, 255), bgColor=(255, 255, 255), fontSize=30)
                y_offset += 40
            
            try:
                # 使用numpy处理中文路径问题
                is_success, im_buf_arr = cv2.imencode(".png", result_copy)
                if is_success:
                    with open(fileName, 'wb') as f:
                        im_buf_arr.tofile(f)
                else:
                    raise Exception("无法编码图像")
                
                # 创建CSV文件名
                csv_fileName = fileName.rsplit('.', 1)[0] + '.csv'
                
                # 保存CSV数据
                with open(csv_fileName, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['参数', '数值'])
                    writer.writerow(['粗糙度评分', f'{roughness_score:.2f}'])
                    writer.writerow(['粗糙度等级', roughness_grade])
                    writer.writerow(['平滑度指数', f'{smoothness:.4f}'])
                    writer.writerow(['曲率变化指数', f'{curvature:.4f}'])
                    writer.writerow(['检测到的轮廓数', f'{contour_count}'])
                
                QMessageBox.information(self, "保存成功", f"粗糙度分析结果已保存到:\n{fileName}\n{csv_fileName}")
            
            except Exception as e:
                print(f"保存粗糙度结果时发生错误: {str(e)}")
                QMessageBox.warning(self, "保存错误", f"保存文件时出错: {str(e)}")

    def showManualMeasurementResult(self, points, distance, h_distance, v_distance):
        """显示手动测量结果对话框
        
        Args:
            points: 包含两个QPoint的列表，表示用户选择的两个点
            distance: 两点之间的直线距离（像素值）
            h_distance: 水平距离（像素值）
            v_distance: 垂直距离（像素值）
        """
        # 创建结果图像
        result_img = None
        if self.original_image is not None:
            # 转换为彩色图像
            result_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            
            # 绘制测量线
            p1, p2 = points
            cv2.line(result_img, (p1.x(), p1.y()), (p2.x(), p2.y()), (0, 0, 255), 2)
            
            # 绘制端点
            cv2.circle(result_img, (p1.x(), p1.y()), 5, (255, 0, 0), -1)
            cv2.circle(result_img, (p2.x(), p2.y()), 5, (255, 0, 0), -1)
            
            # 在线中间位置显示距离
            mid_x = (p1.x() + p2.x()) // 2
            mid_y = (p1.y() + p2.y()) // 2
            
            # 将像素距离转换为实际尺寸（微米），使用不同方向的转换因子
            # 对于斜线，使用对应方向的分量计算
            um_h_distance = h_distance * PIXEL_TO_UM_X  # 水平方向使用X转换因子
            um_v_distance = v_distance * PIXEL_TO_UM_Y  # 垂直方向使用Y转换因子
            
            # 对于直线距离，根据水平和垂直分量的比例进行加权计算
            if distance > 0:
                h_weight = h_distance / distance
                v_weight = v_distance / distance
                um_distance = np.sqrt((um_h_distance * h_weight)**2 + (um_v_distance * v_weight)**2) / np.sqrt(h_weight**2 + v_weight**2)
            else:
                um_distance = 0.0
            
            # 添加标签
            distance_text = f"{um_distance:.1f} μm"
            self.addChineseText(result_img, distance_text, (mid_x + 10, mid_y), 
                               textColor=(0, 0, 255), bgColor=(255, 255, 255), fontSize=20)
            
            # 在结果图像上显示
            self.displayImage(result_img, self.resultImageLabel)
        
        # 创建结果对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("手动测量结果")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        # 添加结果表格
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        # 添加测量结果
        result_label = QLabel(f"直线距离: {distance:.1f} 像素 ({um_distance:.1f} μm)")
        form_layout.addRow("测量结果:", result_label)
        
        h_label = QLabel(f"水平距离: {h_distance:.1f} 像素 ({um_h_distance:.1f} μm)")
        form_layout.addRow("", h_label)
        
        v_label = QLabel(f"垂直距离: {v_distance:.1f} 像素 ({um_v_distance:.1f} μm)")
        form_layout.addRow("", v_label)
        
        # 添加转换因子信息
        factor_label = QLabel(f"X方向: {PIXEL_TO_UM_X:.2f} μm/px, Y方向: {PIXEL_TO_UM_Y:.2f} μm/px")
        form_layout.addRow("转换因子:", factor_label)
        
        # 添加选项，让用户选择这个测量是直径还是深度
        option_group = QGroupBox("将此测量标记为")
        option_layout = QHBoxLayout()
        
        diameter_btn = QPushButton("直径")
        diameter_btn.clicked.connect(lambda: self.saveMeasurementAsType(result_img, um_h_distance, "diameter", dialog))
        option_layout.addWidget(diameter_btn)
        
        depth_btn = QPushButton("深度")
        depth_btn.clicked.connect(lambda: self.saveMeasurementAsType(result_img, um_v_distance, "depth", dialog))
        option_layout.addWidget(depth_btn)
        
        option_group.setLayout(option_layout)
        
        # 添加取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        
        # 组装布局
        layout.addLayout(form_layout)
        layout.addWidget(option_group)
        layout.addWidget(cancel_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def saveMeasurementAsType(self, result_img, value, measurement_type, dialog):
        """将手动测量结果保存为特定类型（直径或深度）
        
        Args:
            result_img: 显示测量结果的图像
            value: 测量值（微米）
            measurement_type: 测量类型，"diameter"或"depth"
            dialog: 要关闭的对话框
        """
        try:
            # 更新测量结果
            if measurement_type == "diameter":
                self.diameter = value
                self.diameterLabel.setText(f"{value:.1f} μm")
                print(f"将手动测量结果 {value:.1f} μm 保存为直径")
            elif measurement_type == "depth":
                self.depth = value
                self.depthLabel.setText(f"{value:.1f} μm")
                print(f"将手动测量结果 {value:.1f} μm 保存为深度")
            
            # 保存结果图像
            if result_img is not None:
                self.result_image = result_img
            
            # 关闭对话框
            dialog.accept()
            
            # 显示确认消息
            QMessageBox.information(self, "测量保存", f"测量结果已保存为{measurement_type}类型")
        except Exception as e:
            print(f"保存测量结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def updateRotationAngle(self, value):
        """实时更新旋转角度滑动条的值并应用旋转"""
        # 防止递归更新
        if self.rotationInput.value() != value:
            # 更新输入框的值，但不触发valueChanged信号
            self.rotationInput.blockSignals(True)
            self.rotationInput.setValue(value)
            self.rotationInput.blockSignals(False)
        
        # 如果没有原始图像或角度为0，不做任何处理
        if self.original_image is None or value == 0:
            return
        
        # 应用旋转
        self._applyRotation(value)
    
    def applyInputRotation(self, value):
        """从输入框接收旋转角度值并应用旋转"""
        # 防止递归更新
        if self.rotationSlider.value() != value:
            # 更新滑动条的值，但不触发valueChanged信号
            self.rotationSlider.blockSignals(True)
            self.rotationSlider.setValue(value)
            self.rotationSlider.blockSignals(False)
        
        # 如果没有原始图像或角度为0，不做任何处理
        if self.original_image is None or value == 0:
            return
            
        # 应用旋转
        self._applyRotation(value)
        
    
    def find_hole_edge_at_row_from_left(self, row, start_col=None, end_col=None):
        """从左侧查找指定行的孔洞边缘
        
        Args:
            row: 要搜索的行
            start_col: 开始列（默认为0）
            end_col: 结束列（默认为图像宽度）
            
        Returns:
            找到的边缘列索引，如果未找到则返回None
        """
        if self.original_image is None or self.binary_image is None:
            return None
        
        if row < 0 or row >= self.binary_image.shape[0]:
            return None
            
        start_col = 0 if start_col is None else start_col
        end_col = self.binary_image.shape[1] if end_col is None else end_col
        
        # 获取指定行的像素值
        row_values = self.binary_image[row, start_col:end_col]
        
        # 查找从左到右的第一个黑到白的转换点（孔洞左边缘）
        edge_idx = None
        for i in range(len(row_values) - 1):
            if row_values[i] == 0 and row_values[i+1] == 255:
                edge_idx = i + start_col
                break
                
        return edge_idx
    
    def find_hole_edge_at_row_from_right(self, row, start_col=None, end_col=None):
        """从右侧查找指定行的孔洞边缘
        
        Args:
            row: 要搜索的行
            start_col: 开始列（默认为0）
            end_col: 结束列（默认为图像宽度）
            
        Returns:
            找到的边缘列索引，如果未找到则返回None
        """
        if self.original_image is None or self.binary_image is None:
            return None
        
        if row < 0 or row >= self.binary_image.shape[0]:
            return None
            
        start_col = 0 if start_col is None else start_col
        end_col = self.binary_image.shape[1] if end_col is None else end_col
        
        # 获取指定行的像素值
        row_values = self.binary_image[row, start_col:end_col]
        
        # 查找从右到左的第一个白到黑的转换点（孔洞右边缘）
        edge_idx = None
        for i in range(len(row_values) - 1, 0, -1):
            if row_values[i] == 0 and row_values[i-1] == 255:
                edge_idx = i + start_col
                break
                
        return edge_idx
        
    def _applyRotation(self, value):
        """实际执行旋转操作的内部方法"""
        # 保存原始图像的副本（如果还没有保存）
        if not hasattr(self, 'original_image_backup') or self.original_image_backup is None:
            self.original_image_backup = self.original_image.copy()
        
        # 获取原始图像尺寸
        height, width = self.original_image_backup.shape
        
        # 计算旋转所需的较大画布尺寸（确保足够容纳旋转后的图像）
        diagonal = int(np.sqrt(width**2 + height**2)) + 20  # 加一些边距
        
        # 计算原始图像边缘区域的平均灰度值作为背景色
        # 使用图像边缘的10%区域来计算
        edge_width = max(5, int(width * 0.1))
        edge_height = max(5, int(height * 0.1))
        
        # 获取四个边缘区域
        top_edge = self.original_image_backup[:edge_height, :]
        bottom_edge = self.original_image_backup[-edge_height:, :]
        left_edge = self.original_image_backup[:, :edge_width]
        right_edge = self.original_image_backup[:, -edge_width:]
        
        # 计算边缘区域的平均灰度值
        bg_samples = np.concatenate([
            top_edge.flatten(), 
            bottom_edge.flatten(), 
            left_edge.flatten(), 
            right_edge.flatten()
        ])
        
        # 使用众数作为背景色，更能代表主要背景
        # 如果计算众数太慢，可以用平均值代替
        try:
            from scipy import stats
            background_color = int(stats.mode(bg_samples, keepdims=False)[0])
        except:
            # 如果scipy不可用，使用平均值
            background_color = int(np.mean(bg_samples))
        
        # 确保背景色在有效范围内
        background_color = max(0, min(255, background_color))
        
        # 创建与原图相似背景色的正方形画布
        canvas = np.ones((diagonal, diagonal), dtype=np.uint8) * background_color
        
        # 将原始图像放在画布中央
        x_offset = (diagonal - width) // 2
        y_offset = (diagonal - height) // 2
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = self.original_image_backup
        
        # 获取旋转矩阵
        center = (diagonal // 2, diagonal // 2)  # 画布中心作为旋转中心
        rotation_matrix = cv2.getRotationMatrix2D(center, value, 1.0)
        
        # 执行旋转，使用计算的背景色而非白色
        try:
            rotated_canvas = cv2.warpAffine(canvas, rotation_matrix, (diagonal, diagonal),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                          borderValue=background_color)  # 使用计算出的背景色
            
            # 更新当前图像为旋转后的图像
            self.original_image = rotated_canvas
            self.displayImage(self.original_image, self.originalImageLabel)
            
            # 清除二值图和结果图像
            self.binary_image = None
            self.result_image = None
            self.binaryImageLabel.clear()
            self.resultImageLabel.clear()
            
            # 记录旋转角度
            self.rotation_angle = value
            # 添加标记表明图像已旋转，将使用专用算法
            self.is_image_rotated = True if value != 0 else False
            
            # 每次旋转后自动处理图像
            self.processImage()
        except Exception as e:
            QMessageBox.warning(self, "旋转失败", f"图像旋转过程中出错: {str(e)}")
            # 重置滑动条和输入框
            self.rotationSlider.blockSignals(True)
            self.rotationInput.blockSignals(True)
            self.rotationSlider.setValue(0)
            self.rotationInput.setValue(0)
            self.rotationSlider.blockSignals(False)
            self.rotationInput.blockSignals(False)

    def startOCTReconstruction(self):
        """启动OCT圆孔重建对话框"""
        try:
            # The ImageLabel class is passed to the dialog
            dialog = oct_module.OCTHoleReconstructionDialog(
                parent=self, 
                image_label_class=ImageLabel,
                reference_depth=self.reference_depth_for_oct
            )
            # 同步全局的像素到微米转换因子
            if hasattr(self, 'PIXEL_TO_UM_X'):
                dialog.pixel_to_um_x = self.PIXEL_TO_UM_X
            if hasattr(self, 'PIXEL_TO_UM_Y'):
                dialog.pixel_to_um_y = self.PIXEL_TO_UM_Y
            dialog.pixel_to_um = dialog.pixel_to_um_x  # 兼容旧代码
            dialog.exec_()
        except Exception as e:
            print(f"启动OCT圆孔重建时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"启动OCT圆孔重建时出错: {str(e)}")

    def toggleCropMode(self):
        """切换裁剪模式"""
        # 确保有图像
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        if not hasattr(self, 'originalImageLabel') or self.originalImageLabel is None:
            QMessageBox.warning(self, "警告", "UI初始化错误，裁剪功能不可用")
            return
                
        self.is_cropping = not getattr(self, 'is_cropping', False)
        self.originalImageLabel.setCropActive(self.is_cropping)
        print(f"设置裁剪模式: {self.is_cropping}")
        
        if hasattr(self, 'cropAction'):
            if self.is_cropping:
                self.cropAction.setText("取消裁剪")
            else:
                self.cropAction.setText("裁剪图像")
                print("裁剪模式关闭，恢复默认光标")
        
        print(f"裁剪模式切换完成: {self.is_cropping}")
        print(f"原始图像标签是否有效: {self.originalImageLabel is not None}, 裁剪状态: {getattr(self.originalImageLabel, 'crop_active', 'N/A')}")

    def toggleManualMeasureMode(self):
        """切换手动测量模式"""
        # 获取当前图像标签
        image_label = self.originalImageLabel
        
        # 如果没有图像，则不能进行手动测量
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        # 切换手动测量模式
        self.is_manual_measure_active = not getattr(self, 'is_manual_measure_active', False)
        
        # 更新图像标签的状态
        image_label.setManualMeasureActive(self.is_manual_measure_active)
        
        # 如果启用了手动测量，禁用裁剪功能；反之亦然
        if hasattr(self, 'cropAction'):
            self.cropAction.setEnabled(not self.is_manual_measure_active)
        
        # 显示或隐藏状态提示
        if self.is_manual_measure_active:
            self.statusBar().showMessage("手动测量模式：选择两点进行测量")
            # 添加新的提示对话框，解释示波器光标的用途和操作方式
            QMessageBox.information(self, "手动测量模式", 
                                   "现在进入手动测量模式：\n\n"
                                   "- 左键点击：确定测量点\n"
                                   "- 右键点击：切换测量模式（直径/深度）\n\n"
                                   "当前模式：\n"
                                   "- 红色垂直线（→）用于测量孔洞直径\n"
                                   "- 蓝色水平线（↓）用于测量孔洞深度\n\n"
                                   "请点击两点来完成测量。测量完成后，可以选择将结果保存为直径或深度。")
        else:
            self.statusBar().clearMessage()
        
        # 更新UI
        # self.process_btn.setEnabled(not self.is_manual_measure_active)  # 处于手动测量模式时禁用处理按钮

    def manualMeasurementFinished(self, points):
        """处理手动测量完成事件
        
        Args:
            points: 包含两个QPoint的列表，表示用户选择的两个点
        """
        try:
            if len(points) != 2:
                print("错误：需要恰好两个点进行测量")
                return
                
            # 计算两点之间的距离
            p1, p2 = points
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            
            # 直线距离（像素）
            distance = np.sqrt(dx * dx + dy * dy)
            
            # 水平和垂直距离（像素）
            h_distance = abs(dx)
            v_distance = abs(dy)
            
            print(f"手动测量结果：直线距离 = {distance:.1f} px, 水平距离 = {h_distance:.1f} px, 垂直距离 = {v_distance:.1f} px")
            
            # 显示结果对话框
            self.showManualMeasurementResult(points, distance, h_distance, v_distance)
        except Exception as e:
            print(f"处理手动测量时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    @pyqtSlot(float)
    def update_diameter_from_oct(self, reconstructed_diameter):
        """接收OCT重建直径并更新主界面结果"""
        print(f"从OCT模块接收到直径: {reconstructed_diameter}")
        # 检查是否已有深度测量
        if hasattr(self, 'hole_depth') and self.hole_depth > 0:
            self.standard_diameter = reconstructed_diameter
            self.depth_diameter_ratio = self.hole_depth / self.standard_diameter

            # 更新UI标签，并明确指出数据来源
            self.standardDiameterLabel.setText(f"标准直径 (OCT): {self.standard_diameter:.2f} μm")
            self.standardDiameterLabel.setStyleSheet("font-size: 13px; font-weight: bold; color: #28a745;") # 使用绿色高亮
            self.ratioLabel.setText(f"深径比: {self.depth_diameter_ratio:.2f}")
            
            QMessageBox.information(self, "结果更新", f"已采用OCT重建直径 ({reconstructed_diameter:.2f} μm) 更新计算结果。")
        else:
            # 如果没有深度，只更新直径
            self.standard_diameter = reconstructed_diameter
            self.standardDiameterLabel.setText(f"标准直径 (OCT): {self.standard_diameter:.2f} μm")
            self.standardDiameterLabel.setStyleSheet("font-size: 13px; font-weight: bold; color: #28a745;")
            self.ratioLabel.setText("深径比: (待计算深度)")
            QMessageBox.warning(self, "部分更新", f"已获取OCT重建直径 ({reconstructed_diameter:.2f} μm)。\n请先对单张截面图进行处理以计算深度。")

    def set_reference_depth(self):
        """设置OCT模块可以使用的参考深度"""
        if hasattr(self, 'hole_depth') and self.hole_depth > 0:
            self.reference_depth_for_oct = self.hole_depth
            self.statusBar().showMessage(f"已设置参考深度: {self.reference_depth_for_oct:.2f} μm，可在OCT模块中使用。")
            QMessageBox.information(self, "参考深度已设置", f"已设置参考深度: {self.reference_depth_for_oct:.2f} μm")
        else:
            QMessageBox.warning(self, "设置失败", "没有有效的深度值可供设置。")

    def open_pixel_calibrator(self):
        """打开像素校准窗口"""
        # 创建并显示校准窗口，将主窗口实例传递给它
        self.calibration_window = PixelCalibrationApp(main_app=self)
        self.calibration_window.show()

    def update_conversion_factors(self, new_factor):
        """由校准窗口调用，以更新转换系数"""
        if new_factor > 0:
            self.pixelToUmXSpinBox.setValue(new_factor)
            # SpinBox的值变化会自动调用updatePixelToUmX，所以不需要手动更新params
            print(f"X方向转换系数已更新为: {new_factor}")

    def updateBinaryThreshold(self, value):
        self.params['binary_threshold'] = value
        self.binaryThresholdLabel.setText(str(value))
        if self.original_image is not None:
            self.processImage()
    
    def updateInvertBinary(self, state):
        self.params['invert_binary'] = state == Qt.Checked
        if self.original_image is not None:
            self.processImage()

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 左右箭头键切换图片
        if event.key() == Qt.Key_Left:
            self.loadPreviousImage()
        elif event.key() == Qt.Key_Right:
            self.loadNextImage()
        # 调用父类的事件处理
        super().keyPressEvent(event)

    def measureWithoutGap(self):
        """
        无缺口测量方法。
        该方法适用于顶部没有明显缺口的情况。
        它通过寻找底部的短横线来确定直径，并计算到顶部直线的距离作为深度。
        """
        if self.original_image is None or self.binary_image is None:
            QMessageBox.warning(self, "警告", "请先加载并处理图像")
            return

        try:
            result_img = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
            binary_img = self.binary_image

            # 1. 寻找顶部表面直线
            row_projection = np.sum(binary_img, axis=1)
            row_projection_smooth = np.convolve(row_projection, np.ones(5)/5, mode='same')
            threshold = np.max(row_projection_smooth) * (self.params['row_projection_threshold'] / 100.0)
            significant_rows = np.where(row_projection_smooth > threshold)[0]
            
            horizontal_lines = []
            current_group = []
            if len(significant_rows) > 0:
                for i in range(len(significant_rows)):
                    if not current_group or significant_rows[i] - current_group[-1] <= 5:
                        current_group.append(significant_rows[i])
                    else:
                        horizontal_lines.append(int(np.mean(current_group)))
                        current_group = [significant_rows[i]]
                if current_group:
                    horizontal_lines.append(int(np.mean(current_group)))
            
            if not horizontal_lines:
                QMessageBox.warning(self, "错误", "无法找到顶部表面。")
                return

            top_line_index = self.params.get('top_line_index', 0)
            if len(horizontal_lines) <= top_line_index:
                QMessageBox.warning(self, "错误", f"顶部线索引 ({top_line_index}) 超出范围，共找到 {len(horizontal_lines)} 条。请选择一个较小的值。")
                return

            top_surface_row = horizontal_lines[top_line_index]
            cv2.line(result_img, (0, top_surface_row), (result_img.shape[1], top_surface_row), (0, 255, 255), 2)

            # 2. 在图像下半部分寻找最显著的底部短横线
            search_start_row = top_surface_row + 50
            roi = binary_img[search_start_row:, :]
            
            # 使用形态学操作寻找水平线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.params['horizontal_kernel_size'], 1))
            opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_line = None
            max_line_y = 0

            for cnt in contours:
                if cv2.contourArea(cnt) > self.params['short_line_min_length']:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # 寻找最低的显著横线
                    if (y + search_start_row) > max_line_y:
                        max_line_y = y + search_start_row
                        best_line = (x, y + search_start_row, w, h)

            if best_line is None:
                QMessageBox.warning(self, "错误", "无法在图像下半部分找到底部短横线。")
                return

            bottom_x, bottom_y, bottom_w, bottom_h = best_line
            
            # 3. 计算直径和深度
            diameter = bottom_w * PIXEL_TO_UM_X
            depth = (bottom_y - top_surface_row) * PIXEL_TO_UM_Y

            # 4. 可视化结果
            # 绘制底部短横线（直径）
            cv2.rectangle(result_img, (bottom_x, bottom_y), (bottom_x + bottom_w, bottom_y + bottom_h), (0, 0, 255), 2)
            # 绘制深度测量线
            depth_line_x = bottom_x + bottom_w // 2
            cv2.line(result_img, (depth_line_x, top_surface_row), (depth_line_x, bottom_y), (255, 0, 0), 2)
            
            # 在图像上添加文本
            self.addChineseText(result_img, f"直径: {diameter:.2f} um", (bottom_x, bottom_y - 40), (0, 0, 255), fontSize=20)
            self.addChineseText(result_img, f"深度: {depth:.2f} um", (depth_line_x + 10, top_surface_row + (bottom_y - top_surface_row)//2), (255, 0, 0), fontSize=20)

            # 5. 更新UI
            self.result_image = result_img
            self.displayImage(self.result_image, self.resultImageLabel, "无缺口测量结果")
            self.diameterLabel.setText(f"直径: {diameter:.2f} μm")
            self.depthLabel.setText(f"深度: {depth:.2f} μm")
            
            # 清空其他标准测量标签
            self.upperDiameterLabel.setText("上方0.1mm处直径: -- μm")
            self.lowerDiameterLabel.setText("下方0.1mm处直径: -- μm")
            self.standardDiameterLabel.setText("标准直径: -- μm")
            self.ratioLabel.setText("深径比: --")
            
        except Exception as e:
            QMessageBox.critical(self, "测量错误", f"在无缺口测量过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def toggleNoGapMeasureMode(self):
        """切换无缺口测量模式"""
        self.is_no_gap_measure_active = not self.is_no_gap_measure_active

        if self.is_no_gap_measure_active:
            self.noGapMeasureAction.setText("退出无缺口测量")
            self.statusBar().showMessage("无缺口测量模式已激活。")
            
            # 确保其他模式被关闭
            if getattr(self, 'is_manual_measure_active', False):
                self.toggleManualMeasureMode()
            if getattr(self, 'is_cropping', False):
                self.toggleCropMode()
            
            self.manualMeasureAction.setEnabled(False)
            self.cropAction.setEnabled(False)
            
            # 立即运行一次测量
            if self.original_image is not None:
                self.processImage()
        else:
            self.noGapMeasureAction.setText("无缺口测量")
            self.statusBar().clearMessage()
            
            self.manualMeasureAction.setEnabled(True)
            self.cropAction.setEnabled(True)

            # 恢复到常规测量状态
            if self.original_image is not None:
                self.processImage()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HoleDetectionApp()
    window.show()
    sys.exit(app.exec_())
