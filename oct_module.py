import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
                           QLabel, QTextEdit, QFileDialog, QMessageBox, QAbstractItemView,
                           QInputDialog, QApplication, QListWidgetItem, QProgressDialog, QComboBox,
                           QSlider, QGridLayout, QDialogButtonBox, QGroupBox, QFrame, QCheckBox,
                           QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap
import cv2
import oct_utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

class FineTuneDialog(QDialog):
    """用于微调检测参数的对话框"""
    def __init__(self, parent, image, initial_params):
        super().__init__(parent)
        self.setWindowTitle("微调检测参数")
        self.image = image
        # 确保所有参数都存在
        self.params = self.get_default_params()
        self.params.update(initial_params)
        
        self.setup_ui()
        self.update_preview()

    def get_default_params(self):
        # 从主窗口获取默认参数，确保一致性
        if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent, 'params'):
             return self.parent().parent.params.copy()
        # 如果无法获取，则使用一组硬编码的默认值
        return {
            'gaussian_kernel': 5,
            'adaptive_block_size': 51,
            'adaptive_c': 5,
            'top_line_index': 1,
            'row_projection_threshold': 70,
            'gap_min_width': 50,
            'horizontal_kernel_size': 25,
            'column_projection_threshold': 30,
            'column_peak_window': 10,
            'bottom_enhance_contrast': 1.5,
            'bottom_search_range': 0.8,
            'bottom_line_index': 0,
            'short_line_min_length': 5,
            'short_line_min_white_ratio': 0.4,
            'short_line_max_white_ratio': 0.9,
            'invert_binary': False
        }

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # 左侧预览图像
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 400)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # 右侧参数
        params_group = QGroupBox("参数")
        params_layout = QGridLayout()
        
        # 预处理
        params_layout.addWidget(QLabel("<b>预处理</b>"), 0, 0, 1, 3)
        self.add_slider(params_layout, 1, "高斯核(奇数)", "gaussian_kernel", 1, 21, 2)
        self.add_slider(params_layout, 2, "二值化块大小(奇数)", "adaptive_block_size", 3, 101, 2)
        self.add_slider(params_layout, 3, "二值化常数C", "adaptive_c", 1, 30, 1)
        self.invert_checkbox = QCheckBox("反转二值图像")
        self.invert_checkbox.setChecked(self.params.get('invert_binary', False))
        self.invert_checkbox.stateChanged.connect(self.update_preview)
        params_layout.addWidget(self.invert_checkbox, 4, 0, 1, 3)

        # 顶部检测
        params_layout.addWidget(self.create_separator(), 5, 0, 1, 3)
        params_layout.addWidget(QLabel("<b>顶部检测</b>"), 6, 0, 1, 3)
        self.add_spinbox(params_layout, 7, "顶部线索引", "top_line_index", 0, 10)
        self.add_slider(params_layout, 8, "行投影阈值(%)", "row_projection_threshold", 10, 90, 1, is_percent=True)
        self.add_slider(params_layout, 9, "缺口最小宽度", "gap_min_width", 10, 200, 5)

        # 底部检测
        params_layout.addWidget(self.create_separator(), 10, 0, 1, 3)
        params_layout.addWidget(QLabel("<b>底部检测</b>"), 11, 0, 1, 3)
        self.add_spinbox(params_layout, 12, "底部线索引", "bottom_line_index", 0, 10)
        self.add_slider(params_layout, 13, "底部搜索范围(%)", "bottom_search_range", 10, 100, 5, is_percent=True, is_float=True)
        
        # 短横线过滤
        params_layout.addWidget(self.create_separator(), 14, 0, 1, 3)
        params_layout.addWidget(QLabel("<b>短横线过滤</b>"), 15, 0, 1, 3)
        self.add_slider(params_layout, 16, "短横线最小长度", "short_line_min_length", 1, 20, 1)
        self.add_slider(params_layout, 17, "白色比例下限(%)", "short_line_min_white_ratio", 10, 60, 1, is_percent=True, is_float=True)
        self.add_slider(params_layout, 18, "白色比例上限(%)", "short_line_max_white_ratio", 60, 100, 1, is_percent=True, is_float=True)
        
        params_layout.setRowStretch(19, 1)
        params_group.setLayout(params_layout)
        
        # 包含参数和按钮的垂直布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(params_group)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        right_layout.addWidget(button_box)

        main_layout.addLayout(right_layout)
        self.setMinimumWidth(800)

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def add_slider(self, layout, row, label, param_name, min_val, max_val, step, is_percent=False, is_float=False):
        layout.addWidget(QLabel(label), row, 0)
        slider = QSlider(Qt.Horizontal)
        
        val = self.params.get(param_name, (min_val + max_val) / 2)
        if is_float:
            slider.setRange(int(min_val), int(max_val))
            slider.setValue(int(val * 100))
        else:
            slider.setRange(min_val, max_val)
            slider.setValue(val)
        
        slider.setSingleStep(step)
        
        value_label = QLabel()
        self.update_label_text(value_label, slider.value(), is_percent, is_float)

        slider.valueChanged.connect(lambda v, name=param_name, lbl=value_label, p=is_percent, f=is_float: self.on_slider_change(v, name, lbl, p, f))
        
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)
        
        setattr(self, f"{param_name}_slider", slider)

    def add_spinbox(self, layout, row, label, param_name, min_val, max_val):
        layout.addWidget(QLabel(label), row, 0)
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(self.params.get(param_name, min_val))
        spinbox.valueChanged.connect(lambda v, name=param_name: self.on_spinbox_change(v, name))
        layout.addWidget(spinbox, row, 1, 1, 2)

    def on_slider_change(self, value, param_name, label, is_percent, is_float):
        if "kernel" in param_name or "block_size" in param_name:
            if value % 2 == 0:
                value += 1
                slider = getattr(self, f"{param_name}_slider")
                slider.setValue(value)
                return
        
        if is_float:
            self.params[param_name] = value / 100.0
        else:
            self.params[param_name] = value
            
        self.update_label_text(label, value, is_percent, is_float)
        self.update_preview()
        
    def on_spinbox_change(self, value, param_name):
        self.params[param_name] = value
        self.update_preview()
        
    def update_label_text(self, label, value, is_percent, is_float):
        text = ""
        if is_float:
            text = f"{value / 100.0:.2f}"
        else:
            text = str(value)
        
        if is_percent:
            text += "%"
            
        label.setText(text)

    def update_preview(self):
        # 从主界面的processImage复制逻辑来确保一致性
        self.params['invert_binary'] = self.invert_checkbox.isChecked()
        parent_app = self.parent().parent
        if not parent_app: return

        try:
            # 应用高斯滤波减少噪声
            gaussian_kernel_size = self.params['gaussian_kernel']
            blurred = cv2.GaussianBlur(self.image, (gaussian_kernel_size, gaussian_kernel_size), 0)
            
            # 应用自适应二值化
            adaptive_block_size = self.params['adaptive_block_size']
            adaptive_c = self.params['adaptive_c']
            binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
            
            # 全局OTSU二值化
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 合并两种二值化结果
            binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            binary_opened = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel)
            binary_final = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel)
            
            # 如果需要反转二值图像
            if self.params['invert_binary']:
                binary_final = 255 - binary_final
            
            binary_image = binary_final
            
            h, w = binary_image.shape[:2]
            q_img = QImage(binary_image.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.width(), self.preview_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"微调预览更新失败: {e}")


    def get_params(self):
        self.params['invert_binary'] = self.invert_checkbox.isChecked()
        return self.params

class OCTHoleReconstructionDialog(QDialog):
    """OCT圆孔重建对话框"""
    
    def __init__(self, parent=None, image_label_class=None, reference_depth=0.0):
        super().__init__(parent)
        self.parent = parent
        self.image_label_class = image_label_class
        self.reference_depth = reference_depth
        
        # 获取像素到微米的转换比例
        self.pixel_to_um_x = 1.0
        self.pixel_to_um_y = 1.0
        if hasattr(self.parent, 'PIXEL_TO_UM_X'):
            self.pixel_to_um_x = self.parent.PIXEL_TO_UM_X
        if hasattr(self.parent, 'PIXEL_TO_UM_Y'):
            self.pixel_to_um_y = self.parent.PIXEL_TO_UM_Y
        # 保留旧的转换比例以兼容原有代码
        self.pixel_to_um = self.pixel_to_um_x
        
        # 初始化变量
        self.oct_images = []
        self.oct_points_3d = None
        self.oct_plane_params = None
        self.oct_circle_center = None
        self.oct_radius = None
        self.oct_current_index = -1
        self.fit_method = 'algebraic'  # 默认使用代数拟合方法
        self.result_image_path = None  # 用于保存结果图片路径
        self.fine_tune_params = {} # 保存每张图的微调参数
        self.result_fig = None # 用于保存matplotlib的figure对象
        self.diameter_lines = [] # 保存绘图中的直径线对象
        
        # 设置中文字体
        self.font_prop = self.get_chinese_font()

        self.setup_ui()
    
    def get_chinese_font(self):
        """获取可用的中文字体"""
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        for font in fonts:
            if font in [f.name for f in fm.fontManager.ttflist]:
                return fm.FontProperties(family=font)
        return None # 如果没有找到，返回None

    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle("OCT圆孔重建")
        self.setMinimumSize(1200, 800)
        
        # 创建主布局
        layout = QVBoxLayout()
        
        # 创建说明标签
        infoLabel = QLabel("通过多张OCT图像扫描截面重建真实圆孔直径")
        infoLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(infoLabel)
        
        # 创建图像列表
        self.oct_image_list = QListWidget()
        self.oct_image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.oct_image_list.itemClicked.connect(self.show_selected_image)
        layout.addWidget(QLabel("OCT图像列表:"))
        layout.addWidget(self.oct_image_list)
        
        # 按钮组
        buttonLayout = QHBoxLayout()
        
        # 添加单个图像按钮
        addSingleButton = QPushButton("添加单个图像")
        addSingleButton.clicked.connect(self.add_single_oct_image)
        buttonLayout.addWidget(addSingleButton)

        # 批量添加图像按钮
        addBatchButton = QPushButton("批量添加图像")
        addBatchButton.clicked.connect(self.add_oct_images_batch)
        buttonLayout.addWidget(addBatchButton)
        
        # 删除图像按钮
        removeButton = QPushButton("删除图像")
        removeButton.clicked.connect(self.remove_oct_image)
        buttonLayout.addWidget(removeButton)

        # 添加自动检测按钮
        self.autoDetectButton = QPushButton("自动检测孔径")
        self.autoDetectButton.clicked.connect(self.handle_auto_detect)
        self.autoDetectButton.setEnabled(False) # 初始时禁用
        buttonLayout.addWidget(self.autoDetectButton)
        
        # 批量自动检测按钮
        batchDetectButton = QPushButton("批量自动检测")
        batchDetectButton.clicked.connect(self.batch_auto_detect)
        buttonLayout.addWidget(batchDetectButton)
        
        # 开始处理按钮
        processButton = QPushButton("开始重建")
        processButton.clicked.connect(self.process_oct_reconstruction)
        
        # 导出结果按钮
        self.exportButton = QPushButton("导出结果")
        self.exportButton.clicked.connect(self.export_results)
        self.exportButton.setEnabled(False)  # 初始时禁用

        # 关闭按钮
        closeButton = QPushButton("关闭")
        closeButton.clicked.connect(self.close)

        buttonLayout.addWidget(processButton)
        buttonLayout.addWidget(self.exportButton)
        buttonLayout.addWidget(closeButton)
        
        layout.addLayout(buttonLayout)
        
        # 拟合方法下拉框
        fitMethodLayout = QHBoxLayout()
        fitMethodLayout.addWidget(QLabel("拟合方法:"))
        self.fit_method_combo = QComboBox()
        self.fit_method_combo.addItems(["代数法", "几何法"])
        fitMethodLayout.addWidget(self.fit_method_combo)
        fitMethodLayout.addStretch()
        
        layout.addLayout(fitMethodLayout)
        
        # 对齐中点复选框
        alignLayout = QHBoxLayout()
        self.align_midpoints_checkbox = QCheckBox("重建前对齐中点")
        self.align_midpoints_checkbox.setToolTip("在拟合圆前，将所有截面直径的中点对齐，以校正水平偏移。")
        alignLayout.addWidget(self.align_midpoints_checkbox)
        
        # 添加上下0.1mm直径分析复选框
        self.analyze_upper_lower_checkbox = QCheckBox("分析上下0.1mm处直径并取平均值")
        self.analyze_upper_lower_checkbox.setToolTip("分析圆孔下方0.1mm和上方0.1mm的直径并取平均值")
        alignLayout.addWidget(self.analyze_upper_lower_checkbox)
        
        alignLayout.addStretch()
        layout.addLayout(alignLayout)
        
        # 创建图像显示区域
        imageLayout = QHBoxLayout()
        
        # 原始图像显示
        self.oct_image_label = self.image_label_class(self)
        self.oct_image_label.setText("请添加并选择OCT图像")
        self.oct_image_label.setFixedSize(400, 400)
        self.oct_image_label.setAlignment(Qt.AlignCenter)
        self.oct_image_label.setStyleSheet("border: 1px solid black;")
        imageLayout.addWidget(self.oct_image_label)
        
        # 结果显示区域 - 使用Matplotlib画布
        result_display_group = QGroupBox("重建结果俯视图 (可缩放平移)")
        self.result_display_layout = QVBoxLayout()
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.result_canvas = FigureCanvas(self.fig)
        
        # 添加导航工具栏
        self.toolbar = NavigationToolbar(self.result_canvas, self)
        
        self.result_display_layout.addWidget(self.toolbar)
        self.result_display_layout.addWidget(self.result_canvas)
        result_display_group.setLayout(self.result_display_layout)
        
        imageLayout.addWidget(result_display_group)
        
        layout.addLayout(imageLayout)
        
        # 添加结果显示文本区域
        self.oct_result_text = QTextEdit()
        self.oct_result_text.setReadOnly(True)
        self.oct_result_text.setMaximumHeight(120)
        self.oct_result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 8px;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 14px;
            }
        """)
        layout.addWidget(QLabel("重建结果:"))
        layout.addWidget(self.oct_result_text)
        
        # 显示像素转换系数
        self.pixelInfoLabel = QLabel(f"当前像素转换系数: X方向 {self.pixel_to_um_x:.2f} μm/px，Y方向 {self.pixel_to_um_y:.2f} μm/px")
        self.pixelInfoLabel.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.pixelInfoLabel)
        
        # 添加像素转换系数调整控件
        pixelConversionLayout = QHBoxLayout()
        
        # X方向转换系数控件
        pixelXLayout = QHBoxLayout()
        pixelXLayout.addWidget(QLabel("X方向转换系数:"))
        self.pixelToUmXSpinBox = QDoubleSpinBox()
        self.pixelToUmXSpinBox.setMinimum(0.1)
        self.pixelToUmXSpinBox.setMaximum(10.0)
        self.pixelToUmXSpinBox.setSingleStep(0.1)
        self.pixelToUmXSpinBox.setValue(self.pixel_to_um_x)
        self.pixelToUmXSpinBox.setSuffix(" μm/px")
        self.pixelToUmXSpinBox.valueChanged.connect(self.updatePixelToUmX)
        pixelXLayout.addWidget(self.pixelToUmXSpinBox)
        
        # Y方向转换系数控件
        pixelYLayout = QHBoxLayout()
        pixelYLayout.addWidget(QLabel("Y方向转换系数:"))
        self.pixelToUmYSpinBox = QDoubleSpinBox()
        self.pixelToUmYSpinBox.setMinimum(0.1)
        self.pixelToUmYSpinBox.setMaximum(10.0)
        self.pixelToUmYSpinBox.setSingleStep(0.1)
        self.pixelToUmYSpinBox.setValue(self.pixel_to_um_y)
        self.pixelToUmYSpinBox.setSuffix(" μm/px")
        self.pixelToUmYSpinBox.valueChanged.connect(self.updatePixelToUmY)
        pixelYLayout.addWidget(self.pixelToUmYSpinBox)
        
        # 将两个控件布局添加到主布局
        pixelConversionLayout.addLayout(pixelXLayout)
        pixelConversionLayout.addLayout(pixelYLayout)
        layout.addLayout(pixelConversionLayout)
        
        self.setLayout(layout)
    
    def add_single_oct_image(self):
        """添加单个OCT图像并手动设置扫描位置"""
        try:
            filePath, _ = QFileDialog.getOpenFileName(
                self, "选择OCT图像", "", "图像文件 (*.png *.jpg *.bmp *.tif *.tiff);;所有文件 (*)")
            
            if filePath:
                # 弹出对话框输入扫描位置（扫描时在Y方向上的实际位置，单位:微米）
                position, ok = QInputDialog.getDouble(
                    self, "输入扫描位置", "请输入此OCT图像在Y方向上的扫描位置(单位:微米):", 0, -10000, 10000, 2)
                
                if ok:
                    # 读取图像
                    img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        QMessageBox.warning(self, "错误", f"无法读取图像: {filePath}")
                        return
                    
                    # 添加到列表
                    self.oct_images.append({"path": filePath, "image": img, "position": position, "points": []})
                    
                    # 更新列表控件
                    item = QListWidgetItem(f"图像 {len(self.oct_images)}: Y={position}μm")
                    self.oct_image_list.addItem(item)
                    self.oct_image_list.setCurrentItem(item)
                    
                    # 显示图像
                    self.show_selected_image()
        except Exception as e:
            print(f"添加OCT图像时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"添加OCT图像时出错: {str(e)}")
    
    def add_oct_images_batch(self):
        """批量添加OCT图像并自动设置Y坐标"""
        try:
            # 弹出对话框输入Y轴间隔
            interval, ok1 = QInputDialog.getDouble(
                self, "输入扫描间隔", "请输入OCT图像之间的Y轴扫描间隔(单位:微米):", 100, 0, 10000, 2)
            
            if not ok1:
                return

            # 建议起始Y坐标
            suggested_start_y = 0.0
            if self.oct_images:
                last_position = self.oct_images[-1]["position"]
                suggested_start_y = last_position + interval

            # 弹出对话框输入起始Y坐标
            start_y, ok2 = QInputDialog.getDouble(
                self, "输入起始位置", f"请输入第一张图像的Y轴起始位置(单位:微米):", suggested_start_y, -100000, 100000, 2)

            if not ok2:
                return

            filePaths, _ = QFileDialog.getOpenFileNames(
                self, "选择OCT图像(将按文件名排序)", "", "图像文件 (*.png *.jpg *.bmp *.tif *.tiff);;所有文件 (*)")
            
            if filePaths:
                filePaths.sort() # 按文件名排序以确保顺序正确
                
                initial_image_count = len(self.oct_images)

                for i, filePath in enumerate(filePaths):
                    # 读取图像
                    img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        QMessageBox.warning(self, "错误", f"无法读取图像: {filePath}")
                        continue
                    
                    # 计算Y坐标
                    position = start_y + i * interval
                    
                    # 添加到列表
                    self.oct_images.append({"path": filePath, "image": img, "position": position, "points": []})
                    
                    # 更新列表控件
                    item_text = f"图像 {len(self.oct_images)}: Y={position:.2f}μm"
                    item = QListWidgetItem(item_text)
                    self.oct_image_list.addItem(item)
                
                if filePaths:
                    # 选中新添加的第一个图像
                    self.oct_image_list.setCurrentRow(initial_image_count)
                    self.show_selected_image()

        except Exception as e:
            print(f"批量添加OCT图像时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"批量添加OCT图像时出错: {str(e)}")
    
    def handle_auto_detect(self):
        """根据按钮状态处理自动检测或微调"""
        if self.oct_current_index < 0:
            return

        is_fine_tune = self.autoDetectButton.text() == "微调检测结果"
        self.auto_detect_hole_diameter(fine_tune=is_fine_tune)

    def auto_detect_hole_diameter(self, fine_tune=False):
        """使用父窗口的孔洞检测算法自动检测孔径"""
        try:
            if self.oct_current_index < 0 or self.oct_current_index >= len(self.oct_images):
                QMessageBox.warning(self, "警告", "请先选择一个OCT图像")
                return
            
            current_image_data = self.oct_images[self.oct_current_index]
            img = current_image_data["image"].copy()

            # 获取微调参数 - 如果没有，则从主窗口获取当前参数
            if self.oct_current_index in self.fine_tune_params:
                params = self.fine_tune_params[self.oct_current_index]
            elif self.parent and hasattr(self.parent, 'params'):
                params = self.parent.params.copy()
            else:
                 params = {
                    "gaussian_kernel": 5, "adaptive_block_size": 21, "adaptive_c": 5
                 }

            if fine_tune:
                dialog = FineTuneDialog(self, img, params)
                if dialog.exec_() == QDialog.Accepted:
                    params = dialog.get_params()
                    self.fine_tune_params[self.oct_current_index] = params
                else:
                    return # 用户取消

            # 使用父窗口的孔洞检测算法
            if self.parent and hasattr(self.parent, 'detect_hole_dimensions'):
                # 临时保存原始图像和二值图像
                original_img = None
                binary_img = None
                
                if hasattr(self.parent, 'original_image'):
                    original_img = self.parent.original_image
                
                if hasattr(self.parent, 'binary_image'):
                    binary_img = self.parent.binary_image
                
                # 设置当前图像
                self.parent.original_image = img
                
                # 使用与主窗口完全一致的图像处理流程
                # 这部分代码直接从主窗口的 processImage 复制和修改而来
                try:
                    # 应用高斯滤波减少噪声
                    gaussian_kernel_size = params['gaussian_kernel']
                    blurred = cv2.GaussianBlur(img, (gaussian_kernel_size, gaussian_kernel_size), 0)
                    
                    # 应用自适应二值化
                    adaptive_block_size = params['adaptive_block_size']
                    adaptive_c = params['adaptive_c']
                    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
                    
                    # 全局OTSU二值化
                    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 合并两种二值化结果
                    binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
                    
                    # 形态学操作
                    kernel = np.ones((3, 3), np.uint8)
                    binary_opened = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel)
                    binary_final = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel)
                    
                    # 如果需要反转二值图像
                    if params.get('invert_binary', False):
                        binary_final = 255 - binary_final
                    
                    self.parent.binary_image = binary_final

                    # 确保其他参数也从params字典传递
                    original_main_params = self.parent.params.copy()
                    self.parent.params.update(params)

                    # 调用孔洞检测方法
                    self.parent.detect_hole_dimensions()

                    # 恢复主窗口的原始参数
                    self.parent.params = original_main_params

                except Exception as e:
                    print(f"在OCT模块中调用主检测逻辑失败: {e}")
                    QMessageBox.warning(self, "检测失败", f"孔洞检测算法执行失败: {e}")
                    # 恢复父窗口状态
                    if original_img is not None: self.parent.original_image = original_img
                    if binary_img is not None: self.parent.binary_image = binary_img
                    return
                
                # 获取结果
                if hasattr(self.parent, 'hole_start') and hasattr(self.parent, 'hole_end') and hasattr(self.parent, 'upper_surface_row'):
                    # 获取检测到的点
                    p1 = (self.parent.hole_start, self.parent.upper_surface_row)
                    p2 = (self.parent.hole_end, self.parent.upper_surface_row)
                    
                    # 保存点坐标
                    self.oct_images[self.oct_current_index]["points"] = [p1, p2]
                    
                    # 计算直径
                    distance_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    position = self.oct_images[self.oct_current_index]["position"]
                    
                    # 计算微米单位直径
                    distance_um = distance_px * self.pixel_to_um_x
                    
                    # 更新列表项文本
                    self.oct_image_list.item(self.oct_current_index).setText(
                        f"图像 {self.oct_current_index+1}: Y={position}μm, 直径={distance_px:.1f}px ({distance_um:.1f}μm)")
                    
                    # 重新显示图像以更新点的显示
                    self.show_selected_image()
                    
                    QMessageBox.information(self, "检测成功", f"自动检测到孔径: {distance_px:.1f}像素 ({distance_um:.1f}μm)")
                else:
                    QMessageBox.warning(self, "检测失败", "无法自动检测孔径，请尝试手动标记")
                
                # 恢复原始图像和二值图像
                if original_img is not None:
                    self.parent.original_image = original_img
                
                if binary_img is not None:
                    self.parent.binary_image = binary_img
            else:
                QMessageBox.warning(self, "检测失败", "孔洞检测算法不可用，请手动标记孔径")
        except Exception as e:
            print(f"自动检测孔径时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"自动检测孔径时出错: {str(e)}")
    
    def remove_oct_image(self):
        """从列表中删除选中的OCT图像"""
        try:
            currentRow = self.oct_image_list.currentRow()
            if currentRow >= 0:
                self.oct_images.pop(currentRow)
                self.oct_image_list.takeItem(currentRow)
                
                # 更新列表显示
                for i in range(self.oct_image_list.count()):
                    self.oct_image_list.item(i).setText(f"图像 {i+1}: Y={self.oct_images[i]['position']}μm")
                
                if self.oct_image_list.count() > 0:
                    self.oct_image_list.setCurrentRow(0)
                    self.show_selected_image()
                else:
                    self.oct_image_label.clear()
                    self.oct_image_label.setText("请添加并选择OCT图像")
        except Exception as e:
            print(f"删除OCT图像时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"删除OCT图像时出错: {str(e)}")
    
    def show_selected_image(self):
        """显示选中的OCT图像及其标记点，并高亮重建图中的对应直线"""
        try:
            currentRow = self.oct_image_list.currentRow()
            if currentRow >= 0:
                self.autoDetectButton.setEnabled(True)
                # 检查是否已检测
                if len(self.oct_images[currentRow].get("points", [])) >= 2:
                    self.autoDetectButton.setText("微调检测结果")
                else:
                    self.autoDetectButton.setText("自动检测孔径")
            else:
                self.autoDetectButton.setEnabled(False)

            if currentRow >= 0 and currentRow < len(self.oct_images):
                img = self.oct_images[currentRow]["image"].copy()
                
                # 转换为彩色图像以便绘制彩色标记
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # 绘制已标记的点
                points = self.oct_images[currentRow].get("points", [])
                if len(points) >= 2:
                    p1, p2 = points[:2]
                    cv2.circle(img_color, (p1[0], p1[1]), 5, (0, 255, 0), -1)
                    cv2.circle(img_color, (p2[0], p2[1]), 5, (0, 255, 0), -1)
                    cv2.line(img_color, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 2)
                    
                    # 计算直径
                    distance_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    # 获取像素到微米的转换比例
                    pixel_to_um = 1.0
                    if hasattr(self.parent, 'PIXEL_TO_UM'):
                        pixel_to_um = self.parent.PIXEL_TO_UM
                    
                    distance_um = distance_pixels * pixel_to_um
                    
                    # 在图像上标注直径
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2 + 20
                    
                    # 添加文字背景
                    text = f"{distance_pixels:.1f} px ({distance_um:.1f} μm)"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_color, 
                                 (mid_x - text_w//2 - 5, mid_y - text_h - 5), 
                                 (mid_x + text_w//2 + 5, mid_y + 5), 
                                 (255, 255, 255), -1)
                    
                    # 添加文字
                    cv2.putText(img_color, text,
                              (mid_x - text_w//2, mid_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示图像
                h, w = img_color.shape[:2]
                bytes_per_line = 3 * w
                q_img = QImage(img_color.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                self.oct_image_label.setPixmap(pixmap)
                
                # 启用点选模式
                self.oct_image_label.setCropActive(False)
                self.oct_image_label.setManualMeasureActive(True)
                
                # 设置回调函数
                self.oct_image_label.manual_measure_callback = self.oct_point_selected
                
                # 存储当前索引
                self.oct_current_index = currentRow
                
        except Exception as e:
            print(f"显示OCT图像时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"显示OCT图像时出错: {str(e)}")
        
        # 在显示完图像后，高亮对应的线
        self.highlight_selected_line()

    def highlight_selected_line(self):
        """在重建图中高亮显示与所选图像对应的直径线"""
        current_row = self.oct_image_list.currentRow()
        if current_row < 0:
            return
            
        # 创建一个从原始图像索引到有效图像索引的映射
        # (只有包含2个点的图像才是有效的)
        valid_indices = [i for i, img in enumerate(self.oct_images) if len(img.get("points", [])) >= 2]
        valid_index_map = {original_idx: valid_idx for valid_idx, original_idx in enumerate(valid_indices)}

        # 获取当前选中行在有效索引中的位置
        highlight_idx = valid_index_map.get(current_row)

        for i, line in enumerate(self.diameter_lines):
            if i == highlight_idx:
                # 高亮选中的线
                line.set_color('cyan')
                line.set_linewidth(3)
                line.set_alpha(1.0)
                line.set_zorder(10) # 确保在最顶层
            else:
                # 恢复其他线的默认样式
                line.set_color('g')
                line.set_linewidth(1.5)
                line.set_alpha(0.7)
                line.set_zorder(3)
        
        # 刷新画布以显示更改
        if hasattr(self, 'result_canvas'):
            self.result_canvas.draw_idle()

    def oct_point_selected(self, points):
        """OCT图像上点击选择点的回调函数"""
        try:
            if hasattr(self, 'oct_current_index') and self.oct_current_index < len(self.oct_images):
                # 转换QPoint为普通坐标
                converted_points = [(p.x(), p.y()) for p in points]
                
                # 保存点坐标（最多保存两个点）
                self.oct_images[self.oct_current_index]["points"] = converted_points[:2]
                
                # 重新显示图像以更新点的显示
                self.show_selected_image()
                
                # 如果有两个点，自动更新列表项文本
                if len(converted_points) >= 2:
                    p1, p2 = converted_points[:2]
                    distance_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    position = self.oct_images[self.oct_current_index]["position"]
                    
                    # 计算微米单位直径
                    distance_um = distance_px * self.pixel_to_um_x
                    
                    self.oct_image_list.item(self.oct_current_index).setText(
                        f"图像 {self.oct_current_index+1}: Y={position}μm, 直径={distance_px:.1f}px ({distance_um:.1f}μm) (手动)")
                
        except Exception as e:
            print(f"OCT点选择时出错: {str(e)}")
    
    def process_oct_reconstruction(self):
        """处理OCT图像重建真实圆孔"""
        try:
            # 检查是否有足够的图像和点
            valid_images = [img for img in self.oct_images if len(img.get("points", [])) >= 2]
            
            if len(valid_images) < 3:
                QMessageBox.warning(self, "警告", "需要至少3张已标记直径点的OCT图像才能进行重建!")
                return
            
            # 准备数据
            points_2d = []
            scan_positions = []
            
            for img in valid_images:
                p1, p2 = img["points"][:2]  # 确保只取前两个点
                points_2d.append([p1[0], p1[1], p2[0], p2[1]])  # [x1, y1, x2, y2]
                scan_positions.append(img["position"])
            
            points_2d = np.array(points_2d)
            
            # 获取像素到微米的转换比例
            pixel_to_um = 1.0
            if hasattr(self.parent, 'PIXEL_TO_UM'):
                pixel_to_um = self.parent.PIXEL_TO_UM
            
            # 打印调试信息
            print(f"处理OCT重建，有效图像数: {len(valid_images)}")
            print(f"原始点数据形状: {points_2d.shape}")
            print(f"扫描位置: {scan_positions}")
            print(f"像素转换系数: X={self.pixel_to_um_x}, Y={self.pixel_to_um_y}")
            
            # 1. 转换为XY平面坐标（单位：微米）
            self.oct_points_xy = oct_utils.transform_to_2d_coords(points_2d, scan_positions, self.pixel_to_um_x)
            
            # (可选) 对齐中点
            points_for_fitting = self.oct_points_xy
            if self.align_midpoints_checkbox.isChecked():
                aligned_points = points_for_fitting.copy()
                num_slices = len(valid_images)
                if num_slices > 0:
                    # 使用第一个切片的中点作为参考
                    ref_mid_x = (aligned_points[0, 0] + aligned_points[1, 0]) / 2.0
                    for i in range(num_slices):
                        p1_idx, p2_idx = i * 2, i * 2 + 1
                        current_mid_x = (aligned_points[p1_idx, 0] + aligned_points[p2_idx, 0]) / 2.0
                        shift = ref_mid_x - current_mid_x
                        # 应用平移
                        aligned_points[p1_idx, 0] += shift
                        aligned_points[p2_idx, 0] += shift
                    points_for_fitting = aligned_points
                    print("已执行中点对齐预处理。")

            # 2. 在XY平面上直接拟合圆
            # 根据选择的拟合方法进行圆拟合
            fit_method_text = self.fit_method_combo.currentText()
            fit_method = 'geometric' if fit_method_text == '几何法' else 'algebraic'
            
            if fit_method == 'geometric':
                center_x, center_y, radius = oct_utils.fit_circle_geometric(points_for_fitting)
            else:
                center_x, center_y, radius = oct_utils.fit_circle_2d(points_for_fitting)
                
            self.oct_circle_center = np.array([center_x, center_y])
            self.oct_radius = radius
            self.fit_method = fit_method
            
            # 3. 计算直径并显示结果 - 注意：半径和直径已经是微米单位
            diameter_um = 2 * radius
            
            # === 新增：计算上下0.1mm处的直径 ===
            upper_lower_result_html = ""
            if self.analyze_upper_lower_checkbox.isChecked() and hasattr(self.parent, 'upper_diameter_at_01mm') and hasattr(self.parent, 'lower_diameter_at_01mm'):
                # 获取主程序中的上下0.1mm直径数据
                upper_diameter = self.parent.upper_diameter_at_01mm
                lower_diameter = self.parent.lower_diameter_at_01mm
                
                # 确保有有效数据
                if upper_diameter > 0 and lower_diameter > 0:
                    # 计算平均值
                    avg_diameter = (upper_diameter + lower_diameter) / 2
                    diameter_um = avg_diameter  # 使用平均值替换原始直径
                    
                    # 添加到结果显示
                    upper_lower_result_html = f"""
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>上方0.1mm处直径:</td>
                        <td style='padding: 4px;'>{upper_diameter:.2f} μm</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>下方0.1mm处直径:</td>
                        <td style='padding: 4px;'>{lower_diameter:.2f} μm</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>上下0.1mm平均直径:</td>
                        <td style='padding: 4px;'><span style='color: #9c27b0; font-weight: bold;'>{avg_diameter:.2f} μm</span></td>
                    </tr>
                    """
                    print(f"已使用上下0.1mm处的平均直径: {avg_diameter:.2f}μm")
                else:
                    print(f"上下0.1mm处的直径数据无效: 上={upper_diameter:.2f}μm, 下={lower_diameter:.2f}μm")
                    
            # 如果有参考深度，计算深径比
            depth_ratio_html = ""
            if self.reference_depth > 0 and diameter_um > 0:
                ratio = self.reference_depth / diameter_um
                depth_ratio_html = f"""
                <tr>
                    <td style='font-weight: bold; padding: 4px;'>参考深度:</td>
                    <td style='padding: 4px;'>{self.reference_depth:.2f} μm</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 4px;'>深径比:</td>
                    <td style='padding: 4px;'><span style='color: #28a745; font-weight: bold; font-size: 16px;'>{ratio:.2f}</span></td>
                </tr>
                """

            # 美化结果显示
            result_html = f"""
            <div style='font-family: "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 14px;'>
                <table width='100%'>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>拟合方法:</td>
                        <td style='padding: 4px;'>{fit_method_text}</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>圆心坐标 (X,Y):</td>
                        <td style='padding: 4px;'>({self.oct_circle_center[0]:.2f}, {self.oct_circle_center[1]:.2f}) μm</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>X方向转换系数:</td>
                        <td style='padding: 4px;'>{self.pixel_to_um_x:.2f} μm/px</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>Y方向转换系数:</td>
                        <td style='padding: 4px;'>{self.pixel_to_um_y:.2f} μm/px</td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>半径:</td>
                        <td style='padding: 4px;'><span style='color: #007bff; font-weight: bold;'>{radius:.2f} μm</span></td>
                    </tr>
                    <tr>
                        <td style='font-weight: bold; padding: 4px;'>真实直径:</td>
                        <td style='padding: 4px;'><span style='color: #dc3545; font-weight: bold; font-size: 16px;'>{diameter_um:.2f} μm</span></td>
                    </tr>
                    {upper_lower_result_html}
                    {depth_ratio_html}
                </table>
            </div>
            """
            self.oct_result_text.setHtml(result_html)
            self.exportButton.setEnabled(True)  # 启用导出按钮
            
            # 生成2D可视化结果
            self.visualize_oct_results(points_to_plot=points_for_fitting)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"OCT重建处理时出错: {str(e)}")
            QMessageBox.warning(self, "错误", f"OCT重建处理时出错: {str(e)}")
    
    def visualize_oct_results(self, points_to_plot, pixel_to_um=1.0):
        """生成可视化结果并显示"""
        try:
            # 获取像素到微米的转换比例
            if hasattr(self.parent, 'PIXEL_TO_UM'):
                pixel_to_um = self.parent.PIXEL_TO_UM
            
            # 打印调试信息
            print(f"可视化OCT结果: 待绘制点数={len(points_to_plot) / 2}")
            print(f"OCT点数据形状: {points_to_plot.shape}")
            
            # 使用工具函数生成简化版可视化结果
            self.result_fig = oct_utils.create_simple_visualization(
                points_to_plot, 
                self.oct_circle_center,
                self.oct_radius,
                pixel_to_um=self.pixel_to_um_x,
                fit_method=self.fit_method,
                font_prop=self.font_prop # 传递字体属性
            )
            
            # 替换旧的画布和工具栏
            if hasattr(self, 'toolbar'):
                self.result_display_layout.removeWidget(self.toolbar)
                self.toolbar.deleteLater()
            if hasattr(self, 'result_canvas'):
                self.result_display_layout.removeWidget(self.result_canvas)
                self.result_canvas.deleteLater()

            # 2. 创建新的控件
            self.result_canvas = FigureCanvas(self.result_fig)
            self.toolbar = NavigationToolbar(self.result_canvas, self)

            # 3. 将新控件添加到布局中
            self.result_display_layout.addWidget(self.toolbar)
            self.result_display_layout.addWidget(self.result_canvas)
            
            # 存储直径线对象以便高亮
            if self.result_fig.get_axes():
                ax = self.result_fig.get_axes()[0]
                all_lines = ax.get_lines()
                # 第一个 line 是拟合圆，其余的是直径线
                self.diameter_lines = all_lines[1:]
            else:
                self.diameter_lines = []
            
            # 初始高亮当前选中的行
            self.highlight_selected_line()
            
        except Exception as e:
            print(f"可视化OCT结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"可视化OCT结果时出错: {str(e)}")
    
    def export_results(self):
        """导出重建结果"""
        try:
            if not self.result_fig:
                QMessageBox.warning(self, "警告", "没有可导出的结果。请先进行重建。")
                return

            # 获取默认文件名
            if not self.oct_images:
                base_name = "oct_reconstruction"
            else:
                base_name = os.path.splitext(os.path.basename(self.oct_images[0]["path"]))[0]
            
            default_filename = os.path.join(os.getcwd(), f"{base_name}_reconstruction_result.zip")

            filePath, _ = QFileDialog.getSaveFileName(
                self, "导出结果", default_filename, "ZIP压缩文件 (*.zip)")

            if filePath:
                import zipfile
                import tempfile
                
                # 获取HTML结果(保留格式)
                html_result = self.oct_result_text.toHtml()
                
                # 获取文本结果
                text_result = self.oct_result_text.toPlainText()
                
                # 如果使用了上下0.1mm的直径分析，添加额外的CSV数据
                csv_data = ""
                if self.analyze_upper_lower_checkbox.isChecked() and hasattr(self.parent, 'upper_diameter_at_01mm') and hasattr(self.parent, 'lower_diameter_at_01mm'):
                    upper_diameter = self.parent.upper_diameter_at_01mm
                    lower_diameter = self.parent.lower_diameter_at_01mm
                    if upper_diameter > 0 and lower_diameter > 0:
                        avg_diameter = (upper_diameter + lower_diameter) / 2
                        csv_data = f"""参数,数值,单位
圆孔半径,{self.oct_radius:.2f},μm
常规重建直径,{2 * self.oct_radius:.2f},μm
上方0.1mm处直径,{upper_diameter:.2f},μm
下方0.1mm处直径,{lower_diameter:.2f},μm
上下0.1mm平均直径,{avg_diameter:.2f},μm
"""
                    if self.reference_depth > 0:
                        ratio = self.reference_depth / avg_diameter
                        csv_data += f"参考深度,{self.reference_depth:.2f},μm\n"
                        csv_data += f"深径比,{ratio:.2f},"

                with zipfile.ZipFile(filePath, 'w') as zipf:
                    # 保存文本结果
                    zipf.writestr("result_summary.txt", text_result.encode('utf-8'))
                    
                    # 保存HTML结果
                    zipf.writestr("result_summary.html", html_result.encode('utf-8'))
                    
                    # 如果有CSV数据，保存CSV结果
                    if csv_data:
                        zipf.writestr("measurement_data.csv", csv_data.encode('utf-8'))
                    else:
                        # 创建基本的CSV数据
                        csv_data = f"""参数,数值,单位
圆孔半径,{self.oct_radius:.2f},μm
圆孔直径,{2 * self.oct_radius:.2f},μm
X方向转换系数,{self.pixel_to_um_x:.2f},μm/px
Y方向转换系数,{self.pixel_to_um_y:.2f},μm/px
"""
                        if self.reference_depth > 0:
                            ratio = self.reference_depth / (2 * self.oct_radius)
                            csv_data += f"参考深度,{self.reference_depth:.2f},μm\n"
                            csv_data += f"深径比,{ratio:.2f},"
                        zipf.writestr("measurement_data.csv", csv_data.encode('utf-8'))
                    
                    # 保存可视化图片
                    temp_img_path = os.path.join(tempfile.gettempdir(), "temp_reconstruction.png")
                    self.result_fig.savefig(temp_img_path, dpi=300)
                    zipf.write(temp_img_path, os.path.basename("reconstruction_view.png"))
                    
                    # 保存原始OCT图像
                    oct_image_folder = "source_oct_images"
                    for i, img_data in enumerate(self.oct_images):
                        if os.path.exists(img_data["path"]):
                            arcname = f"{oct_image_folder}/image_{i+1}_{os.path.basename(img_data['path'])}"
                            zipf.write(img_data["path"], arcname)

                QMessageBox.information(self, "成功", f"结果已成功导出到:\n{filePath}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"导出结果时出错: {str(e)}")
    
    def batch_auto_detect(self):
        """批量自动检测所有OCT图像的孔径"""
        try:
            if len(self.oct_images) == 0:
                QMessageBox.warning(self, "警告", "请先添加OCT图像")
                return
            
            # 创建进度对话框
            progress = QProgressDialog("正在批量检测孔径...", "取消", 0, len(self.oct_images), self)
            progress.setWindowTitle("批量处理")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 临时保存父窗口的原始图像和二值图像
            original_img = None
            binary_img = None
            
            if self.parent:
                if hasattr(self.parent, 'original_image'):
                    original_img = self.parent.original_image
                if hasattr(self.parent, 'binary_image'):
                    binary_img = self.parent.binary_image
            
            # 获取像素到微米的转换比例
            pixel_to_um = 1.0
            if hasattr(self.parent, 'PIXEL_TO_UM'):
                pixel_to_um = self.parent.PIXEL_TO_UM
            
            success_count = 0
            fail_count = 0
            
            # 处理每个图像
            for i, img_data in enumerate(self.oct_images):
                progress.setValue(i)
                QApplication.processEvents()
                
                if progress.wasCanceled():
                    break
                
                # 获取图像
                img = img_data["image"].copy()
                
                # 使用父窗口的孔洞检测算法
                if self.parent and hasattr(self.parent, 'detect_hole_dimensions'):
                    # 设置当前图像
                    self.parent.original_image = img
                    
                    # 进行预处理以生成二值图像
                    # 使用高斯模糊减少噪声
                    blurred = cv2.GaussianBlur(img, (5, 5), 0)
                    
                    # 自适应阈值分割
                    binary = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 21, 5)
                    
                    # 设置二值图像
                    self.parent.binary_image = binary
                    
                    # 确保diameter_history属性存在
                    if not hasattr(self.parent, 'diameter_history'):
                        self.parent.diameter_history = []
                    
                    if not hasattr(self.parent, 'measurement_count'):
                        self.parent.measurement_count = 0
                    
                    try:
                        # 调用孔洞检测方法
                        self.parent.processImage()  # 先进行图像处理
                        self.parent.detect_hole_dimensions()  # 然后检测孔洞
                        
                        # 获取结果
                        if hasattr(self.parent, 'hole_start') and hasattr(self.parent, 'hole_end') and hasattr(self.parent, 'upper_surface_row'):
                            # 获取检测到的点
                            p1 = (self.parent.hole_start, self.parent.upper_surface_row)
                            p2 = (self.parent.hole_end, self.parent.upper_surface_row)
                            
                            # 保存点坐标
                            self.oct_images[i]["points"] = [p1, p2]
                            
                            # 计算直径
                            distance_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            position = self.oct_images[i]["position"]
                            
                            # 计算微米单位直径
                            distance_um = distance_px * self.pixel_to_um_x
                            
                            # 更新列表项文本
                            self.oct_image_list.item(i).setText(
                                f"图像 {i+1}: Y={position}μm, 直径={distance_px:.1f}px ({distance_um:.1f}μm)")
                            
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        print(f"图像 {i+1} 自动检测失败: {str(e)}")
                        fail_count += 1
            
            # 恢复父窗口原始状态
            if self.parent:
                if original_img is not None:
                    self.parent.original_image = original_img
                if binary_img is not None:
                    self.parent.binary_image = binary_img
            
            # 完成进度
            progress.setValue(len(self.oct_images))
            
            # 显示结果
            QMessageBox.information(
                self, 
                "批量检测完成", 
                f"批量检测完成。\n成功: {success_count} 张图像\n失败: {fail_count} 张图像"
            )
            
            # 如果当前有选择的图像，刷新显示
            if hasattr(self, 'oct_current_index') and self.oct_current_index >= 0:
                self.show_selected_image()
                
        except Exception as e:
            print(f"批量自动检测时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"批量自动检测时出错: {str(e)}") 
    
    def updatePixelToUmX(self, value):
        """更新X方向的像素转换系数"""
        self.pixel_to_um_x = value
        self.pixel_to_um = value  # 兼容旧代码
        self.update_preview()
    
    def updatePixelToUmY(self, value):
        """更新Y方向的像素转换系数"""
        self.pixel_to_um_y = value
        self.update_preview()
        
    def update_preview(self):
        """更新预览和显示信息"""
        # 更新像素转换系数信息标签
        if hasattr(self, 'pixelInfoLabel'):
            self.pixelInfoLabel.setText(f"当前像素转换系数: X方向 {self.pixel_to_um_x:.2f} μm/px，Y方向 {self.pixel_to_um_y:.2f} μm/px")
        
        # 如果已经有重建结果，刷新显示
        if hasattr(self, 'oct_points_xy') and hasattr(self, 'oct_circle_center') and hasattr(self, 'oct_radius'):
            # 重新生成可视化结果
            self.visualize_oct_results(points_to_plot=self.oct_points_xy) 