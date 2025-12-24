import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QWidget, QMessageBox, QGroupBox,
                            QGridLayout, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal

# 忽略弃用警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ImageDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc;")
        self.setText("请加载图像")
        self.zoom_factor = 1.0
        
    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.updateDisplay()
        
    def updateDisplay(self):
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            if self.zoom_factor != 1.0:
                scaled_pixmap = self.original_pixmap.scaled(
                    int(self.original_pixmap.width() * self.zoom_factor), 
                    int(self.original_pixmap.height() * self.zoom_factor),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = self.original_pixmap.scaled(
                    self.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            super().setPixmap(scaled_pixmap)
        
    def setZoomFactor(self, factor):
        self.zoom_factor = factor
        self.updateDisplay()
        
    def resizeEvent(self, event):
        self.updateDisplay()
        super().resizeEvent(event)

class SelectableImageDisplay(ImageDisplay):
    selectionUpdated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rectangles = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.selection_active = False
        self.max_rectangles = 2
        self.setMouseTracking(True)
        
    def enableSelection(self, enable=True):
        self.selection_active = enable
        if not enable:
            self.drawing = False
            self.start_point = None
            self.end_point = None
        self.setCursor(Qt.CrossCursor if enable else Qt.ArrowCursor)
        
    def clearSelection(self):
        self.rectangles = []
        self.start_point = None
        self.end_point = None
        self.update()
        
    def getImageCoordinates(self, pos):
        if not hasattr(self, 'original_pixmap') or not self.original_pixmap:
            return None
        
        scaled_size = self.original_pixmap.size()
        if self.zoom_factor != 1.0:
            scaled_size = scaled_size * self.zoom_factor
        else:
            scaled_size = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio).size()
            
        img_x_offset = (self.width() - scaled_size.width()) / 2
        img_y_offset = (self.height() - scaled_size.height()) / 2
        
        if (pos.x() < img_x_offset or pos.x() >= img_x_offset + scaled_size.width() or
            pos.y() < img_y_offset or pos.y() >= img_y_offset + scaled_size.height()):
            return None
            
        img_x = int((pos.x() - img_x_offset) * self.original_pixmap.width() / scaled_size.width())
        img_y = int((pos.y() - img_y_offset) * self.original_pixmap.height() / scaled_size.height())
        
        return QPoint(img_x, img_y)
    
    def mousePressEvent(self, event):
        if self.selection_active and event.button() == Qt.LeftButton:
            img_pos = self.getImageCoordinates(event.pos())
            if img_pos:
                if len(self.rectangles) >= self.max_rectangles:
                    self.rectangles.pop(0)
                self.drawing = True
                self.start_point = img_pos
                self.end_point = img_pos
                self.update()
                
    def mouseMoveEvent(self, event):
        if self.drawing and self.selection_active:
            img_pos = self.getImageCoordinates(event.pos())
            if img_pos:
                self.end_point = img_pos
                self.update()
                
    def mouseReleaseEvent(self, event):
        if self.drawing and self.selection_active and event.button() == Qt.LeftButton:
            self.drawing = False
            img_pos = self.getImageCoordinates(event.pos())
            if img_pos and self.start_point:
                x1 = min(self.start_point.x(), self.end_point.x())
                y1 = min(self.start_point.y(), self.end_point.y())
                x2 = max(self.start_point.x(), self.end_point.x())
                y2 = max(self.start_point.y(), self.end_point.y())
                self.rectangles.append((x1, y1, x2, y2))
                self.update()
                self.selectionUpdated.emit()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not hasattr(self, 'original_pixmap') or not self.original_pixmap:
            return
            
        painter = QPainter(self)
        scaled_size = self.original_pixmap.size()
        if self.zoom_factor != 1.0:
            scaled_size *= self.zoom_factor
        else:
            scaled_size = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio).size()
            
        img_x_offset = (self.width() - scaled_size.width()) / 2
        img_y_offset = (self.height() - scaled_size.height()) / 2
        
        scale_x = scaled_size.width() / self.original_pixmap.width()
        scale_y = scaled_size.height() / self.original_pixmap.height()
        
        colors = [QColor(255, 0, 0, 128), QColor(0, 255, 0, 128)]
        for i, rect_coords in enumerate(self.rectangles):
            x1, y1, x2, y2 = rect_coords
            win_x1 = img_x_offset + x1 * scale_x
            win_y1 = img_y_offset + y1 * scale_y
            rect_width = (x2 - x1) * scale_x
            rect_height = (y2 - y1) * scale_y
            
            painter.setPen(QPen(colors[i % len(colors)], 2))
            painter.setBrush(colors[i % len(colors)])
            painter.drawRect(QRect(int(win_x1), int(win_y1), int(rect_width), int(rect_height)))
            
        if self.drawing and self.start_point and self.end_point:
            x1 = self.start_point.x() * scale_x + img_x_offset
            y1 = self.start_point.y() * scale_y + img_y_offset
            x2 = self.end_point.x() * scale_x + img_x_offset
            y2 = self.end_point.y() * scale_y + img_y_offset
            painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))))

class PixelCalibrationApp(QMainWindow):
    def __init__(self, main_app=None):
        super().__init__()
        self.main_app = main_app
        self.image = None
        self.binary_image = None
        self.processed_image = None
        self.conversion_factor = 0.0
        self.known_distance_um = 1000.0
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('像素校准工具')
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # 图像显示
        image_layout = QHBoxLayout()
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        self.original_image_label = ImageDisplay()
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        
        binary_group = QGroupBox("二值化图像")
        binary_layout = QVBoxLayout()
        self.binary_image_label = SelectableImageDisplay()
        self.binary_image_label.selectionUpdated.connect(self.onSelectionUpdated)
        binary_layout.addWidget(self.binary_image_label)
        binary_group.setLayout(binary_layout)
        
        processed_group = QGroupBox("结果图像")
        processed_layout = QVBoxLayout()
        self.processed_image_label = ImageDisplay()
        processed_layout.addWidget(self.processed_image_label)
        processed_group.setLayout(processed_layout)
        
        image_layout.addWidget(original_group)
        image_layout.addWidget(binary_group)
        image_layout.addWidget(processed_group)
        main_layout.addLayout(image_layout)
        
        # 缩放控制
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("缩放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.updateZoom)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_value = QSpinBox()
        self.zoom_value.setRange(10, 500)
        self.zoom_value.setValue(100)
        self.zoom_value.setSuffix("%")
        self.zoom_value.valueChanged.connect(self.zoom_slider.setValue)
        zoom_layout.addWidget(self.zoom_value)
        main_layout.addLayout(zoom_layout)

        # 主控制区
        main_control_group = QGroupBox("控制面板")
        main_control_layout = QHBoxLayout(main_control_group)
        
        # 左侧通用操作
        left_panel = QVBoxLayout()
        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        
        self.apply_button = QPushButton("应用到主窗口")
        self.apply_button.clicked.connect(self.apply_to_main_window)
        self.apply_button.setEnabled(False)
        
        left_panel.addWidget(self.load_button)
        left_panel.addWidget(self.save_button)
        left_panel.addWidget(self.apply_button)
        left_panel.addStretch(1)
        main_control_layout.addLayout(left_panel)
        
        # 中间参数设置
        params_group = QGroupBox("通用参数")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("二值化阈值:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.updateBinaryImage)
        params_layout.addWidget(self.threshold_slider, 0, 1)
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(128)
        self.threshold_value.valueChanged.connect(self.threshold_slider.setValue)
        params_layout.addWidget(self.threshold_value, 0, 2)
        
        params_layout.addWidget(QLabel("已知距离(微米):"), 1, 0)
        self.known_distance = QDoubleSpinBox()
        self.known_distance.setRange(1, 100000)
        self.known_distance.setValue(1000)
        self.known_distance.setSingleStep(100)
        params_layout.addWidget(self.known_distance, 1, 1, 1, 2)
        
        self.invert_binary = QCheckBox("反转二值化")
        self.invert_binary.stateChanged.connect(self.updateBinaryImage)
        params_layout.addWidget(self.invert_binary, 2, 0)
        main_control_layout.addWidget(params_group)

        # 右侧检测模式
        self.tabs = QTabWidget()
        
        # 自动检测Tab
        auto_tab = QWidget()
        auto_layout = QGridLayout(auto_tab)
        auto_layout.addWidget(QLabel("最小轮廓面积:"), 0, 0)
        self.min_contour_area = QSpinBox()
        self.min_contour_area.setRange(100, 100000)
        self.min_contour_area.setValue(5000)
        auto_layout.addWidget(self.min_contour_area, 0, 1)
        auto_layout.addWidget(QLabel("最小凹陷深度:"), 1, 0)
        self.min_defect_depth = QSpinBox()
        self.min_defect_depth.setRange(1, 100)
        self.min_defect_depth.setValue(5)
        auto_layout.addWidget(self.min_defect_depth, 1, 1)
        self.auto_process_button = QPushButton("自动检测凹陷")
        self.auto_process_button.clicked.connect(self.process_image_auto)
        self.auto_process_button.setEnabled(False)
        auto_layout.addWidget(self.auto_process_button, 2, 0, 1, 2)
        self.tabs.addTab(auto_tab, "自动检测")
        
        # 手动检测Tab
        manual_tab = QWidget()
        manual_layout = QGridLayout(manual_tab)
        self.selection_button = QPushButton("启用/禁用框选")
        self.selection_button.setCheckable(True)
        self.selection_button.toggled.connect(self.toggleSelection)
        self.selection_button.setEnabled(False)
        manual_layout.addWidget(self.selection_button, 0, 0)
        self.clear_selection_button = QPushButton("清除框选")
        self.clear_selection_button.clicked.connect(self.clearSelection)
        self.clear_selection_button.setEnabled(False)
        manual_layout.addWidget(self.clear_selection_button, 0, 1)
        self.manual_calculate_button = QPushButton("计算手动距离")
        self.manual_calculate_button.clicked.connect(self.calculateManualDistance)
        self.manual_calculate_button.setEnabled(False)
        manual_layout.addWidget(self.manual_calculate_button, 1, 0, 1, 2)
        self.tabs.addTab(manual_tab, "手动框选")
        
        main_control_layout.addWidget(self.tabs)
        
        # 结果显示
        result_group = QGroupBox("结果")
        result_layout = QGridLayout(result_group)
        result_layout.addWidget(QLabel("像素距离:"), 0, 0)
        self.pixel_distance_label = QLabel("0 像素")
        result_layout.addWidget(self.pixel_distance_label, 0, 1)
        result_layout.addWidget(QLabel("转换比例:"), 1, 0)
        self.conversion_label = QLabel("0 微米/像素")
        result_layout.addWidget(self.conversion_label, 1, 1)
        main_control_layout.addWidget(result_group)
        
        main_layout.addWidget(main_control_group)
        self.statusBar().showMessage('准备就绪')
        
    def updateZoom(self, value):
        self.zoom_value.setValue(value)
        factor = value / 100.0
        self.original_image_label.setZoomFactor(factor)
        self.binary_image_label.setZoomFactor(factor)
        self.processed_image_label.setZoomFactor(factor)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.bmp *.tif)")
        if file_path:
            self.image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                QMessageBox.critical(self, "错误", f"无法加载图像: {file_path}")
                return
                
            self.original_image_label.setPixmap(QPixmap(file_path))
            self.updateBinaryImage()
            
            self.auto_process_button.setEnabled(True)
            self.selection_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.statusBar().showMessage(f'已加载图像: {file_path}')
            
    def updateBinaryImage(self, value=None):
        if self.image is None:
            return
        
        threshold = self.threshold_slider.value()
        invert = self.invert_binary.isChecked()
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(self.image, threshold, 255, thresh_type)
        
        self.binary_image = binary
        height, width = binary.shape
        q_image = QImage(binary.data, width, height, width, QImage.Format_Grayscale8)
        self.binary_image_label.setPixmap(QPixmap.fromImage(q_image))
        self.binary_image_label.clearSelection()

    def toggleSelection(self, checked):
        self.binary_image_label.enableSelection(checked)
        self.clear_selection_button.setEnabled(checked)
        if not checked:
            self.manual_calculate_button.setEnabled(False)

    def clearSelection(self):
        self.binary_image_label.clearSelection()
        self.manual_calculate_button.setEnabled(False)

    def onSelectionUpdated(self):
        if len(self.binary_image_label.rectangles) == 2:
            self.manual_calculate_button.setEnabled(True)
        else:
            self.manual_calculate_button.setEnabled(False)

    def calculateManualDistance(self):
        if len(self.binary_image_label.rectangles) < 2:
            return
        
        rect1, rect2 = self.binary_image_label.rectangles
        center1_x = (rect1[0] + rect1[2]) / 2
        center2_x = (rect2[0] + rect2[2]) / 2
        
        horizontal_distance = abs(center2_x - center1_x)
        self.updateResultsAndDisplay(horizontal_distance, rects=[rect1, rect2])

    def process_image_auto(self):
        if self.binary_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        min_area = self.min_contour_area.value()
        min_depth = self.min_defect_depth.value()
        
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            QMessageBox.warning(self, "检测失败", "未找到任何轮廓。")
            return

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = next((cnt for cnt in contours if cv2.contourArea(cnt) > min_area), None)
        
        if main_contour is None:
            QMessageBox.warning(self, "检测失败", f"未找到面积大于 {min_area} 的轮廓。")
            return

        try:
            hull_indices = cv2.convexHull(main_contour, returnPoints=False)
            defects = cv2.convexityDefects(main_contour, hull_indices)
            if defects is None: raise cv2.error
        except cv2.error:
            QMessageBox.warning(self, "检测失败", "无法计算凹陷或未找到凹陷点。")
            return

        M = cv2.moments(main_contour)
        centroid_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        dip_points = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far_point = tuple(main_contour[f][0])
            depth = d / 256.0
            if depth > min_depth and far_point[1] < centroid_y:
                dip_points.append({'point': far_point, 'depth': depth})
        
        if len(dip_points) < 2:
            QMessageBox.warning(self, "检测失败", f"只找到 {len(dip_points)} 个符合条件的凹陷点。")
        else:
            # 修复: 先对字典列表排序，再提取点
            sorted_dips = sorted(dip_points, key=lambda item: item['depth'], reverse=True)
            
            # 提取最深的两个点
            p1_dict = sorted_dips[0]
            p2_dict = sorted_dips[1]
            
            # 按x坐标对这两个点排序
            final_points = sorted([p1_dict['point'], p2_dict['point']], key=lambda p: p[0])
            p1, p2 = final_points
            
            horizontal_distance = abs(p2[0] - p1[0])
            self.updateResultsAndDisplay(horizontal_distance, points=[p1, p2])

    def updateResultsAndDisplay(self, distance, rects=None, points=None):
        self.known_distance_um = self.known_distance.value()
        if distance > 0:
            self.conversion_factor = self.known_distance_um / distance
            self.pixel_distance_label.setText(f"{distance:.2f} 像素")
            self.conversion_label.setText(f"{self.conversion_factor:.4f} 微米/像素")
            self.save_button.setEnabled(True)
            if self.main_app:
                self.apply_button.setEnabled(True)
        else:
            self.conversion_factor = 0.0
            self.pixel_distance_label.setText("测量失败")
            self.conversion_label.setText("N/A")
            self.save_button.setEnabled(False)
            self.apply_button.setEnabled(False)
        
        result_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        p1, p2 = None, None
        if rects:
            p1_center = (int((rects[0][0] + rects[0][2]) / 2), int((rects[0][1] + rects[0][3]) / 2))
            p2_center = (int((rects[1][0] + rects[1][2]) / 2), int((rects[1][1] + rects[1][3]) / 2))
            
            for r in rects:
                cv2.rectangle(result_image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            
            cv2.line(result_image, (p1_center[0], p1_center[1]), (p2_center[0], p1_center[1]), (255, 0, 0), 2)
            p1 = p1_center
            p2 = p2_center
        
        if points:
            p1, p2 = points
            cv2.circle(result_image, p1, 7, (0, 0, 255), -1)
            cv2.circle(result_image, p2, 7, (0, 0, 255), -1)
            cv2.line(result_image, p1, p2, (255, 0, 0), 2)
        
        if (points or rects) and distance > 0:
            mid_point_x = int((p1[0] + p2[0]) / 2)
            mid_point_y = int(p1[1] - 15)
            cv2.putText(result_image, f"{distance:.2f} px", (mid_point_x, mid_point_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self.processed_image = result_image
        height, width, _ = result_image.shape
        bytes_per_line = width * 3
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.processed_image_label.setPixmap(QPixmap.fromImage(q_image))

    def save_result(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "无结果", "没有可保存的处理结果。")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果图像", "", 
                                                  "PNG 文件 (*.png);;JPG 文件 (*.jpg)", options=options)
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                QMessageBox.information(self, "保存成功", f"结果已保存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时出错: {str(e)}")

    def apply_to_main_window(self):
        if self.main_app and self.conversion_factor > 0:
            self.main_app.update_conversion_factors(self.conversion_factor)
            QMessageBox.information(self, "应用成功", f"新的转换系数 {self.conversion_factor:.4f} μm/px 已应用到主窗口。")
            self.close()
        else:
            QMessageBox.warning(self, "应用失败", "没有有效的转换系数可以应用，或者未关联主窗口。")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PixelCalibrationApp()
    window.show()
    sys.exit(app.exec_()) 