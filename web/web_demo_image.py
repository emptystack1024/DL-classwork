import gradio as gr
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

# 定义模型路径
model_path = 'models/ua-10.onnx'

# 图像处理函数
def image1(img):
    # 加载YOLO模型
    model = YOLO(model_path, task='detect')
    results = model(img)
    
    # 绘制检测结果
    for r in results:
        img = r.plot()
        im = Image.fromarray(img)
    
    # 返回结果
    return im

# 定义Gradio接口
app1 = gr.Interface(fn=image1, inputs="image", outputs="image", title="导入图片", description="请导入图片文件，以进行车辆分类模型的预测。")

demo = gr.TabbedInterface(
    [app1],
    tab_names=["导入图片"],
    title="车辆分类模型"
)

demo.launch()
