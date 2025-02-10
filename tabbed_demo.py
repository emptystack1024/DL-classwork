import gradio as gr
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
from collections import defaultdict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 定义模型路径
model_path = '/home/ubuntu/文档/GitHub/DL大作业/models/ua-40.onnx'

labels = ['truck', 'car', 'van', 'bus']

def test1(img):
    # 加载YOLO模型
    model = YOLO(model_path, task='detect')
    results = model(img)
    
    # 绘制检测结果
    for r in results:
        img = r.plot()
        im = Image.fromarray(img)
    
    # 返回结果
    return im

demo = gr.Interface(fn=test1, inputs="image", outputs="image", title="导入图片", description="请导入图片文件，以进行车辆分类模型的预测。")

def yolo_pre(video_path):
    yolo = YOLO(model_path)
    # video_path='./pred.mp4' #检测视频的地址
    cap = cv2.VideoCapture(video_path)  # 创建一个 VideoCapture 对象，用于从视频文件中读取帧
    # 获取视频帧的维度
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./result.mp4', fourcc, 20.0, (frame_width, frame_height)) #保存检测后视频的地址
 
    while cap.isOpened():
        status, frame = cap.read()  # 使用 cap.read() 从视频中读取每一帧
        if not status:
            break
        result = yolo.predict(source=frame, save=True, imgsz=1280)
        result = result[0]
        anno_frame = result.plot()
        #cv2.imshow('行人', anno_frame)
        out.write(anno_frame) #写入保存
        # 注释的框架是通过调用 result.plot() 获得的，它会在框架上绘制边界框和标签。
        # 带注释的框架使用 cv2.imshow() 窗口名称“行人”显示。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    video_yolo_path='./result.mp4'
    return video_yolo_path

# 视频处理函数
def test2(vid):
    # 加载YOLO模型
    model = YOLO(model_path, task='detect')
    
    # 临时文件保存检测结果视频
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # 打开输入视频
    cap = cv2.VideoCapture(vid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        status, frame = cap.read()  # 修正此处
        if not status:
            break

        # 将帧传递给YOLO模型进行检测
        results = model(frame)
        
        # 绘制检测结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 获取检测到的框
            confs = result.boxes.conf.cpu().numpy()  # 获取置信度
            classes = result.boxes.cls.cpu().numpy()  # 获取类别

            frame = result.plot()  # 绘制检测结果

            # for box, conf, cls in zip(boxes, confs, classes):
            #     x1, y1, x2, y2 = map(int, box)
            #     label = f'{labels[int(cls)]} {conf:.2f}'
            #     if cls == 0:
            #     	color = (255, 0 ,0)  # 绿色框
            #     elif cls == 1:
            #     	color = (0, 225, 0)
            #     elif cls == 2:
            #     	color = (0, 0, 225)
            #     else:
            #     	color = (255, 188 ,211)

            #     # 绘制矩形框和标签
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 写入处理后的帧到输出视频
        out.write(frame)
    
    cap.release()
    out.release()

    print("处理完成")
    
    return output_path

demo2 = gr.Interface(fn=test2, inputs=gr.Video(), outputs="playable_video", title="导入视频", description="请导入视频文件，以进行车辆分类模型的预测。")


def test3(vid):
    # 加载YOLO模型
    model = YOLO(model_path, task='detect')
    
    # 临时文件保存检测结果视频
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # 打开输入视频
    cap = cv2.VideoCapture(vid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Store the track history
    track_history = defaultdict(lambda: [])
    
    # 定义视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        status, frame = cap.read()  # 修正此处
        if not status:
            break

        # 将帧传递给YOLO模型进行检测
        results = model.track(frame, persist=True)
        
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
        
            # Visualize the results on the frame
            frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

		        # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
	
        	
        
        # 写入处理后的帧到输出视频
        out.write(frame)
    
    cap.release()
    out.release()

    print("处理完成")
    
    return output_path

demo3 = gr.Interface(fn=test3, inputs=gr.Video(), outputs="playable_video", title="导入视频", description="请导入视频文件，以进行车辆分类模型的预测。")

gr.TabbedInterface([demo, demo2, demo3], css="#htext span {white-space: pre} #htext2 span {white-space: pre} #htext3 span {white-space: pre}").launch()
