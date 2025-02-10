# /home/ubuntu/anaconda3/envs/yolo/lib/python3.10
# cd /home/ubuntu/文档/GitHub/DL大作业 ; sudo /usr/bin/env /home/ubuntu/anaconda3/envs/yolo/lib/python3.10 /home/ubuntu/.vscode/extensions/ms-python.debugpy-2024.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 58259 -- /home/ubuntu/文档/GitHub/DL大作业/web_demo.py
import gradio as gr
from ultralytics import YOLO
import PIL.Image as Image
import cv2
import tempfile
from collections import defaultdict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

model_path = './models/ua-10.onnx'

labels = ['truck', 'car', 'van', 'bus']

# 视频处理函数
def mymodel(vid):
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

app4 = gr.Interface(fn=mymodel, inputs=gr.Video(), outputs="playable_video", title="导入视频", description="请导入视频文件，以进行车辆分类模型的预测。")


demo = gr.TabbedInterface(
                          [app4],
                          tab_names=["导入视频"],
                          title="车辆分类模型"
                          )
demo.launch()
