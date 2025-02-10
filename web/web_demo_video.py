# /home/ubuntu/anaconda3/envs/yolo/lib/python3.10
# cd /home/ubuntu/文档/GitHub/DL大作业 ; sudo /usr/bin/env /home/ubuntu/anaconda3/envs/yolo/lib/python3.10 /home/ubuntu/.vscode/extensions/ms-python.debugpy-2024.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 58259 -- /home/ubuntu/文档/GitHub/DL大作业/web_demo.py
import gradio as gr
from ultralytics import YOLO
import PIL.Image as Image
import cv2
import tempfile

model_path = './models/ua-10.onnx'

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
        result = yolo.predict(source=frame, save=True)
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

app4 = gr.Interface(fn=mymodel, inputs=gr.Video(), outputs="playable_video", title="导入视频", description="请导入视频文件，以进行车辆分类模型的预测。")


demo = gr.TabbedInterface(
                          [app4],
                          tab_names=["导入视频"],
                          title="车辆分类模型"
                          )
demo.launch()
