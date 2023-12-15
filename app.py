import argparse
import gradio as gr
import cv2
import os
import time 
import ffmpeg
from ultralytics import YOLO

object_name = "LicensePlate"

# Run YOLO 
def run_yolo(input_video_url):
    
    # model define
    model = YOLO(f'./weights/pd_base.pt')
    results = model(input_video_url)

    cap = cv2.VideoCapture(input_video_url)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 

    video_name = os.path.basename(input_video_url)
    output_path = os.getcwd()+'/video/out/' + video_name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in range(len(results)):
        result = results[i]
        img = result.plot()

        out.write(img)
    out.release()
    now = time.localtime()
    resized_video = time.strftime('%Y%m%d%H%M%S') + '.mp4'
    ffmpeg.input(output_path).output(resized_video , crf=35).run() # Reduce Video File Size 
    return input_video_url, resized_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_name',
        type=str,
        default='0.0.0.0'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860
    )
    args=parser.parse_args()

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            markdown = gr.Markdown(f"# {object_name} Detector")
            input1 = gr.Textbox(label = "Video URL") # Put Video URL
            btn1 = gr.Button("Run", size="sm")
        with gr.Column():
            output1 = gr.Video(autoplay=True) # Play Original Video 
        with gr.Column():
            output2 = gr.Video(autoplay=True) # Play Result Video
        btn1.click(fn = run_yolo, inputs=input1, outputs=[output1, output2])

demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        debug=True
    )