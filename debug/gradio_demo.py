from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import socket

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# Define the processing function
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
):
    # Handle different image input formats from Gradio
    if image_input is None:
        return None, "No image provided"
    
    # Convert to PIL Image if needed
    if isinstance(image_input, np.ndarray):
        image_input = Image.fromarray(image_input)
    elif not isinstance(image_input, Image.Image):
        # Try to convert from other formats
        try:
            image_input = Image.fromarray(image_input)
        except:
            return None, "Invalid image format"

    # Calculate the box overlay ratio based on the image size
    box_overlay_ratio = image_input.size[0] / 3200
    # Define the configuration for drawing bounding boxes
    draw_bbox_config = {
        "box_threshold": box_threshold,
        "iou_threshold": iou_threshold,
        "box_overlay_ratio": box_overlay_ratio,
        "use_paddleocr": use_paddleocr,
        "imgsz": imgsz
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    # parsed_content_list = str(parsed_content_list)
    return image, str(parsed_content_list)

with gr.Blocks() as demo:
    # Define the input components
    image_input = gr.Image(label="Input Image")
    box_threshold = gr.Slider(0, 1, value=0.5, label="Box Threshold")
    iou_threshold = gr.Slider(0, 1, value=0.5, label="IoU Threshold")
    use_paddleocr = gr.Checkbox(label="Use PaddleOCR",value=True)
    #use_paddleocr_component = gr.Checkbox(label='Use PaddleOCR', value=True, interactive=True)
    imgsz = gr.Slider(100, 1000, value=640, label="Image Size")

    # Define the output components
    processed_image_output = gr.Image(label="Processed Image")
    parsed_content_output = gr.Textbox(label="Parsed Content")

    # Define the event listener
    process_button = gr.Button("Process")
    process_button.click(
        fn=process,
        inputs=[image_input, box_threshold, iou_threshold, use_paddleocr, imgsz],
        outputs=[processed_image_output, parsed_content_output]
    )



free_port = get_free_port()
# demo.launch(debug=False, show_error=True, share=True)
# demo.launch(share=True, server_port=7861, server_name='127.0.0.1')
demo.launch(share=True, server_port=free_port, server_name='0.0.0.0')