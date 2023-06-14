# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import numpy as np
import torch
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import cv2
import math
from typing import List
from text_vision import get_words_boxes
from PIL import Image, ImageDraw, ImageFont

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_boxes_xyxy(boxes):
    output = []
    for b in boxes:
        output.append(b['xyxy'])
    return torch.Tensor(output)

def merge_masks(masks):
    output = np.zeros_like(masks[0][0], dtype=bool)
    for m in masks:
        output = np.bitwise_or(output, m[0])
    return output

def draw_boxes(boxes, image, random_color=True):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    image = Image.fromarray(image)
    # 创建draw对象
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        color_tuple = tuple((color*255).astype(np.uint8))
        # 绘制边框
        draw.rectangle(box['xyxy'], outline=color_tuple, width=1)
        # 绘制标签
        label_xyxy = [box['xyxy'][0], box['xyxy'][1]-8, box['xyxy'][0] + 12, box['xyxy'][1]]
        draw.rectangle(label_xyxy, fill=color_tuple)
        # 获取文本宽度和高度
        text_width, text_height = draw.textsize(box['text'])
        text_x = (label_xyxy[0] + label_xyxy[2] - text_width) // 2
        text_y = (label_xyxy[1] + label_xyxy[3] - text_height) // 2
        text_color = (255,255,255)
        draw.text((text_x, text_y), box['text'], fill=text_color, anchor="mm")
    
    return np.array(image)

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)), np.array(mask_image_pil)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)

        self.model = sam

    def predict(
        self,
        image_path: Path = Input(description="image"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        sam_predictor = SamPredictor(self.model)
        # 读取图片
        image = cv2.imread(str(image_path))
        # 预处理
        processed_image = preprocess(image)
        boxes = get_words_boxes(str(image_path))
        # Set image
        sam_predictor.set_image(processed_image)
        # Get masks
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(get_boxes_xyxy(boxes), processed_image.shape[:2]).to(self.device)
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        if len(masks) == 0:
            raise Exception(
                f"No object detected in the image"
            )
        # Merge masks
        merged_masks = merge_masks(masks.cpu().numpy())
        # Draw boxes
        annotated_frame = draw_boxes(boxes, image)
        # Show masks
        annotated_frame_with_mask, split_mask = show_mask(merged_masks, annotated_frame)
        output1 = Image.fromarray(annotated_frame_with_mask)
        output2 = Image.fromarray(split_mask)
        # Save output image
        output1.save(f'/tmp/out-1.png') 
        output2.save(f'/tmp/out-2.png') 
        return [Path(f'/tmp/out-1.png'), Path(f'/tmp/out-2.png')]
