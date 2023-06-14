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


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def showmasksOnImage(image, masks):
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # 将masks显示在原图上
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_masks[0]['segmentation'].shape[0],
                  sorted_masks[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_masks:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(f'/tmp/out-1.png')
    plt.close()


def showAllMasks(image, masks):
    # 显示所有的masks
    image = np.uint8(image)
    sorted_masks = sorted(masks, key=(lambda x: x['bbox'][0]))
    mask_num = len(sorted_masks)
    ncols = 3
    nrows = math.ceil(mask_num / ncols)
    fig, axes = plt.subplots(nrows, ncols)
    count = 0
    # 子图取消横纵坐标
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].axis('off')

    for ann in sorted_masks:
        img = np.zeros((sorted_masks[0]['segmentation'].shape[0],
                        sorted_masks[0]['segmentation'].shape[1]), dtype=np.uint8)
        m = ann['segmentation']
        bbox = ann['bbox']
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        img[m] = 255
        result = cv2.bitwise_and(image, image, mask=img)
        mask = result[y:y+h+1, x:x+w+1]
        axes[int(count / ncols)][count %
                                 ncols].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        count += 1

    # 调整子图的大小和间距
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.2, hspace=0.4)
    fig.savefig(f'/tmp/out-2.png', dpi=600)
    plt.close(fig)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.model = sam

    def predict(
        self,
        image: Path = Input(description="image"),
        points_per_site: int = Input(
            description="""The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.""",
            default=12,
        ),
        points_per_batch: int = Input(
            description="""Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.""",
            default=500,
        ),
        pred_iou_thresh: float = Input(
            description="""A filtering threshold in [0,1], using the
            model's predicted mask quality.""",
            default=0.88,
            ge=0,
            le=1,
        ),
        stability_score_thresh: float = Input(
            description="""A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.""",
            default=0.95,
            ge=0,
            le=1,
        ),
        stability_score_offset: float = Input(
            description="""The amount to shift the cutoff when
            calculated the stability score.""",
            default=1.0,
        ),
        box_nms_thresh: float = Input(
            description="""The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.""",
            default=0.7,
        ),
        crop_n_layers: int = Input(
            description="""If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.""",
            default=0,
        ),
        crop_nms_thresh: float = Input(
            description="""The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.""",
            default=0.7,
        ),
        crop_overlap_ratio: float = Input(
            description="""Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.""",
            default=512 / 1500,
        ),
        crop_n_points_downscale_factor: int = Input(
            description="""The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.""",
            default=1,
        ),
        min_mask_region_area: int = Input(
            description="""If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.""",
            default=0,
        ),
        output_mode: str = Input(
            description="""The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.""",
            default="binary_mask",
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_site,
            points_per_batch,
            pred_iou_thresh,
            stability_score_thresh,
            stability_score_offset,
            box_nms_thresh,
            crop_n_layers,
            crop_nms_thresh,
            crop_overlap_ratio,
            crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode
        )
        image = cv2.imread(str(image))
        processed_input = preprocess(image)
        masks = mask_generator.generate(processed_input)
        if len(masks) == 0:
            raise Exception(
                f"No object detected in the image"
            )
        showmasksOnImage(image, masks)
        showAllMasks(image, masks)
        return [Path(f'/tmp/out-1.png'), Path(f'/tmp/out-2.png')]
