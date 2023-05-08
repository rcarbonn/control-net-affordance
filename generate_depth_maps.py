import cv2
import torch

from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector

apply_midas = MidasDetector()


def img_to_depth(input_image, image_resolution=512, detect_resolution=384):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = detected_map

    return detected_map

