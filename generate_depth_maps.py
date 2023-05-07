# from share import *
import config

import cv2
# import einops
# import gradio as gr
# import numpy as np
import torch
# import random

from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
# from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler


apply_midas = MidasDetector()

# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)


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


