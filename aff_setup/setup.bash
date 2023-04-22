wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth
mv control_sd15_seg.pth ./models/
python tool_add_control.py ./models/control_sd15_seg.pth ./models/control_sd15_ini2.ckpt
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth
mv upernet_global_small.pth ./annotator/ckpts/