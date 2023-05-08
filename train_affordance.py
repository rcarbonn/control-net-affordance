from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from affordance_dataset import ADE20kAffordanceDataset
from iitf_dataset import IITFAffordanceDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = './models/control_sd15_ini2.ckpt'
resume_path = './lightning_logs/version_3/checkpoints/epoch=9-step=7729.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = True


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
data_dir = '/home/ubuntu/vlr_proj'
# dataset = ADE20kAffordanceDataset(data_dir)
dataset = IITFAffordanceDataset(data_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# model_ckpt = pl.callbacks.ModelCheckpoint(monitor="step", every_n_train_steps=1000, save_top_k=1)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=2, max_epochs=10, enable_checkpointing=True)


# Train!
trainer.fit(model, dataloader)
trainer.save_checkpoint("new5.ckpt")
