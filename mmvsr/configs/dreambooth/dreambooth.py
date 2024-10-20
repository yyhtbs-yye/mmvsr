# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.gen_default_runtime import *

from mmengine.dataset.sampler import InfiniteSampler
from torch.optim import AdamW

from mmvsr.datasets.dreambooth_dataset import DreamBoothDataset
from mmvsr.datasets.transforms.aug_shape import Resize
from mmvsr.datasets.transforms.formatting import PackInputs
from mmvsr.datasets.transforms.loading import LoadImageFromFile
from mmvsr.engine import VisualizationHook
from mmvsr.models.data_preprocessors.data_preprocessor import DataPreprocessor
from mmvsr.models.editors.disco_diffusion.clip_wrapper import ClipWrapper
from mmvsr.models.editors.dreambooth import DreamBooth

stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

val_prompts = [
    'a sks dog in basket', 'a sks dog on the mountain',
    'a sks dog beside a swimming pool', 'a sks dog on the desk',
    'a sleeping sks dog', 'a screaming sks dog', 'a man in the garden'
]

model = dict(
    type=DreamBooth,
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='unet',
    ),
    text_encoder=dict(
        type=ClipWrapper,
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type=DataPreprocessor),
    val_prompts=val_prompts)

train_cfg = dict(max_iters=1000)

optim_wrapper.update(
    modules='.*unet',
    optimizer=dict(type=AdamW, lr=5e-6),
    accumulative_counts=4  # batch size = 4 * 1 = 4
)

pipeline = [
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=Resize, scale=(512, 512)),
    dict(type=PackInputs)
]

dataset = dict(
    type=DreamBoothDataset,
    data_root='./data/dreambooth',
    concept_dir='imgs',
    prompt='a photo of sks dog',
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    persistent_workers=True,
    batch_size=1)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# hooks
default_hooks.update(dict(logger=dict(interval=10)))
custom_hooks = [
    dict(
        type=VisualizationHook,
        interval=50,
        fixed_input=True,
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1)
]
