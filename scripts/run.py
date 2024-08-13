from argparse import ArgumentParser
from pathlib import Path

import logging

from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra import compose
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.sam2_video_interface import SAM2VideoInterface


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--video", type=str, required=True)


def prelude():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.sam2.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def main(args):
    # apply_postprocessing is just assumed to be True here
    # prelude()
    cfg = compose(config_name=args.config)
    OmegaConf.resolve(cfg)
    interface = instantiate(cfg, _recursive_=True)  # type: SAM2VideoInterface
    load_checkpoint(interface, args.ckpt)

    # RUN
    video_dir = Path(args.video)
    frame_names = [
        p for p in video_dir.iterdir()
        if p.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(p.stem))
    interface.init_state(args.video)

    ann_obj_id = 1
    ann_frame_idx = 0

    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = interface.add_new_points(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(video_dir / frame_names[ann_frame_idx]))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(),
              plt.gca(), obj_id=out_obj_ids[0])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
