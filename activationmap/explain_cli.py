"""
Command-line explainability for GEDI-ORDER models.

Runs the modern GradientTape XAI engine (`activationmap.explain`) over a
folder of images and writes, per image, a heatmap overlay plus a CSV of
predictions. Works for classification models (choose a class or let it use
the predicted class) and for scalar survival/regression heads.

Example
-------
    python activationmap/explain_cli.py \\
        --im_dir GradCAM_example/Mito_T8-12-Lipo_True \\
        --model_path model_01.keras \\
        --method gradcam++ --backbone resnet50 \\
        --resdir out_explain --imtype tif
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from imageio import imwrite

import param_gedi as param
from activationmap.grad_ops import GradOps
from activationmap.explain import Explainer, find_last_conv_layer
from models.backbones import gradcam_layer_for, _BACKBONES

METHODS = {
    "gradcam": "gradcam",
    "gradcam++": "gradcam_plus_plus",
    "scorecam": "score_cam",
    "ig": "integrated_gradients",
}


def _load_images(im_dir, imtype, gops):
    files = sorted(glob.glob(os.path.join(im_dir, f"*.{imtype}")))
    imgs, crops, names = [], [], []
    for f in files:
        from imageio.v3 import imread
        raw = imread(f)
        # Guard against channels-first arrays (C, H, W) with C in {3, 4}.
        if raw.ndim == 3 and raw.shape[0] in (3, 4) and raw.shape[-1] not in (3, 4):
            raw = np.transpose(raw, (1, 2, 0))
        proc, cropped = gops.img_parse(raw)
        imgs.append(proc.astype(np.float32))
        crops.append(cropped)
        names.append(os.path.splitext(os.path.basename(f))[0])
    return np.array(imgs), crops, names


def main():
    ap = argparse.ArgumentParser(description="Modern XAI for GEDI-ORDER models")
    ap.add_argument("--im_dir", required=True, help="folder of images")
    ap.add_argument("--model_path", required=True, help=".keras model")
    ap.add_argument("--resdir", required=True, help="output folder")
    ap.add_argument("--method", default="gradcam", choices=list(METHODS))
    ap.add_argument("--layer_name", default=None,
                    help="conv layer to localize on (default: auto / backbone preset)")
    ap.add_argument("--backbone", default=None, choices=list(_BACKBONES),
                    help="use this backbone's preset Grad-CAM layer")
    ap.add_argument("--target", type=int, default=None,
                    help="class index to explain (default: predicted class)")
    ap.add_argument("--imtype", default="tif")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--ig_steps", type=int, default=32,
                    help="integration steps for the 'ig' method (fewer = faster)")
    ap.add_argument("--alpha", type=float, default=0.4, help="overlay opacity")
    args = ap.parse_args()

    os.makedirs(args.resdir, exist_ok=True)
    p = param.Param(parent_dir=args.resdir, res_dir=args.resdir)
    gops = GradOps(p, vgg_normalize=True)

    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    layer = args.layer_name
    if layer is None and args.backbone:
        layer = gradcam_layer_for(args.backbone)
    if layer is None:
        layer = find_last_conv_layer(model)
    print(f"Explaining with method={args.method}, layer={layer}")

    exp = Explainer(model, layer_name=layer)
    method_fn = getattr(exp, METHODS[args.method])

    imgs, crops, names = _load_images(args.im_dir, args.imtype, gops)
    if len(imgs) == 0:
        print("No images found.")
        return
    print(f"Loaded {len(imgs)} images.")

    rows = []
    for i in range(0, len(imgs), args.batch_size):
        batch = imgs[i:i + args.batch_size]
        preds = model.predict(batch, verbose=0)
        if args.method == "ig":
            heatmaps = method_fn(batch, target=args.target, steps=args.ig_steps)
        else:
            heatmaps = method_fn(batch, target=args.target)
        for j in range(len(batch)):
            name = names[i + j]
            pred = preds[j]
            pred_cls = int(np.argmax(pred)) if pred.shape[-1] > 1 else float(pred[0])
            overlay = exp.overlay(crops[i + j], heatmaps[j], alpha=args.alpha)
            imwrite(os.path.join(args.resdir, f"{name}_{args.method}.png"), overlay)
            rows.append({"filename": name, "prediction": pred_cls,
                         "raw_output": np.array2string(pred, precision=4)})
        print(f"  processed {min(i + args.batch_size, len(imgs))}/{len(imgs)}")

    csv = os.path.join(args.resdir, f"predictions_{args.method}.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"Done. Overlays + {csv} written to {args.resdir}")


if __name__ == "__main__":
    main()
