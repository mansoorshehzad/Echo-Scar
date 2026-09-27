"""Extract 512-channel nnU-Net bottleneck features from each cine frame.

All views use eight encoder stages as specified by the author for the final
analysis. Other checkpoint settings are provisional until the final plans and
weights are available. Supply --model-config to override them. ES and ED are
read from existing nnU-Net segmentation masks; this module does not segment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_features import SUMMARY_NAMES, VIEW_SEGMENTS, phase_indices, summarize
from extract_radiomics import time_first


MODEL_CONFIGS = {
    "a2c": dict(dataset="Dataset001_CamusA2C", stages=8, channels=(32, 64, 128, 256, 512, 512, 512, 512),
                classes=4, myocardium_label=2, mean=85.81, std=47.21),
    "a4c": dict(dataset="Dataset002_CamusA4C", stages=8, channels=(32, 64, 128, 256, 512, 512, 512, 512),
                classes=4, myocardium_label=2, mean=84.71, std=43.66),
    "a3c": dict(dataset="Dataset007_MIMICA3CPlus", stages=8,
                channels=(32, 64, 128, 256, 512, 512, 512, 512),
                classes=2, myocardium_label=1, mean=77.89, std=50.90),
}


def model_config(view: str, config_file: Path | None = None) -> dict:
    """Use paper-stage defaults, with an optional final checkpoint override."""
    config = dict(MODEL_CONFIGS[view])
    if config_file is not None:
        config.update(json.loads(config_file.read_text()))
    if len(config["channels"]) != config["stages"]:
        raise ValueError("The channel list must have one entry per encoder stage")
    if config["channels"][-1] != 512:
        raise ValueError("The paper's encoder feature vector needs 512 bottleneck channels")
    return config


def build_network(config: dict):
    """Reconstruct the network used to produce a view-specific checkpoint."""
    import torch.nn as nn
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    stages = config["stages"]
    return PlainConvUNet(
        input_channels=1,
        n_stages=stages,
        features_per_stage=config["channels"],
        conv_op=nn.Conv2d,
        kernel_sizes=[[3, 3]] * stages,
        strides=[[1, 1]] + [[2, 2]] * (stages - 1),
        n_conv_per_stage=[2] * stages,
        num_classes=config["classes"],
        n_conv_per_stage_decoder=[2] * (stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        deep_supervision=True,
    )


def load_network(checkpoint_root: Path, view: str, device: str, config: dict | None = None):
    """Load one trusted local nnU-Net checkpoint and verify its architecture."""
    import torch

    config = config or model_config(view)
    checkpoint = (checkpoint_root / config["dataset"] /
                  "nnUNetTrainer__nnUNetPlans__2d" / "fold_all" / "checkpoint_final.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    network = build_network(config)
    # nnU-Net checkpoints can contain more than tensors; load only checkpoints
    # produced by the study team or another source you trust.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(state["network_weights"])
    return network.eval().to(device)


def frame_tensor(frame: np.ndarray, config: dict, device: str):
    """Normalize and pad one frame for the view-specific encoder."""
    import torch
    import torch.nn.functional as functional

    normalized = (frame.astype(np.float32) - config["mean"]) / config["std"]
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    divisor = 2 ** (config["stages"] - 1)
    height, width = frame.shape
    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    return functional.pad(tensor, (0, pad_w, 0, pad_h)), pad_h, pad_w


def extract_case(cine: np.ndarray, segmentation: np.ndarray, network, config: dict, device: str,
                 frame_rate: float = 30.0) -> dict[str, np.ndarray]:
    """Summarize pooled encoder channels using phases from an existing mask."""
    import torch

    if cine.ndim != 3 or len(cine) < 2:
        raise ValueError("Expected at least two cine frames in T x H x W order")
    if cine.shape != segmentation.shape:
        raise ValueError(f"Cine shape {cine.shape} differs from segmentation {segmentation.shape}")
    bottlenecks = []
    with torch.inference_mode():
        for frame in cine:
            tensor, _, _ = frame_tensor(frame, config, device)
            # The last encoder skip is the 512-channel bottleneck map.
            last_skip = network.encoder(tensor)[-1]
            bottlenecks.append(last_skip.mean(dim=(2, 3))[0].cpu().numpy())
    es, ed = phase_indices(segmentation, config["myocardium_label"])
    summaries = summarize(np.stack(bottlenecks), es, ed, frame_rate)
    summaries["feature_vector"] = np.concatenate([summaries[name] for name in SUMMARY_NAMES])
    summaries["es_idx"] = np.array(es)
    summaries["ed_idx"] = np.array(ed)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cine-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True,
                        help="Existing nnU-Net masks, one <case>.nii.gz per cine")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=VIEW_SEGMENTS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--model-config", type=Path,
                        help="JSON overrides from the final nnU-Net plans/checkpoint")
    args = parser.parse_args()
    import nibabel as nib

    config = model_config(args.view, args.model_config)
    network = load_network(args.checkpoint_root, args.view, args.device, config)
    files = sorted(args.cine_dir.glob("*.nii.gz"))
    if not files:
        parser.error("No .nii.gz cines found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cine_path in files:
        case_id = cine_path.name.removesuffix(".nii.gz").removesuffix("_0000")
        seg_path = args.seg_dir / f"{case_id}.nii.gz"
        if not seg_path.is_file():
            raise FileNotFoundError(f"Missing segmentation for {cine_path.name}: {seg_path}")
        cine = time_first(np.asarray(nib.load(cine_path).dataobj))
        segmentation = time_first(np.asarray(nib.load(seg_path).dataobj))
        result = extract_case(cine, segmentation, network, config, args.device, args.frame_rate)
        np.savez_compressed(args.output_dir / f"{case_id}.npz", **result)
        print(f"{case_id}: ES={int(result['es_idx'])}, ED={int(result['ed_idx'])}")


if __name__ == "__main__":
    main()
