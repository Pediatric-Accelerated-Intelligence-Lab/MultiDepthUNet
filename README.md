# MultiDepth UNet for Resource-Aware Federated Medical Segmentation

This repository contains the MultiDepth UNet implementation for the paper:

Fed-ADApt: Federated Anytime Depth Adaptation for Resource-Aware Medical Image Segmentation

## Overview

Federated medical image segmentation models such as UNet are often trained under the assumption that all participating clients have identical compute and memory resources. This is unrealistic in federated learning (FL), especially for low-resource imaging settings.

This repository implements a multi-depth UNet that supports resource-aware training and inference across heterogeneous client capacities. The core idea is to keep a shared decoder path while allowing each client to activate only a subset of encoder depth levels according to its available compute budget.

The implementation in `model/unet.py` is designed for this setting and includes:

- dynamic active-depth execution via `active_layers`
- a shared decoder path with depth-aware bypass behavior
- FLOPs estimation for compute-budget analysis
- learnable-parameter inspection across client depths
- multi-depth inference from cached encoder features

## Paper context

The corresponding paper addresses the challenge of federated medical image segmentation under heterogeneous deployment and training budgets. It proposes a framework that enables:

- low-resource sites to participate in federated training without full-capacity model requirements
- adaptive inference at deployment time based on local compute availability
- robustness under domain shift across decentralized clinical datasets
- efficient training and inference for 2D and 3D segmentation tasks

The method is motivated by the observation that typical FL methods assume homogeneous clients, while real clinical environments often contain devices with very different memory, compute, power, and latency constraints.

## Model details

The implementation provides two main model classes:

- `UNet` — single-depth depth-adaptive model
- `UNetMultiDepth` — multi-depth model for evaluating multiple active depths from a shared feature-extraction pass

Key behavior:

- `active_layers` controls how many UNet levels are active
- deeper levels can be bypassed while preserving a shared decoder path
- the decoder uses a single shared `up_full` path, with a frozen adapter on bypass branches when needed
- this design reduces aggregation mismatch across heterogeneous federated clients and supports adaptive deployment

## Quick Start

This project was created with `Python 3.12` in mind:

```bash
python -m venv .fl
source .fl/bin/activate
pip install -r requirements.txt
python model/unet.py
```

This script runs a model sanity check that prints:

- output shapes for multiple depths
- FLOPs estimates for different `active_layers`
- learnable-parameter counts for different active depths

## Notes

- The architecture is designed for federated learning under heterogeneous hardware constraints.
- Clients with different compute budgets can train or infer using different network depths without maintaining separate models for each client.
- The implementation serves as a research reference for the paper’s multi-depth UNet design.
