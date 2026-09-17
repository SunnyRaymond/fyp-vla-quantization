# CLAP: A Closed-Loop Diffusion Transformer Action Foundation Model for Robotic Manipulation

paper_id: semanticscholar:1bd4e6432bd37af66b82183f49d76693ee3fc145
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

The development of large Vision-Language-Action (VLA) models has enhanced the robot’s ability to manipulate objects in unseen scenarios based on language instructions. While existing VLAs have demonstrated promise in various scenarios, they still struggle with effective multi-modal data feature extraction and lack a closed-loop inference framework. In this paper, we propose an advanced VLA model. Unlike previous works that repurpose VLM for action prediction using simple action quantization, we componentized the VLA architecture with a specialized action module conditioned on the model output and a critic module for inference. We demonstrate the performance improvement of diffusion action transformers in modeling continuous temporal actions, with the critic module applied during inference to form a closed-loop model. Extensive experiments on real robots demonstrate that our model significantly outperforms existing methods, with the ability to handle complex, high-precision tasks and generalize to unseen objects and backgrounds.

## Method


