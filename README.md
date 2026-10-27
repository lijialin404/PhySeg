# PhySeg: Progressively Refining SAM for Accurate Phytoplankton Segmentation
Official implementation of **Progressively Refining SAM for Accurate Phytoplankton Segmentation**.

> **Progressively Refining SAM for Accurate Phytoplankton Segmentation**  
> Jialin Li, Jinkun Jiang, Yuezun Li\*, and Junyu Dong  
> School of Computer Science and Technology, Ocean University of China, Qingdao, China  
> \* Corresponding author
>
> **Abstract:**  
> Accurate phytoplankton segmentation is essential for ecological monitoring, yet it remains largely underexplored due to the subtle visual characteristics of phytoplankton and the scarcity of pixel-level annotations. To address these challenges, we propose PhySeg, a SAM-based segmentation framework equipped with a lightweight refinement module that progressively restores fine-grained details through multi-scale feature fusion. In addition, we construct a phytoplankton segmentation dataset by manually annotating instance-level masks for an existing dataset, providing a valuable resource for future research. Experimental results demonstrate that PhySeg improves segmentation accuracy, validating the effectiveness of the proposed module.

![PhySeg架构图](fig/figure_overview.jpg)

![PhySegD采样图](fig/figure_sample.jpg)

## Installation

The code has been verified on Windows with Python **3.12.13**, PyTorch **2.10.0+cu128**, torchvision **0.25.0+cu128**, and an NVIDIA RTX 5070 Ti Laptop GPU. Training and evaluation use CUDA and BF16.

```bash
conda create -n physeg python=3.12.13 -y
conda activate physeg
python -m pip install -r requirements.txt
```

The dependencies in [requirements.txt](requirements.txt) were also installed and checked in a fresh Conda environment, where both complete evaluation splits reproduced the reference results.

**arxiv链接**: 

**权重和数据集**:
https://pan.baidu.com/s/18b09leGGkA4P0iNjLn6bcg?pwd=j8y7
