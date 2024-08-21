# SelfDRSC++: Self-Supervised Learning for Dual Reversed Rolling Shutter Correction
Under Review
---
[[arXiv](https://arxiv.org/abs/2305.19862)]

This repository is the official PyTorch implementation of SelfDRSC++: Self-supervised Learning for Dual Reversed Rolling Shutter Correction.

### Introduction
Modern consumer cameras commonly employ the rolling shutter (RS) imaging mechanism, via which images are captured by scanning scenes row-by-row, resulting in RS distortion for dynamic scenes. To correct RS distortion, existing methods adopt a fully supervised learning manner that requires high framerate global shutter (GS) images as ground-truth for supervision. In this paper, we propose an enhanced Self-supervised learning framework for Dual reversed RS distortion Correction (SelfDRSC++). Firstly, we introduce a lightweight DRSC network that incorporates a bidirectional correlation matching block to refine the joint optimization of optical flows and corrected RS features, thereby improving correction performance while reducing network parameters. Subsequently, to effectively train the DRSC network, we propose a self-supervised learning strategy that ensures cycle consistency between input and reconstructed dual reversed RS images. The RS reconstruction in SelfDRSC++ can be interestingly formulated as a specialized instance of video frame interpolation, where each row in reconstructed RS images is interpolated from predicted GS images by utilizing RS distortion time maps. By achieving superior performance while simplifying the training process, SelfDRSC++ enables feasible one-stage self-supervised training. Additionally, besides start and end RS scanning time, SelfDRSC++ allows supervision of GS images at arbitrary intermediate scanning times, thus enabling the learned DRSC network to generate high framerate GS videos. 

### Examples of the Demo
https://github.com/user-attachments/assets/f9249552-d150-4dfa-ad7d-87d739244716

![r1](https://github.com/user-attachments/assets/11fac362-93d0-4030-8e92-076043cdafc3)

### Prerequisites
- Python >= 3.8, PyTorch >= 1.7.0
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm


### Datasets
Please download the RS-GOPRO datasets from [GoogleDrive](https://drive.google.com/file/d/1Txq0tU-1r3T2TjN-DQIe7YHyqwv9rCma/view) or [BaiduDisk](https://pan.baidu.com/s/1LNjrFYJJAUgt3H4ZUumOJw?pwd=vsad)(password: vsad).

## Dataset Organization Form
```
|--dataset
    |--train  
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ： 
        |--video 2
            :
        |--video n
    |--valid
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ：   
        |--video 2
         :
        |--video n
    |--test
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ：   
        |--video 2
         :
        |--video n
```

## Download Pre-trained Model of SelfDRSC++
Please download the pre-trained RIFE from [BaiduDisk](https://pan.baidu.com/s/1RjLN2yOix94hg7m35HIFPA?pwd=b4kg)(password:b4kg) or [GoogleDrive](https://drive.google.com/drive/folders/1x1JSjlNzL1LfrgqxaEakVHrQDxxNVUjB?usp=sharing). Please put these models to `./pretrained`.
Our results on the RS-GOPRO datasets and real demos can also be downloaded from [BaiduDisk](https://pan.baidu.com/s/1J9PjilYK522aEzoCsl96sg?pwd=gbfn)(password:gbfn).

## Getting Started
### 1) Testing
1.Testing on RS-GOPRO dataset:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 test.py --opt options/test_amt_rife_dr_rsflow_multi_psnr.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.

1.Testing on real RS data:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 test_real.py --opt options/test_amt_rife_dr_rsflow_multi_real.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.
If you test on your own data, remember to change `self.H` and `self.W` in `./data/dataset_rsgopro_self_real.py`， They correspond to the height and width of the images respectively. If you want to generate a higher frame rate, you can change `test {'frames'}` in the JSON file.

### 2) Training
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 train.py --opt options/train_amt_rife_dr_rsflow_multi_psnr.json --dist True
```
Please change `data_root` and `pretrained_rsg` in options according to yours.

## Cite
If you use any part of our code, or SelfDRSC++ is useful for your research, please consider citing:
```
@inproceedings{shang2023self,
  title={Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive},
  author={Shang, Wei and Ren, Dongwei and Feng, Chaoyu and Wang, Xiaotao and Lei, Lei and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13086--13094},
  year={2023}
}
```

## Contact
If you have any questions, please contact csweishang@gmail.com.

## Acknowledgements
This code is built on [SelfDRSC](https://github.com/shangwei5/SelfDRSC) and [InterpAny-Clearer](https://github.com/zzh-tech/InterpAny-Clearer). We thank the authors for sharing the codes.

