## Semantic-Guidance
This repository contains code for our CVPR-2021 paper on [Combining Semantic Guidance and Deep Reinforcement Learning For Generating Human Level Paintings](https://arxiv.org/pdf/2011.12589.pdf).

The Semantic Guidance pipeline distills different forms of object awareness (semantic segmentation, object localization and guided backpropagation maps) into the painting process itself. The resulting agent is able to paint canvases with increased saliency of foreground objects and enhanced granularity of key image features.

<!-- ### Abstract
Generation of stroke-based non-photorealistic imagery, is an important problem in the computer vision community. As an endeavor in this direction, substantial recent research efforts have been focused on teaching machines "how to paint", in a manner similar to a human painter. However, the applicability of previous methods has been limited to datasets with little variation in position, scale and saliency of the foreground object. As a consequence, we find that these methods struggle to cover the granularity and diversity possessed by real world images. 

To this end, we propose a Semantic Guidance pipeline with **1)** a bi-level painting procedure for learning the distinction between foreground and background brush strokes at training time. **2)** We also introduce invariance to the position and scale of the foreground object through a neural alignment model, which combines object localization and spatial transformer networks in an end to end manner, to zoom into a particular semantic instance. **3)** The distinguishing features of the in-focus object are then amplified by maximizing a novel guided backpropagation based focus reward. The proposed agent does not require any supervision on human stroke-data and successfully handles variations in foreground object attributes, thus, producing much higher quality canvases for the CUB-200 Birds and Stanford Cars-196 datasets. Finally, we demonstrate the further efficacy of our method on complex datasets with multiple foreground object instances by evaluating an extension of our method on the challenging Virtual-KITTI dataset. -->
## Contents
* [Demo](#demo)
* [Environment Setup](#environment-setup)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Training](#training)
* [Citation](#citation)


## Demo
Our method shows high performance even when facing high variance in position, scale and saliency of the foreground objects.

| Target Image     | Baseline (Huang et al. 2019) | Semantic Guidance (Ours)  |
|:-------------:|:-------------:|:-------------:|
|<img src="assets/target_bird_5602.png" width="250" height="250"/>|<img src="./assets/bird_5602.gif" width="250" height="250" />|<img src="./assets/sg_bird_5602.gif" width="250" height="250"/>|
|<img src="assets/target_bird_4648.png" width="250" height="250"/>|<img src="./assets/bird_4648.gif" width="250" height="250"/>|<img src="./assets/sg_bird_4648.gif" width="250" height="250"/>|
|<img src="assets/target_bird_4008.png" width="250" height="250"/>|<img src="./assets/bird_4008.gif" width="250" height="250"/>|<img src="./assets/sg_bird_4008.gif" width="250" height="250"/>|



### Environment Setup

* Set up the python environment for running the experiments.
```bash
conda env update --name semantic-guidance --file environment.yml
conda activate semantic-guidance
```

### Dataset and Preprocessing
* Download [CUB-200-2011 Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html) dataset and place it in the `data/cub200/CUB_200_2011/` folder.
```bash
data
├── cub200/
│   └── CUB_200_2011/
│       └── images/
│             └── ...
│       └── images.txt
```

* Download the [pretrained model]() for object localization and semantic segmentation. Place it in the root directory of the repo.

* Run the preprocessing script to generate object localization, semantic segmentation and bounding box predictions.
```bash
cd semantic-guidance
python preprocess.py
```

* **OR** you can also directly download the preprocessed dataset from [here](), and place the prediction folders in the original data directory. The final data directory should look like:
```bash
data
├── cub200/
│   └── CUB_200_2011/
│       └── images/
│             └── ...
│       └── segmentations_pred/
│             └── ...
│       └── gbp_global/
│             └── ...
│       └── bounding_boxes_pred.txt
│       └── images.txt
```

### Training

* Train the baseline model from [Huang el al. 2019](https://arxiv.org/abs/1903.04411)
```bash
cd semantic-guidance
python train.py --dataset cub200 --debug --batch_size=96 --max_eps_len 50  --bundle_size=5 --exp_suffix baseline
```

* Train the deep reinforcement learning based painting agent using Semantic Guidance pipeline.
```bash
cd semantic-guidance
python train.py --dataset cub200 --debug --batch_size=96 --max_eps_len 50  --bundle_size=3 --use_gbp --use_bilevel --exp_suffix semantic-guidance
```

# Citation

If you use / discuss ideas from the semantic guidance pipeline in your work, please cite our paper:
```
@inproceedings{singh2021combining,
  title={Combining Semantic Guidance and Deep Reinforcement Learning For Generating Human Level Paintings},
  author={Jaskirat Singh and Liang Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

# Under Construction

This repository is under construction. Code and pretrained models would be added soon!
