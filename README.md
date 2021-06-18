## Semantic-Guidance: Distilling Object Awareness into Paintings [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Automatically%20generate%20human-level%20paintings%20using%20a%20combination%20of%20Deep-RL%20and%20Semantic-Guidance&url=https://github.com/1jsingh/semantic-guidance&&hashtags=LearningToPaint,CVPR2021)

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
* [Testing using Pretrained Models](#testing-using-pretrained-models)
* [Citation](#citation)


## Demo
Traditional reinforcement learning based methods for the "*learning to paint*" problem, show poor performance on real world datasets with high variance in position, scale and saliency of the foreground objects. To address this we propose a semantic guidance pipeline, which distills object awareness knowledge into the painting process, and thereby learns to generate semantically accurate canvases under adverse painting conditions.

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

* Download differentiable neural renderer: [renderer.pkl](https://anu365-my.sharepoint.com/:u:/g/personal/u7019589_anu_edu_au/EWdoJgGzUJtEt1Qc_LkS9DwBj-bwem_I2BMT-W4VzcEuNw?e=3MUUBL) and place it in the `data/.` folder.

* Download combined model for object localization and semantic segmentation from [here](https://anu365-my.sharepoint.com/:u:/g/personal/u7019589_anu_edu_au/EbzRihTmKhtAjlXW-U5l8sUB751nZDGJQ4qXF4dk2wVV3A?e=yO0WGM), and place it in place it in the `data/.` folder.

* Run the preprocessing script to generate object localization, semantic segmentation and bounding box predictions.
```bash
cd semantic_guidance
python preprocess.py
```

* **OR** you can also directly download the preprocessed birds dataset from [here](https://anu365-my.sharepoint.com/:u:/g/personal/u7019589_anu_edu_au/EY0RrfqyE2FEsaWFyBm5Mt4BUDc8M7d7XjarBKsU3SXqEQ), and place the prediction folders in the original data directory. The final data directory should look like:
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
└── renderer.pkl
└── birds_obj_seg.pkl
```

### Training

* Train the baseline model from [Huang et al. 2019](https://arxiv.org/abs/1903.04411)
```bash
cd semantic_guidance
python train.py \
--dataset cub200 \
--debug \
--batch_size=96 \
--max_eps_len=50  \
--bundle_size=5 \
--exp_suffix baseline
```

* Train the deep reinforcement learning based painting agent using Semantic Guidance pipeline.
```bash
cd semantic_guidance
python train.py \
--dataset cub200 \
--debug \
--batch_size=96 \
--max_eps_len=50  \
--bundle_size=5 \
--use_bilevel \
--use_gbp \
--exp_suffix semantic-guidance
```

### Testing using Pretrained Models

* Download the pretrained models for the [Baseline](https://anu365-my.sharepoint.com/:u:/g/personal/u7019589_anu_edu_au/EWoJ8_jprlRNvZegNCnBEnkBiLNG3SQKXPm119yJjB1mVg?e=HzK4sA) and [Semantic Guidance](https://anu365-my.sharepoint.com/:u:/g/personal/u7019589_anu_edu_au/EeUqajWhphlOg3EzXC8qYc8B090FMd2GYHSmjCdJ6bnmnA?e=cOTIHJ) agents. Place the downloaded models in `./semantic_guidance/pretrained_models` directory.
```bash
semantic-guidance
├── semantic_guidance/
│   └── pretrained_models/
│       └── actor_baseline.pkl
│       └── actor_semantic_guidance.pkl
```

* Generate the painting sequence using pretrained baseline agent.
```bash
cd semantic_guidance
python test.py \
--img ../input/target_bird_4648.png \
--actor pretrained_models/actor_baseline.pkl \
--use_baseline
```

* Use the pretrained Semantic Guidance agent to paint canvases.
```bash
cd semantic_guidance
python test.py \
--img ../input/target_bird_4648.png \
--actor pretrained_models/actor_semantic_guidance.pkl 
```

* The `test` script stores the final canvas state in the `./output` folder and saves a video for the painting sequence in the `./video` directory.


# Citation

If you find this work useful in your research, please cite our paper:
```
@inproceedings{singh2021combining,
  title={Combining Semantic Guidance and Deep Reinforcement Learning For Generating Human Level Paintings},
  author={Jaskirat Singh and Liang Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

<!-- # Under Construction

This repository is under construction. Code and pretrained models would be added soon! -->
