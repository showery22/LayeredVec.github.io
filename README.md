# Layered Image Vectorization via Semantic Simplification

### [Project Page](https://szuviz.github.io/layered_vectorization/)&ensp;&ensp;&ensp;[Paper](https://arxiv.org/abs/2406.05404)

<!-- > <a href="/"> Zhenyu Wang</a>, -->
>Zhenyu Wang,
Jianxi Huang</a>,
<a href="https://zhdsun.github.io/">Zhida Sun</a>,
Yuanhao Gong,
<a href="https://danielcohenor.com/">Daniel Cohen-Or</a>,
<a href="https://deardeer.github.io/">Min Lu</a>
> <br>
<div>
  <img src="static/images/layered6.png" alt="teaser" width="900" height="auto">
</div>

<!-- > <p>This work presents a novel progressive image vectorization technique aimed at generating layered vectors that represent the original image from coarse to fine detail levels. Our approach introduces semantic simplification, which combines Score Distillation Sampling and semantic segmentation to iteratively simplify the input image. Subsequently, our method optimizes the vector layers for each of the progressively simplified images. Our method provides robust optimization, which avoids local minima and enables adjustable detail levels in the final output. The layered, compact vector representation enhances usability for further editing and modification. 
</p> -->
> <p>This work presents a progressive image vectorization technique that reconstructs the raster image as layer-wise vectors from semantic-aligned macro structures to finer details. Our approach introduces a new image simplification method leveraging the feature-average effect in the Score Distillation Sampling mechanism, achieving effective visual abstraction from the detailed to coarse. Guided by the sequence of progressive simplified images, we propose a two-stage vectorization process of structural buildup and visual refinement, constructing the vectors in an organized and manageable manner. The resulting vectors are layered and well-aligned with the target image's explicit and implicit semantic structures. Our method demonstrates high performance across a wide range of images. Comparative analysis with existing vectorization methods highlights our technique's superiority in creating vectors with high visual fidelity, and more importantly, achieving higher semantic alignment and more compact layered representation. The project homepage is https://szuviz.github.io/layered_vectorization/.
</p>

## Installation
<!-- We suggest users to use the conda for creating new python environment. 

**Requirement**: 5.0<GCC<6.0;  nvcc >10.0. -->

```bash
git clone https://github.com/showery22/LayeredVec.github.io.git
cd LayeredVec.github.io
conda create -n lv python=3.10
conda activate lv
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```
```bash
cd ..
cd LayeredVectorization
pip install -r requirements.txt
```
## Model Checkpoints
- **`SAM`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- **`SD`: "runwayml/stable-diffusion-v1-5"**

## Run
```bash
conda activate lv
cd LayeredVectorization
python main.py --config config/base_config.yaml --target_image target_imgs/robot.png --file_save_name robot
```

## Reference

    @article{wang2024layered,
        title={Layered Image Vectorization via Semantic Simplification},
        author={Wang, Zhenyu and Huang, Jianxi and Sun, Zhida and Gong, Yuanhao and Cohen-Or, Daniel and Lu, Min},
        journal={arXiv preprint arXiv:2406.05404},
        year={2024}
    }


