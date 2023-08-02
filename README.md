# ScanBot: Autonomous Reconstruction via Deep Reinforcement Learning
autoscanning system capable of exploring, reconstructing, and understanding an unknown scene within *one navigation pass* by using deep reinforcement learning.

## Install Dependencies
This code is tested on **Ubuntu 20.04**. We use earlier versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab). The specific commits are mentioned below.

Installing habitat-sim:
```shell
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout 011191f65f37587f5a5452a93d840b5684593a00;
pip install -r requirements.txt; 
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
python setup.py install --headless --with-cuda
```

Installing habitat-lab:
```shell
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout c5729353e8ec4a4058de55845cec371994f6a7f1;
pip install -r requirements.txt; 
python setup.py develop --all
```

Installing Open3d:
```shell
git clone https://github.com/isl-org/Open3D
# Only needed for Ubuntu
util/install_deps_ubuntu.sh
cd Open3d
git clone https://github.com/isl-org/Open3D-ML.git
mkdir build && cd build
cmake -DBUILD_CUDA_MODULE=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=$PWD/../Open3D-ML \
      -DUSE_SYSTEM_JPEG=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DBUILD_CACHED_CUDA_MANAGER=OFF \
 ..
 make -j install-pip-package
```

Other requirements include:
* [cuda](https://developer.nvidia.com/cuda-downloads)
* [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html)

## Download and Installation
Firstly, you need to build pybind `.so` library for path_finder and channels_constructor
After executing following commands, a `astar_path_finder.cpython-XXm-x86_64-linux-gnu.so` and a `channels_constructor.cpython-XXm-x86_64-linux-gnu.so` library will appear in `habitat_scanbot2d`, in which `XX` is your python version.
```shell
git clone https://github.com/HezhiCao/scanbot.git
cd scanbot
pip install -e .
cd habitat_scanbot2d/torch_extensions
python setup.py build
cd .. # suppose you are in habitat_scanbot2d
ln -s torch_extensions/build/lib.linux-x86_64-cpython-37/channels_constructor.cpython-XXm-x86_64-linux-gnu.so .
```
The code requires datasets in a data folder in the following format (same as habitat-lab):
```
habitat_scanbot2d/
  data/
    scene_datasets/
      mp3d_gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        mp3d_gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-lab#datasets

Then regenerate navmesh to forbid cross-floor navigation:
```
cd habitat_scanbot2d
python scripts/navmesh_generator.py -d data/scene_datasets/mp3d_gibson
```

## Train & Test
### GLobal Policy
After training, the model will be saved in `data/new_checkpoints`, as well as training logs in `train.log` file
```shell
python3 global_training.py
```
Multiple GPUs
```
cd habitat_scanbot2d
./train.sh
```

You can investigate some statistics using `evaluation`
```shell
python3 global_evaluation.py
```
### Local Policy
We will update the local policy soon.

## Citation
If you use this code for your research, please cite our paper:
```
@article {cao2023scanbot,
    author   = {Hezhi Cao, Xi Xia, Guan Wu, Ruizhen Hu, and Ligang Liu},
    title    = {ScanBot: Autonomous Reconstruction via Deep Reinforcement Learning},
    journal  = {ACM Transactions on Graphics (SIGGRAPH 2023)},
    volume   = {42},
    number   = {4},
    pages    = {Article 235},
    year     = {2023}
    }
```

## Acknowledgement

We thank the anonymous reviewers for their valuable comments and suggestions. This work is supported by the National Key R&D Program of China (2022YFB3303400), National Natural Science Foundation of China (62025207), Guangdong Natural Science Foundation (2021B1515020085), and Shenzhen Science and Technology Program (RCYX20210609103121030).

## Contact

If you have some ideas or questions about our research to share with us, please contact caohezhi21@mail.ustc.edu.cn
