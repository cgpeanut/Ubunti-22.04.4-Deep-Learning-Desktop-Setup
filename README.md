# Ubuntu 22.04 for Deep Learning

Machine setup
Install Ubuntu 22.04 minimal 
Install and configure openssh server
Update Ubuntu
Create Update Script
Install NVIDIA Drivers for Deep Learning
Install cuDNN v8.9.4, for CUDA 12.x

Virtual Environment Setup
Machine Learning Environment
Deep Learning Environment (TensorFlow-GPU)
Deep Learning Environment (PyTorch-GPU)
Run Very Large Language Models on Your Computer

----------------------------------------------------------------------------------------------------

**Machine Setup**

**Install Ubuntu 22.04.4**:

- Computer name: Buruha
- Name: Roberto Roxas
- User name: roxasrr
- Password: ********

----------------------------------------------------------------------------------------------------

**Install and configure openssh server**:

```
$ sudo apt update
$ sudo apt install openssh-server
$ sudo systemctl enable --now ssh
$ sudo systemctl status ssh
$ sudo ufw allow ssh
$ sudo ufw enable && sudo ufw reload

$ git config --global user.name "Your Name"
$ git config --global user.email "youremail@yourdomain.com"
```

**Update Ubuntu**:

```
$ sudo apt update
$ sudo apt full-upgrade --yes
$ sudo apt autoremove --yes
$ sudo apt autoclean --yes
$ sudo reboot
```
----------------------------------------------------------------------------------------------------

**Install NVIDIA Drivers for Deep Learning**:

Check Display Hardware:

```
$ sudo lshw -C display
```

----------------------------------------------------------------------------------------------------

Install NVIDIA GPU Driver:

- Install from GUI: Software & Updates > Additional Drivers > NVIDIA > nvidia-driver-535 (proprietary,tested)

Try $ sudo ubuntu-drivers autoinstall if NVIDIA drivers are disabled.

You can also install it from the terminal:

I installed CUDA Toolkit 12.4 deb (local)

```
$ sudo apt install nvidia-driver-535
```

**Check TensorFlow and CUDA Compatibilities:**:

- https://www.tensorflow.org/install/gpu
- https://www.tensorflow.org/install/source#gpu

----------------------------------------------------------------------------------------------------

**Install CUDA Toolkit (CUDA 12.0):**: (in my case I installed 11.8)

1. Install prerequisites:

```
$ sudo apt install build-essential

$ sudo apt install linux-headers-$(uname -r)
```

2. Download CUDA 12.0 (https://developer.nvidia.com/cuda-toolkit-archive)

```
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
```

3. Install CUDA 12.0:

```
$ sudo sh cuda_12.0.0_525.60.13_linux.run — override (without Driver)
```

4. Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables (Add following lines to ~/.bashrc):

```
export PATH=$PATH:/usr/local/cuda-12.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64:/usr/local/cuda-12.0/extras/CUPTI/lib64
```

5. Check that GPUs are visible using the following command (A reboot may be required):

```
$ nvidia-smi
```
----------------------------------------------------------------------------------------------------

**Install cuDNN v8.9.4, for CUDA 12.x:**:

https://developer.nvidia.com/cudnn
https://developer.nvidia.com/rdp/cudnn-archive

Login & Download:

```
$ sudo dpkg -i cudnn-local-repo-ubuntu2204–8.6.0.163_1.0–1_amd64.deb

$ sudo cp cudnn-local-repo-ubuntu2204–8.9.5.29/cudnn-local-275FA572-keyring.gpg /usr/share/keyrings/

$ sudo apt update

$ sudo apt install libcudnn8

$ sudo apt install libcudnn8-dev

$ sudo apt install libcudnn8-samples # Optional

Reboot:

$ reboot
```
----------------------------------------------------------------------------------------------------
**Virtual Environment Setup**

**Machine Learning Environment**

```
$ python3 -m venv ~/venvs/ml

$ source ~/venvs/ml/bin/activate

(ml) $ pip install --upgrade pip setuptools wheel

(ml) $ pip install --upgrade numpy scipy matplotlib ipython jupyter pandas sympy nose

(ml) $ pip install --upgrade scikit-learn scikit-image

(ml) $ deactivate

```
**Deep Learning Environment (TensorFlow-GPU)**

https://www.tensorflow.org/install/gpu

```
$ sudo apt install python3.10-venv

$ python3 -m venv ~/venvs/tfgpu

$ source ~/venvs/tfgpu/bin/activate

(tfgpu) $ python3 -m pip install tensorflow[and-cuda]

# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

(tfgpu) $ pip install --upgrade pip setuptools wheel

(tfgpu) $ pip install --upgrade opencv-python opencv-contrib-python

(tfgpu) $ pip install --upgrade tensorflow tensorboard keras

(tfgpu) $ deactivate

Verification:

$ source ~/venvs/tfgpu/bin/activate

(tfgpu) $ python

->>> from tensorflow.python.client import device_lib

->>> device_lib.list_local_devices()

->>> exit()

(tfgpu) $ deactivate
```

**Deep Learning Environment (PyTorch-GPU)**

```
https://pytorch.org/get-started/locally/

$ python3 -m venv ~/venvs/torchgpu

$ source ~/venvs/torchgpu/bin/activate

(torchgpu) $ pip install --upgrade pip setuptools wheel

(torchgpu) $ pip install --upgrade opencv-python opencv-contrib-python

(torchgpu) $ pip install --upgrade torch torchvision torchaudio

(torchgpu) $ deactivate

Verification:

$ source ~/venvs/torchgpu/bin/activate

(torchgpu) $ python

->>> import torch

->>> torch.cuda.is_available()

->>> exit()

(torchgpu) $ deactivate

```
**Run Very Large Language Models on Your Computer**

With PyTorch and Hugging Face’s device_map

The following command creates an environment named “device_map” and activates it.

```
conda create -n device_map python=3.9

conda activate device_map

Then, install the following package. Note that it may also work with higher or lower versions of CUDA.

conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia

conda install -c conda-forge transformers

conda install -c conda-forge accelerate

```
