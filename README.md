# BA3C-CPU

This is the repository containing the source code for our paper concerning playing Atari Games on CPU.

The tensorflow directory contains our fork of TensorFlow 0.11rc0, adapted to use MKL convolutions. It can be compiled from sources using the bazel build system.

The tensorpack directory contains our fork of the Tensorpack framework used for performing the experiments, along with the implementation of the BA3C algorithm for training agents for playing atari games.

In order to reproduce our results:

1. Download the MKL library from (https://software.intel.com/en-us/intel-mkl) (we used 2017 initial version)
2. Install it in the /usr/local/intel/compilers_and_libraries/linux/mkl/ directory
3. set MKL env variables with:
```
source /usr/local/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh mic
```
4. Install our custom TensorFlow wheel:
```
pip install tensorflow-mkl-0.11.0rc0-py2-none-any.whl
```
5. Verify that the TensorFlow has been correctly installed with: 
```
python -c 'import tensorflow as tf; print tf.__version__'
```
6. Install the dependencies from the requirements.txt file by:
```
pip install -r requirements.txt
```
7. Add the tensorpack directory to PYTHONPATH:
```
export PYTHONPATH=tensorpack/
```
8. Run the training script to start training agents for playing atari games with:
```
python train-atari.py --mkl 1 --cpu 1 --queue_size 1 --my_sim_master_queue 128 --train_log_path logs --predict_batch_size 16 --do_train 1 --predict_batch_size 16  --simulator_procs 200 --env Breakout-v0 --nr_towers 4  --nr_predict_towers 5
```
