# BA3C-CPU

This is the repository containing the source code for our paper concerning playing Atari Games on CPU.

The tensorflow directory contains our fork of TensorFlow 0.11rc0, adapted to use MKL convolutions. It can be compiled from sources using the bazel build system or installed from the wheel provided in the tensorflow-0.11.0rc0-py2-none-any.whl file.

The tensorpack directory contains our fork of the Tensorpack framework used for performing the experiments, along with the implementation of the BA3C algorithm for training agents for playing atari games.

In order to reproduce our results:

1. Download the MKL library from (https://software.intel.com/en-us/intel-mkl) (we used 2017 initial version)
2. Install it in the /usr/local/intel/compilers_and_libraries/linux/mkl/ directory
3. set MKL env variables with:
```
source /usr/local/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64
```
4. Set runtime library path to openmp library:
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/intel/compilers_and_libraries_2017.0.098/linux/compiler/lib/intel64_lin/
```
5. Install our custom TensorFlow wheel:
```
pip install tensorflow-0.11.0rc0-py2-none-any.whl
```
6. Verify that the TensorFlow has been correctly installed with: 
```
python -c 'import tensorflow as tf; print tf.__version__'
```
7. Install the dependencies from the requirements.txt file by:
```
pip install -r requirements.txt
```
8. Add the tensorpack directory to PYTHONPATH:
```
export PYTHONPATH=tensorpack/
```
9. Run the training script to start training agents for playing atari games with:
```
python tensorpack/examples/OpenAIGym/train-atari.py --mkl 1 --cpu 1 --queue_size 1 --my_sim_master_queue 128 --train_log_path logs --predict_batch_size 16 --do_train 1 --predict_batch_size 16  --simulator_procs 200 --env Breakout-v0 --nr_towers 4  --nr_predict_towers 5
```
