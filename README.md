# BA3C-CPU

This is the repository containing the source code for our paper concerning playing Atari Games on CPU.

The tensorflow directory contains our fork of TensorFlow 0.11rc0, adapted to use MKL convolutions. It can be compiled from sources using the bazel build system.

The tensorpack directory contains our fork of the Tensorpack framework used for performing the experiments, along with the implementation of the BA3C algorithm for training agents for playing atari games.

In order to reproduce our results:

1. Download and install the MKL library from (https://software.intel.com/en-us/intel-mkl) (we used 2017 initial version)
2. Build and install our custom TensorFlow. Instructions for building TensorFlow from source can be found in: https://www.tensorflow.org/install/install_sources
3. Install tensorpack's dependencies as outlined in tensorpack/README.md
4. Add the tensorpack directory to PYTHONPATH
5. Run the training script to start training agents for playing atari games with:
```
python train-atari.py --mkl 1 --cpu 1 --queue_size 1 --my_sim_master_queue 128 --train_log_path logs --predict_batch_size 16 --do_train 1 --predict_batch_size 16  --simulator_procs 200 --env Breakout-v0 --nr_towers 4  --nr_predict_towers 5
```


