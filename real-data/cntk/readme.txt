# For single GPU
minibatchSize=4096 maxEpochs=40 deviceId=0 ./t.sh

# For multiple GPUs
# The real minibatch size for training is: $minibatchSize * $gpu_count
minibatchSize=2048 maxEpochs=40 deviceId=0,1 gpu_count=2 ./tm.sh
