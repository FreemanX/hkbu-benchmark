# For single GPU
batch_size=4096 epochs=40 deviceId=0 ./t.sh

# For multiple GPUs
# The real minibatch size for training is: $minibatchSize * $gpu_count
batch_size=2048 epochs=40 deviceId=0,1 gpu_count=2 ./tm.sh
