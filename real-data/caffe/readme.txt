# Before running experiments
# Execute the bash script to generate teh prototxt files
./gen-$network.sh

# For one GPU
caffe train -solver=$network-b$batchSize-GPU-solver.prototxt -gpu=0

# For multiple GPUs
# Real minibatch size is: $batchSize * $gpu_count
caffe train -solver=$network-b$batchSize-GPU-solver.prototxt -gpu=0,1
