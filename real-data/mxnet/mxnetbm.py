import argparse
import os, sys
import time
import subprocess

# Parse arguments
current_time = time.ctime()
parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
        help='Name of log file, default= mxnet_ + current time + .log') 
parser.add_argument('-batchSize', type=str, default='64', help='Batch size for each GPU, default = 64')
parser.add_argument('-network', type=str, default='fcn5', help='name of network[fcn5 | alexnet | resnet | lstm32 | lstm64]')
parser.add_argument('-devId', type=str, help='CPU: -1, GPU:0,1,2,3(Multiple gpu supported)')
parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
parser.add_argument('-numThreads', type=str, default='8', help='number of Threads, default=8')
parser.add_argument('-hostFile', type=str, help='path to running hosts(config in host file) for multiple machine training.')

args = parser.parse_args()
#print(args)

# Build cmd
exePath = ""
numSamples = 0
cmd = "cd "
pyscript=""
network = args.network
if network == "fcn5":
    numSamples = 60000
    exePath = "fc/"
    cmd += "fc; "
    pyscript = "python train_mnist.py --lr 0.05" 
elif network == "alexnet" or network == "resnet":
    numSamples = 50000
    exePath = "cnn/"
    cmd += "cnn; "
    pyscript = "python train_cifar10_" + network + ".py --lr 0.01"
elif "lstm" in network:
    exePath = "rnn/"
    cmd += "rnn; "
    pyscript = "python train_rnn.py --lr 0.1"
    if "64" in network:
        pyscript += " --sequence-lens 64"
    else:
        pyscript += " --sequence-lens 32"
else:
    print("Unknown network type " + network + ", supported ones: fcn, alexnet, resnet, lstm32, lstm64")
    sys.exit(-1)

numNodes = 1 if args.hostFile is None else int(subprocess.check_output("cat " + args.hostFile + " | wc -l", shell=True).strip())
if args.hostFile is not None:
    cmd += "PS_VERBOSE=0 nohup python ../multi-nodes-support/launch.py --launcher ssh -n " + str(numNodes) + " -s 1 -H " + args.hostFile + " " + pyscript
    cmd += " --kv-store dist_sync" + " --num-nodes " + str(numNodes) + " "
else:
    cmd += pyscript

nGPU = len(args.devId.split(','))
batchSize = int(args.batchSize)*nGPU
numEpochs = args.numEpochs
cmd += " --batch-size " + str(batchSize) + " --num-epochs " + numEpochs

devId = args.devId
if devId is not None:
    if "-" not in devId:
        cmd += " --gpus " + devId 
	if "--kv-store" not in cmd:
            cmd += " --kv-store device"
    elif "-1" == devId:
        os.environ["MXNET_CPU_WORKER_NTHREADS"] = args.numThreads
        print os.environ["MXNET_CPU_WORKER_NTHREADS"]
    else:
        print("invalid devId!")
        sys.exit(-1)

else:
    print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
    sys.exit(-2)


logfile = args.log
if ".log" not in logfile:
    logfile += ".log"

#cmd += " &> " + logfile
cmd += " >& " + logfile
t = time.time()
#print cmd
os.system(cmd)
t = time.time() - t
#print("Time diff: " + str(t))
logPath = exePath + logfile
catLog = "cat " + logPath
totalEpochTime = subprocess.check_output( catLog + " | grep Time | cut -d\'=\' -f2 | paste -sd+ - | bc", shell=True)
numEpoch = subprocess.check_output(catLog + " | grep Time | cut -d\'=\' -f2 | wc -l", shell=True)
#print totalEpochTime
#print numEpoch
avgEpoch = 0
if int(numEpoch) != 0:
    avgEpoch = float(totalEpochTime)/float(numEpoch)
#print avgEpoch
if "lstm" in network:
    numSamples = int(subprocess.check_output(catLog + " | grep \"len of data train\" | cut -d' ' -f5", shell=True))
avgBatch = (avgEpoch/numSamples)*float(batchSize)
#print("Avg Batch: " + str(avgBatch))

if "lstm" not in network:
    valAccuracy = subprocess.check_output(catLog + "| grep Validation-a | cut -d'=' -f2", shell=True).strip().split('\n')
    #print valAccuracy
    trainCE = subprocess.check_output(catLog + "| grep Train-cross-entropy | cut -d'=' -f2", shell=True).strip().split('\n')
    #print trainCE
    
    info = ""
    for i in range(len(trainCE)/numNodes):
	valAcu = 0
	train_ce = 0
	for j in range(numNodes):
	    #valAcu += float(valAccuracy[i*numNodes + j])
	    train_ce += float(trainCE[i*numNodes + j])
        if i != 0:
            info += ","
        #info += str(i) + ":" + str(valAcu/numNodes) + ":" + str(train_ce/numNodes) 
        info += str(i) + ":" + "-" + ":" + str(train_ce/numNodes) 
    #print info 
    print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info
else:
    valPerplexity = subprocess.check_output(catLog + " | grep Val | cut -d'=' -f2", shell=True).strip().split('\n')
    trainPerplexity = subprocess.check_output(catLog + " | grep Speed | cut -d'=' -f2", shell=True).strip().split('\n')
    info = " -I "
    for i in range(len(valPerplexity)):
        if i != 0:
            info += ","
        info += str(i) + ":" + valPerplexity[i] + ":" + trainPerplexity[i]
    #print info
    print "-t " + str(t) + " -a " + str(avgBatch) + info

with open(logPath, "a") as logFile:
    logFile.write("Total time: " + str(t) + "\n")
    logFile.write("cmd: " + cmd + "\n")
os.system("cp " + logPath + " ../../logs")
