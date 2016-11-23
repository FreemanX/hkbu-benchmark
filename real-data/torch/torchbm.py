import argparse
import os, sys
import time
import subprocess

# Parse arguments
current_time = time.ctime()
parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
        help='Name of log file, default= mxnet_ + current time + .log') 
parser.add_argument('-batchSize', type=str, default='64', help='Batch size in each GPU, default = 64')
parser.add_argument('-network', type=str, default='fcn5', help='name of network[fcn5 | alexnet | resnet | lstm32 | lstm64]')
parser.add_argument('-devId', type=str, help='CPU: -1, GPU: 0,1,2,3 (Multiple gpu supported)')
parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
parser.add_argument('-numThreads', type=str, default='8', help='number of threads for ffn on CPU, default=8')

args = parser.parse_args()
#print(args)

# Build cmd
cmd = "THC_CACHING_ALLOCATOR=1 th Main.lua "
network = args.network
numSamples = 0
if network == "fcn5":
    cmd += "-LR 0.05 -dataset MNIST -network ffn5"
    numSamples = 50000
elif network == "alexnet" or network == "resnet":
    cmd += "-network " + network + " -LR 0.01 "
    numSamples = 60000
elif "lstm" in network:
	if args.devId is not None:
		if "-" not in args.devId:
			cmd = "THC_CACHING_ALLOCATOR=1 CUDA_VISIBLE_DEVICES=" + args.devId  + " th rnn/recurrent-language-model.lua --cuda " 
		else:
			print("Only GPU supproted!")
			sys.exit(-2)
	else:
		print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
		sys.exit(-2)
	if "64" in network:
		cmd += " --seqlen 64 "
	else:
		cmd += " --seqlen 32 "
	cmd += "--lstm --hiddensize '{256,256}' --startlr 1 --minlr 1 " 
	cmd += "--batchsize " + args.batchSize + " --maxepoch " + args.numEpochs
	logfile = args.log
	if ".log" not in logfile:
		logfile += ".log"
	cmd += " >& " + logfile
	#print cmd
	t = time.time()
	os.system(cmd)
	t = time.time() - t
	#print "total time: " + str(t)
	catLog = "cat " + logfile
	totalEpochBatchTime = subprocess.check_output( catLog + " | grep Speed | cut -d':' -f2 | paste -sd+ - | bc", shell=True).strip()
	numEpoch = subprocess.check_output(catLog + " | grep Speed | cut -d':' -f2 | wc -l", shell=True).strip()
	avgBatch = float(totalEpochBatchTime)/float(numEpoch)
	avgBatch = avgBatch/1000.0
	#print("Avg Batch: " + str(avgBatch))
	trainPPL = subprocess.check_output(catLog + "|grep \"Training PPL\" | cut -d':' -f2", shell=True).replace("	"," ").strip().split("\n")
	valPPL = subprocess.check_output(catLog + "|grep \"Validation PPL\" | cut -d':' -f2", shell=True).replace("	"," ").strip().split("\n")
	#print trainPPL
	#print valPPL
	info = " -I "
	for i in range(int(numEpoch)):
		if i != 0:
			info += ","
		info += str(i) + ":" + valPPL[i].strip() + ":" + trainPPL[i].strip()
	print " -t " + str(t) + " -a " + str(avgBatch) + info
	with open(logfile, "a") as logFile:
		logFile.write("Total time: " + str(t) + "\n")
	os.system("cp " + logfile + " ../../logs")
	sys.exit(0)
else:
    print("Unknown network type " + network + ", supported ones: fcn5, alexnet, resnet, lstm32, lstm64")
    sys.exit(-1)


devId = args.devId
nGPU = len(devId.split(','))
if devId is not None:
    if "-" not in devId:
        if nGPU > 1:
            cmd += " -nGPU " + str(nGPU)
        cmd = "CUDA_VISIBLE_DEVICES=" + devId + " " + cmd
    elif "-1" == devId and network == "ffn":
        cmd += "_cpu -nGPU 0 -type double -threads " + args.numThreads
    else:
        print("Only CNN is not supported on CPU in torch")
        sys.exit(-2)
else:
    print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
    sys.exit(-2)

batchSize = int(args.batchSize)*nGPU
numEpochs = args.numEpochs
cmd += " -batchSize " + str(batchSize) + " -epoch " + numEpochs

logfile = args.log
if ".log" not in logfile:
    logfile += ".log"

cmd += ' -logfile ' + logfile + ' -save ' + logfile + " >& display.tmp"
cmd = 'THC_CACHING_ALLOCATOR=1 ' + cmd
print(cmd)
t = time.time()
os.system(cmd)
t = time.time() - t
#print("Time diff: " + str(t))
os.system("rm display.tmp")
getInfo = 'cat ' + logfile + ' | grep Info'
totalEpochTime = subprocess.check_output( getInfo + " | grep time | cut -d' ' -f6 | cut -d':' -f2 | paste -sd+ - | bc", shell=True)
numEpoch = subprocess.check_output(getInfo + " | grep time | cut -d' ' -f6 | cut -d':' -f2 | wc -l", shell=True)
#print totalEpochTime
#print numEpoch
avgEpoch = 0
if int(numEpoch) != 0:
    avgEpoch = float(totalEpochTime)/float(numEpoch)
avgBatch = (avgEpoch/numSamples)*float(batchSize)
#print("Avg Batch: " + str(avgBatch))

valAccuracy = subprocess.check_output(getInfo + "| grep accu | cut -d' ' -f7 | cut -d':' -f2", shell=True).strip().split('\n')
#print valAccuracy
trainCE = subprocess.check_output(getInfo + "| grep Loss | cut -d' ' -f7 | cut -d':' -f2", shell=True).strip().split('\n')
#print trainCE

info = ""
for i in range(len(valAccuracy)):
    if i != 0:
        info += ","
    info += str(i) + ":" + valAccuracy[i] + ":" +trainCE[i]
#print info 
print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info
with open(logfile, "a") as logFile:
	logFile.write("Total time: " + str(t) + "\n")
os.system("cp " + logfile + " ../../logs")
