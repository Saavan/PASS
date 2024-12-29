#!/bin/bash 
NUM_RUNS=10000000
ARR_DIMS=(2 4 6 8 10 12 14 16)

#Baseline power profiler for 1 second
AMDuProfCLI timechart --interval 100 -e Power -o time_series --duration 1

for dim in "${ARR_DIMS[@]}"
do
   echo "PASS Num Runs:$NUM_RUNS Array Size:$dim" | tee log.txt
   AMDuProfCLI timechart --interval 50 -e Power -o time_series ./pass.out $dim $NUM_RUNS | tee log.txt
done
