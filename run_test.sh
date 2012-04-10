#!/bin/bash

tst=$1
idbase=$2
cfg=$3
iterations=$4

for it in `seq 1 $iterations`; do
	id=$idbase$it
	echo $id
	screen -dmS "iteration$it" ./run_single_iteration.sh $tst $id $cfg
done