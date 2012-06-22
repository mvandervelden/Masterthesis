#!/usr/bin/env bash

CFGFILE=$1
LOGFILECFG=$2
TESTFILE=$3
python voc_cls.py $CFGFILE $LOGFILECFG $TESTFILE
NO_BATCHES=`cat $TESTFILE | sed -n '1p'`
NO_CLASSES=`cat $TESTFILE | sed -n '2p'`
CLASSES=(`cat $TESTFILE | tail -n +3`)
echo $NO_BATCHES
echo $NO_CLASSES
echo ${CLASSES[@]}
# Using 4 parallel processes, define the numbr of iterations per batch to cover
# all classes, and how many processes there should be in each iteration
NO_FULL_ITS=`echo $(($NO_CLASSES/4))`
FULL_ITS=`seq 1 $NO_FULL_ITS`
IT_SIZES=`for it in $FULL_ITS; do echo '4'; done; echo $(($NO_CLASSES%4))`

for B in `seq 1 $NO_BATCHES`; do
    echo "Running batch $B"
    START_CLS=0
    for SZ in $IT_SIZES; do
        echo "Running iteration of $SZ processes"
        STOP_CLS=$(($START_CLS+$SZ-1))
        for P in `seq $START_CLS $STOP_CLS`; do
            CLS=${CLASSES[$P]}
            echo "Running test on class no $P ($CLS)"
            NLOGCFG=`echo $LOGFILECFG | sed "s/.log/_$CLS.log/"`
            less $LOGFILECFG | sed "s/.log','w'/_$CLS.log',/" > $NLOGCFG
            python voc_cls_test.py $CFGFILE $B $CLS $NLOGCFG&
        done
        wait
        START_CLS=$(($START_CLS+$SZ))
    done
done