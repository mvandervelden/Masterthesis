#!/usr/bin/env bash

echo "Running training"
CFGFILE=$1
LOGFILECFG=$2
TMPFOLDER=$3
python voc_cls.py $CFGFILE $LOGFILECFG $TMPFOLDER
mkdir "/local/vdvelden/$TMPFOLDER"
TMPFOLDER="/local/vdvelden/$TMPFOLDER"
NO_BATCHES=`cat $TMPFOLDER/testfile.txt | sed -n '1p'`
NO_CLASSES=`cat $TMPFOLDER/testfile.txt | sed -n '2p'`
CLASSES=(`cat $TMPFOLDER/testfile.txt | tail -n +3`)
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: ${CLASSES[@]}"

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
            python voc_cls_test.py $CFGFILE $TMPFOLDER $B $CLS $NLOGCFG&
        done
        wait
        START_CLS=$(($START_CLS+$SZ))
    done
done