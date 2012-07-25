#!/bin/bash

echo "Running training"
CFGFILE=$1
CFGTESTFILE=$2
TMPFOLDER=$3
# mkdir "$TMPFOLDER"
# python voc_fgbg_det_exemplar.py $CFGFILE $TMPFOLDER
NO_BATCHES=`cat $TMPFOLDER/testinfo.txt | sed -n '1p'`
NO_CLASSES=`cat $TMPFOLDER/testinfo.txt | sed -n '2p'`
CLASSES=(`cat $TMPFOLDER/testinfo.txt | tail -n +3`)
THREADS=1
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: ${CLASSES[@]}"


if [ $THREADS -eq 1 ]; then
    # Only one thread
    for B in `seq 1 $NO_BATCHES`; do
        echo "Running batch $B"
        START_CLS=0
        for P in `seq 0 $(($NO_CLASSES-1))`; do
            CLS=${CLASSES[$P]}
            echo "Running test on class no $P ($CLS)"
            python voc_fgbg_det_exemplar_test.py $CFGTESTFILE $TMPFOLDER $B $CLS
        done
    done
else
    # Using $THREADS parallel processes, define the numbr of iterations per batch to cover
    # all classes, and how many processes there should be in each iteration
    NO_FULL_ITS=`echo $(($NO_CLASSES/$THREADS))`
    FULL_ITS=`seq 1 $NO_FULL_ITS`
    IT_SIZES=`for it in $FULL_ITS; do echo $THREADS; done; echo $(($NO_CLASSES%$THREADS))`

    for B in `seq 1 $NO_BATCHES`; do
        echo "Running batch $B"
        START_CLS=0
        for SZ in $IT_SIZES; do
            echo "Running iteration of $SZ processes"
            STOP_CLS=$(($START_CLS+$SZ-1))
            for P in `seq $START_CLS $STOP_CLS`; do
                CLS=${CLASSES[$P]}
                echo "Running test on class no $P ($CLS)"
                python voc_fgbg_det_exemplar_test.py $CFGTESTFILE $TMPFOLDER $B $CLS&
            done
            wait
            START_CLS=$(($START_CLS+$SZ))
        done
    done
fi