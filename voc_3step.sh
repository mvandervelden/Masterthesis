#!/bin/bash

echo "Running training"
CFGFILE=$1
CFGTESTFILE=$2
TMPFOLDER=$3

mkdir "$TMPFOLDER"

python voc_fgbg_det_exemplar.py $CFGFILE $TMPFOLDER
NO_BATCHES=`cat $TMPFOLDER/testinfo.txt | sed -n '1p'`
NO_CLASSES=`cat $TMPFOLDER/testinfo.txt | sed -n '2p'`
CLASSES=(`cat $TMPFOLDER/testinfo.txt | tail -n +3`)
THREADS=4
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: ${CLASSES[@]}"

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

# perform detection (clustering on all images) per image
for B in `seq 1 $NO_BATCHES`; do
    IMIDS=(`cat $TMPFOLDER/batch_$B.pkl.txt`)
    echo "Running batch $B on detection: img: ${IMIDS[@]}"
    for CLS in $CLASSES; do
        echo "Running detection on class $CLS"
        for ID in $IMIDS; do
            python voc_fgbg_det_exemplar_detect.py $CFGTESTFILE $TMPFOLDER $B $CLS $ID
        done
    done
done
    