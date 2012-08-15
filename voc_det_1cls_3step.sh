#!/bin/bash

echo "Running training"
CFGFILE=$1
TMPFOLDER=$2

# mkdir "$TMPFOLDER"
# 
# python voc_det_exemplar_train.py $CFGFILE $TMPFOLDER
NO_BATCHES=1
NO_CLASSES=1
CLS=object
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: $CLS"

echo "Running test on class no 1 ($CLS)"
python voc_det_exemplar_test.py $CFGFILE $TMPFOLDER 1 $CLS

# perform detection (clustering on all images) per image
for B in `seq 1 $NO_BATCHES`; do
    IMIDS=(`cat $TMPFOLDER/batch_$B.pkl.txt`)
    echo "Running batch $B on detection: img: ${IMIDS[@]}"
    echo "Running detection on class $CLS"
    for ID in ${IMIDS[@]}; do
        echo "Running detection on image $ID"
        python voc_det_exemplar_detect.py $CFGFILE $TMPFOLDER $B $CLS $ID
    done
done
    