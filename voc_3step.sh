#!/bin/bash

echo "Running training"
CFGFILE=$1
TMPFOLDER=$2

# mkdir "$TMPFOLDER"
# 
# python voc_det_exemplar_train.py $CFGFILE $TMPFOLDER
python voc_det_exemplar_mktest.py $CFGFILE $TMPFOLDER
NO_BATCHES=`cat $TMPFOLDER/testinfo.txt | sed -n '1p'`
NO_CLASSES=`cat $TMPFOLDER/testinfo.txt | sed -n '2p'`
CLASSES=(`cat $TMPFOLDER/testinfo.txt | tail -n 1`)
TESTTHREADS=4
DETECTTHREADS=8
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: ${CLASSES[@]}"

# all classes, and how many processes there should be in each iteration
# NO_FULL_ITS=`echo $(($NO_CLASSES/$TESTTHREADS))`
# if [ $NO_FULL_ITS -gt 0 ]; then
#     FULL_ITS=`seq 1 $NO_FULL_ITS`
#     IT_SIZES=`for it in $FULL_ITS; do echo $TESTTHREADS; done; echo $(($NO_CLASSES%$TESTTHREADS))`
# else
#     IT_SIZES=$NO_CLASSES
# fi
# 
# for B in `seq 1 $NO_BATCHES`; do
#     echo "Running batch $B"
#     START_CLS=0
#     for SZ in $IT_SIZES; do
#         echo "Running iteration of $SZ processes"
#         STOP_CLS=$(($START_CLS+$SZ-1))
#         for P in `seq $START_CLS $STOP_CLS`; do
#             CLS=${CLASSES[$P]}
#             echo "Running test on class no $P ($CLS)"
#             python voc_det_exemplar_test.py $CFGFILE $TMPFOLDER $B $CLS&
#         done
#         wait
#         START_CLS=$(($START_CLS+$SZ))
#     done
# done

# perform detection (clustering on all images) per image
for B in `seq 1 $NO_BATCHES`; do
    IMIDS=(`cat $TMPFOLDER/batch_$B.pkl.txt`)
    NO_IMIDS=${#IMIDS[@]}
    NO_FULL_ITS=`echo $(($NO_IMIDS/$DETECTTHREADS))`
    if [ $NO_FULL_ITS -gt 0 ]; then
        FULL_ITS=`seq 1 $NO_FULL_ITS`
        IT_SIZES=`for it in $FULL_ITS; do echo $DETECTTHREADS; done; echo $(($NO_IMIDS%$DETECTTHREADS))`
    else
        IT_SIZES=$NO_IMIDS
    fi
    for CLS in ${CLASSES[@]}; do
        echo "Running detection on class $CLS"
        START_IMID=0
        for SZ in $IT_SIZES; do
            echo "Running batch $B on detection. $SZ images simultaneously"
            STOP_IMID=$(($START_IMID+$SZ-1))
            for I in `seq $START_IMID $STOP_IMID`; do
                IMID=${IMIDS[$I]}
                echo "Running detection on image no $I ($IMID)"
                python voc_det_exemplar_detect.py $CFGFILE $TMPFOLDER $B $CLS $IMID&
            done
            wait
            START_IMID=$(($START_IMID+$SZ))
        done
    done
done
    