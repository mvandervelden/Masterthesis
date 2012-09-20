#!/bin/bash


CFGFILE=$1
echo "Running training"
TMPFOLDER=`cat $CFGFILE | awk '$1 ~ /tmp_dir/ { print $3 }'`

# python train_detection.py $CFGFILE
# echo "Making batches"
# python make_detection_batches.py $CFGFILE
echo "Reading cfg $CFGFILE"
NNTHREADS=`cat $CFGFILE | awk '$1 ~ /nn_threads/ { print $3 }'`
DETECTTHREADS=`cat $CFGFILE | awk '$1 ~ /det_threads/ { print $3 }'`
NO_BATCHES=`cat $TMPFOLDER/testinfo.txt | sed -n '1p'`
NO_CLASSES=`cat $TMPFOLDER/testinfo.txt | sed -n '2p'`
CLASSES=(`cat $TMPFOLDER/testinfo.txt | tail -n 1`)

echo "TMPFOLDER: $TMPFOLDER"
echo "No of NN-threads: $NNTHREADS"
echo "No of DET-threads: $DETECTTHREADS"
echo "No of batches: $NO_BATCHES"
echo "No of classes: $NO_CLASSES"
echo "Classes: ${CLASSES[@]}"
echo ""
echo "Running NN for all batches and classes"

# NO_FULL_ITS=`echo $(($NO_CLASSES/$NNTHREADS))`
# if [ $NO_FULL_ITS -gt 0 ]; then
#     FULL_ITS=`seq 1 $NO_FULL_ITS`
#     IT_SIZES=`for it in $FULL_ITS; do echo $NNTHREADS; done; REM=$(($NO_CLASSES%$NNTHREADS)); if [ $REM != 0 ]; then echo $REM; fi`
# else
#     IT_SIZES=$NO_CLASSES
# fi
# echo "Iterations: $NO_FULL_ITS"
# echo "FULL_ITS: $FULL_ITS"
# echo "It_sizes: $IT_SIZES"
# for B in `seq 1 $NO_BATCHES`; do
#     echo "Running batch $B"
#     START_CLS=0
#     for SZ in $IT_SIZES; do
#         echo "Running iteration of $SZ processes"
#         STOP_CLS=$(($START_CLS+$SZ-1))
#         for P in `seq $START_CLS $STOP_CLS`; do
#             CLS=${CLASSES[$P]}
#             echo "Running NN on class no $P ($CLS)"
#             python get_detection_distances.py $CFGFILE $B $CLS&
#         done
#         wait
#         START_CLS=$(($START_CLS+$SZ))
#     done
# done

echo "Running detection"
# perform detection (clustering on all images) per image
for B in `seq 1 $NO_BATCHES`; do
    IMIDS=(`cat $TMPFOLDER/batches/$B.pkl.txt`)
    NO_IMIDS=${#IMIDS[@]}
    NO_FULL_ITS=`echo $(($NO_IMIDS/$DETECTTHREADS))`
    if [ $NO_FULL_ITS -gt 0 ]; then
        FULL_ITS=`seq 1 $NO_FULL_ITS`
        IT_SIZES=`for it in $FULL_ITS; do echo $DETECTTHREADS; done; REM=$(($NO_IMIDS%$DETECTTHREADS)); if [ $REM != 0 ]; then echo $REM; fi`
    else
        IT_SIZES=$NO_IMIDS
    fi
    echo "  Iterations: $NO_FULL_ITS"
    echo "  FULL_ITS: $FULL_ITS"
    echo "   It_sizes: $IT_SIZES"
    for CLS in ${CLASSES[@]}; do
        echo "Running detection on class $CLS"
        START_IMID=0
        for SZ in $IT_SIZES; do
            echo "Running batch $B, class $CLS on detection. $SZ images simultaneously"
            STOP_IMID=$(($START_IMID+$SZ-1))
            for I in `seq $START_IMID $STOP_IMID`; do
                IMID=${IMIDS[$I]}
                echo "Running detection on image no $I ($IMID)"
                python detection.py $CFGFILE $B $CLS $IMID&
            done
            wait
            START_IMID=$(($START_IMID+$SZ))
        done
    done
done
echo "FINISHED ALL"

