#!/bin/bash
#$ -l h_rt=500:00:00
#$ -cwd


source ~/.bash_profile

START=$(date +%s)
# rm -rf /local/vdvelden
mkdir /local/vdvelden

# Default config file.
# To give a non-default file in qsub, use "qsub -v CFGFILE=foo.cfg detection_script.sh"
# To give a non-default file running the vanilla script, use  "./detection_script.sh foo.cfg"
DEFCFGFILE=localdet_VOC07_das.cfg

if [[ $CFGFILE && ${CFGFILE-x} ]]; then
    echo "CFGFILE defined: ${CFGFILE}"
elif [[ $1 && ${1-x} ]]; then
    echo "config file given in command line: $1"
    CFGFILE=$1
else
    echo "No CFGFILE defined, take: ${DEFCFGFILE}"
    CFGFILE=$DEFCFGFILE
fi


TMPFOLDER=`cat $CFGFILE | awk '$1 ~ /tmp_dir/ { print $3 }'`
RESFOLDER=`cat $CFGFILE | awk '$1 ~ /res_dir/ { print $3 }'`
TGZFILE=`echo $RESFOLDER | awk '{split($0, a, "/")} END{ print a[length(a)]}'`

echo "TMPFOLDER: $TMPFOLDER"
echo "RESFOLDER: $RESFOLDER"
echo "TGZFILE: $TGZFILE"

mkdir $RESFOLDER
cp $CFGFILE $RESFOLDER
CFGFILE=`echo "${RESFOLDER}/${CFGFILE}"`
echo "CFGFILE COPIED TO: $CFGFILE"

echo "Running local NBNN Detection"
python local_exemplar_nbnn.py $CFGFILE

echo "Tarballing results and tmpfiles"
# tar -czf ${TGZFILE}.tmp.tgz --exclude=*.dbin* --exclude=*.dtxt --exclude=*.data --exclude=*.index $TMPFOLDER
tar -czf ${TGZFILE}.res.tgz $RESFOLDER
tar -czf ${TGZFILE}.rankings.tgz ${RESFOLDER}/*/*.txt
scp ${TGZFILE}.*.tgz fs4.das4.science.uva.nl:/var/scratch/vdvelden/

# echo "Cleaning up tmp-dir"
# rm -rf /local/vdvelden

DURATION=$(echo "$(date +%s) - $START" | bc)
DUR_H=$(echo "$DURATION/3600" | bc)
DUR_M=$(echo "($DURATION-$DUR_H*3600)/60" | bc)
DUR_S=$(echo "($DURATION - $DUR_H*3600 - $DUR_M*60)" | bc)
echo "FINISHED ALL IN $DUR_H:$DUR_M:$DUR_S"