#!/bin/bash
JOB_NUM=$1
echo "Showing cleaned log from $JOB_NUM"
E_LOG="detection_script.sh.e$JOB_NUM"
O_LOG="detection_script.sh.o$JOB_NUM"
echo "Errors:"
cat $E_LOG | grep -v 'import\|RuntimeWarning\|VLFeat'
read -p "Next up: Output (press key)"
cat $O_LOG | grep -v 'import\|RuntimeWarning\|VLFeat'