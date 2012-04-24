for it in `seq 1 5`; do
	echo $it
	qsub -v SCRIPT=bcal$it.cfg boi_cal.job
done