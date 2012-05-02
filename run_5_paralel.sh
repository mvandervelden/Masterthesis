for it in `seq 1 5`; do
	echo $it
	qsub -v SCRIPT=bcal$it.cfg run_1.job
done