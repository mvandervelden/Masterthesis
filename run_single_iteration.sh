test=$1
id=$2
cfg=$3
# Set the number of iterations / nodes needed and get the train and testfiles
python boiman.py $test -d $id -c $cfg -p
# Get the number of chunks from the filename
cd $id
no_chunks=`python -c "import glob; print glob.glob('*chunks.pkl')[0][:1]"`
cd ..
# run the number of iterations
for it in `seq 1 $no_chunks`; do
	qrsh -l h_rt=10:00:00 "./run_single_test.sh $test $id $cfg $it" &
done
rm $id/*chunks.pkl
rm $id/trainfiles.pkl
