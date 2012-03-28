test=$1
id=$2
cfg=$3
# Set the number of iterations / nodes needed and get the train and testfiles
python boiman.py $test -d $id -c $cfg -p
# Get the number of chunks from the filename
chnks=$id/*chunks.pkl
tmp=${chnks#*/}
iterations=${tmp%.*}
# run the number of iterations
for it in `seq 1 $iterations`; do
	qrsh -l h_rt=10:00:00 "run_single_test.sh $test $id $cfg $it" &


rm *chunks.pkl
rm trainfiles.pkl