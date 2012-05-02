echo $1
for it in `seq 2 5`; do
	s='s/_1/_'$it'/'
	o=$1'_'$it'.cfg'
	echo $o
	cat $1_1.cfg | sed $s > $o
	cat $o | grep $it
done
