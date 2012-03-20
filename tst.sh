#!/bin/bash
IT=5

for i in `seq 1 $IT`; do
	python boiman.py graz01_person -d `eval date +%y%m%d_`$i > `eval date +%y%m%d_`$i.log &
done
wait

# Current in use, by hand:
# /usr/bin/time -f "Time: %e\nCPU: %P\nmemory_max: %M\nmemory_av: %K\n" -o timing3.res python boiman.py graz01_bike -d `eval date +%y%m%d_`3

# Tryout timetst...
# /usr/bin/time -f "Time: %e\nCPU: %P\nmemory_max: %M\nmemory_av: %K\n" -o timing.res ./tst.sh

# Tryout dist_tst...
# IT=5
# 
# for i in `seq 1 $IT`; do
# 	# Claim a node TODO: find out how nesting quotes works..
# 	/usr/bin/time -f "Time: %e\nCPU: %P\nmemory_max: %M\nmemory_av: %K\n" -o timing$i.res qrsh -l h_rt=01:00:00 "cd code && boiman.py graz01_person -d `eval date +%y%m%d_`$i > `eval date +%y%m%d_`$i.log"
# done
