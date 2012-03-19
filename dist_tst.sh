#!/bin/bash

IT=5

for i in `seq 1 $IT`; do
	# Claim a node TODO: find out how nesting quotes works..
	/usr/bin/time -f "Time: %e\nCPU: %P\nmemory_max: %M\nmemory_av: %K\n" -o timing$i.res qrsh -l h_rt=01:00:00 "cd code && boiman.py graz01_person -d `eval date +%y%m%d_`$i > `eval date +%y%m%d_`$i.log"
done
