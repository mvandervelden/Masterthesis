#!/bin/bash
IT=5

for i in `seq 1 $IT`; do
	python boiman.py graz01_person -d `eval date +%y%m%d_`$i > `eval date +%y%m%d_`$i.log &
done
wait
