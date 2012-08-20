#!/bin/bash
echo $1
cat $1 | awk -F "[^0-9]*" '{ if (min==""){min=max=$8}; if ($8>max){max=$8}; if ($8<min){min=$8}; total+=$8;count+=1} END {printf "Memory use: min: %8dkb max: %8dkb, av:%8dkb\n", min, max, total/count}'
cat $1 | awk -F "[^0-9]*" '{ if (min==""){min=max=$9}; if ($9>max){max=$9}; if ($9<min){min=$9}; total+=$9;count+=1} END {printf "Swap   use: min: %8dkb max: %8dkb, av:%8dkb\n", min, max, total/count}'