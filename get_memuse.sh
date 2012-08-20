
cat $1 | awk -F "[^0-9]*" '{ if (min==""){min=max=$8}; if ($8>max){max=$8}; if ($8<min){min=$8}; total+=$8;count+=1} END {print "Memory use: min: ", min, "max: ", max, "av: ", total/count}'
cat $1 | awk -F "[^0-9]*" '{ if (min==""){min=max=$9}; if ($9>max){max=$9}; if ($9<min){min=$9}; total+=$9;count+=1} END {print "Swap   use: min: ", min, "max: ", max, "av: ", total/count}'