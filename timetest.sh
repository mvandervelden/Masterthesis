#!/bin/bash

/usr/bin/time -f "Time: %e\nCPU: %P\nmemory_max: %M\nmemory_av: %K\n" -o timing.res ./tst.sh