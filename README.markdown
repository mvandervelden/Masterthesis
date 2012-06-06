# Master Thesis Github #


Tests are run using GSE job manager:

`qsub -v SCRIPT=foo bar.job`

The code expects a python package called **nbnn** (Found [here](https://github.com/cvanweelden/nbnn) )

## File description ##

**detection\_cheap.py**
Usage: `$ python detection_cheap.py some/config/file.cfg`

Main file for a cheap kind of detection on the VOC11 data set.

**make5cfg\_iterations.sh** and **make10cfg\_iterations.sh**
Usage: `$ ./makeNcfg_iterations.sh some/config/pattern`

Shell script that searches for a file called `pattern_1.cfg` in path `some/config`
and duplicates this file N times. During duplication, `_1` patterns within the file
are replaced with `_i`. Useful in case multiple iterations with the same settings
need to be run.

**pascal.py**
Dataset classes for the Pascal Detection (future: Classification) tasks, that
extend nbnn's VOC Dataset class. Also contains a VOCResultsHandler class that
interprets and saves VOC detection results

**README.markdown**
This file

**run\_1.job**, **run\_5\_serial.job**, **run\_10\_serial.job**
Job scripts to runthe program using variable config files

**voc\_cheapdetection.cfg**
Example Config file for the detection task

**voctest\_tesf.txt** and **voctest\_trsf.txt**
Example dataset files for VOC detection