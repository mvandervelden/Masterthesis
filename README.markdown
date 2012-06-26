# Master Thesis Github #


Tests are run using GSE job manager:

on DAS4: `qsub -v CFGFILE=foo -v TMPFOLDER=bar testjob.job`
on own computer : `./testjob.sh $CFGFILE $TMPFOLDER`

The code expects a python package called **nbnn** (Found [here](https://github.com/cvanweelden/nbnn) )

## File description ##


**voc\_cls.sh** and **voc\_cls.job**
Main scripts for running tests for classification on full images (VOC)

**voc\_cls.py** and **voc\_cls\_test.py**
Main files that respectively train and test on full images (VOC)

**voc\_cls\_bbox.sh** and **voc\_cls\_bbox.job**
Main scripts for running tests for classification on bboxes of images (VOC)

**voc\_cls\_bbox.py** and **voc\_cls\_bbox\_test.py**
Main files that respectively train and test on bboxes of images (VOC)

**utils.py**
Helper functions

**procedures.py**
Generic main routines for training (and testing)

**setlog.sh**
shell script to make a temporary log config file, used by `init\_log()` in `utils.py`

**osx\_cls.cfg**, **osx\_cls\_bbox.cfg**, **das\_cls.cfg** and **das\_cls\_bbox.cfg**
config files for tests on laptop (osx) and server (das)

**parse\_log.py**

usage: `$ python parse_log.py logfile.log`

Gives the lines in the log where most memory is used

**cal\_perf.py**

usage: `$ python cal_perf.py confmat.txt`

Calculates the MRR and plots the confmat of the results of a caltech101 test

**blank.log.cfg**
Generic log config file that is used (copied) by `setlog.cfg`

**README.markdown**
This file


-----


*Deprecated: need to write new ones*

**detection\_cheap.py**
Usage: `$ python detection_cheap.py some/config/file.cfg`

Main file for a cheap kind of detection on the VOC11 data set.

Shell script that searches for a file called `pattern_1.cfg` in path `some/config`
and duplicates this file N times. During duplication, `_1` patterns within the file
are replaced with `_i`. Useful in case multiple iterations with the same settings
need to be run.

**pascal.py**
Dataset classes for the Pascal Detection (future: Classification) tasks, that
extend nbnn's VOC Dataset class. Also contains a VOCResultsHandler class that
interprets and saves VOC detection results
