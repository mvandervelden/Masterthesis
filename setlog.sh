#!/bin/bash
# $1 = destination logfile
# $2 = write mode (w or a)
# add .cfg and replace slashes to be escaped ('s/ \/ / \\ \/ /')
NLOG=$(echo $1 | sed 's/\//\\\//g')
NLOGCFG=$1.cfg
less blank.log.cfg | sed "s/'blank.log','a'/'$NLOG','$2'/" > $NLOGCFG
