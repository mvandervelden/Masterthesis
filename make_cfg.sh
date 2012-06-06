ID=$2
cat blank.log.cfg | sed "s/blank/$ID/" > $ID.log.cfg
cat $1 | sed "s/blank/$ID/g" > $ID.cfg