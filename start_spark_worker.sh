#!/usr/bin/env bash
# runs on worker node
source /gscratch/comdata/env/cdsc_klone_bashrc
source $SPARK_CONF_DIR/spark-env.sh
echo $(which python3)
echo $PYSPARK_PYTHON
echo "start-worker.sh spark://$1:$SPARK_MASTER_PORT"
start-worker.sh spark://$1:$SPARK_MASTER_PORT
