#!/usr/bin/env bash
# runs on worker node
# instance_name=spark-worker-$(hostname)
# echo $hostname
# instance_url="instance://$instance_name"
# singularity instance list
# singularity instance stop -F "$instance_name"
# singularity instance list
# sleep 5
# ls $HOME/.singularity/instances/sing/$(hostname)/nathante/$instance_name
# rm -r $HOME/.singularity/instances/sing/$(hostname)/nathante/$instance_name
# singularity instance start /gscratch/comdata/users/nathante/cdsc_base.sif $instance_name
source /gscratch/comdata/env/cdsc_klone_bashrc
source $SPARK_CONF_DIR/spark-env.sh
echo $(which python3)
echo $PYSPARK_PYTHON
echo "start-worker.sh spark://$1:$SPARK_MASTER_PORT"
start-worker.sh spark://$1:$SPARK_MASTER_PORT
