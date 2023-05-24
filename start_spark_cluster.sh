#!/usr/bin/env bash
nodes="$(scontrol show hostnames)"

export SPARK_MASTER_HOST=$(hostname)
echo $SPARK_MASTER_HOST
# singularity instance stop spark-boss
# rm -r $HOME/.singularity/instances/sing/$(hostname)/nathante/spark-boss
 
# for node in $nodes
# dol
#     echo $node
#     ssh $node "singularity instance stop --all -F"
# done

# singularity instance start /gscratch/comdata/users/nathante/cdsc_base.sif spark-boss
#apptainer exec /gscratch/comdata/users/nathante/containers/nathante.sif
start-master.sh 
for node in $nodes
do
    # if [ "$node" != "$SPARK_BOSS" ]
    # then
    echo $node
    ssh -t $node start_spark_worker.sh $SPARK_MASTER_HOST
   # fi				
done

