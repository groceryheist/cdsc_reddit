#!/usr/bin/env bash
nodes="$(scontrol show hostnames)"

export SPARK_MASTER_HOST=$(hostname)
echo $SPARK_MASTER_HOST

start-master.sh 
for node in $nodes
do
    echo $node
    ssh -t $node start_spark_worker.sh $SPARK_MASTER_HOST
done

