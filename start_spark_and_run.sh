
#!/usr/bin/env bash

# Script to start a spark cluster and run a script on klone
source $SPARK_CONF_DIR/spark-env.sh
echo "#!/usr/bin/bash" > job_script.sh
echo "source ~/.bashrc" >> job_script.sh
echo "export PYSPARK_PYTHON=python3" >> job.script.sh
echo "export JAVA_HOME=/gscratch/comdata/local/open-jdk" >> job.script.sh
echo "export SPARK_CONF_DIR=/gscratch/comdata/local/spark_config" >> job.script.sh
echo "echo \$(hostname)" >> job_script.sh
echo "source $SPARK_CONF_DIR/spark-env.sh" >> job.script.sh
echo "start_spark_cluster.sh" >> job_script.sh
echo "spark-submit --verbose --master spark://\$(hostname):$SPARK_MASTER_PORT $2 ${@:3}" >> job_script.sh
echo "stop-all.sh" >> job_script.sh

chmod +x job_script.sh

let "cpus = $1 * 40" 
salloc -p compute-bigmem -A comdata --nodes=$1 --time=48:00:00 -c 40 --mem=362G --exclusive srun -n1 job_script.sh

