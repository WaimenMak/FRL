#!/bin/bash
###Use queue (partition) q1
#SBATCH -p q1
### Use 1 nodes and 8 cores
#SBATCH -N 1 -n 10
#SBATCH --mail-user=21422885@life.hkbu.edu.hk
#SBATCH --mail-type=end
#SBATCH --time=20:00:00
# std oupt
#SBATCH -o eval.o

cd ${HOME}/FRL/FRL/fed_reg/
export conda_env=${HOME}/anaconda3/envs/frl
export PATH=${conda_env}/bin:${HOME}/anaconda3/condabin:${PATH}
export LD_LIBRARY_PATH=${conda_env}/lib:${LD_LIBRARY_PATH}

mean=0
sigma=0
time=3

for i in `seq $time`  #from 1
do
	python dist_reg_v3.py --trial=$i
	value=$(tail -n 1 eval.o)
	if [ $i -eq 1 ]
	then
		# value1=$i
		value1=$value
	elif [ $i -eq 2 ]
	then
		# value2=$i
		value2=$value
	elif [ $i -eq 3 ]
	then
		# value3=$i
		value3=$value
	fi	
	# value=$i
	mean=`expr $mean + $value`

done
mean=`expr $mean / $time`

echo "1:$value1, 2:$value2, 3:$value3"
echo "mean:$mean"

# sigma=0
# for i in `seq $time`
# do
# 	if [ $i -eq 1 ]
# 	then
# 		val=$value1
# 	elif [ $i -eq 2 ]
# 	then
# 		val=$value2
# 	elif [ $i -eq 3 ]
# 	then
# 		val=$value3
# 	fi	
# 	num1=`expr $val - $mean`
# 	num=`expr $num1 \* $num1`
# 	sigma=`expr $sigma + $num`
# done
# res=$(bc <<< "scale=2; sqrt($sigma)")

# echo "$mean+-$res

