if [ ! $# -eq 4 ]
then 
    echo "Usage: ./sweep_param.sh [function_num] [function_param_start] [function_param_end] [file]"
    exit 1
fi 

function_num=$1 
function_param_start=$2
function_param_end=$3
output_file=$4

echo "" > $4 
for param_num in $(seq $function_param_start $function_param_end);
do
    ./run $function_num $param_num >> $4;
done