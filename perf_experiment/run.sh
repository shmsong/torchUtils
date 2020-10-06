
# tensor dimensions in the micro benchmark
sizes='1024 1024'

# number of iterations
N=5

# run twice with different order to isolate the effect
for i in $(seq $N)
do
nvprof python big_tensor.py  --executor profiling --sizes $sizes --big_tensor   2>&1  | egrep '(CudaCodeGen|iter)'
nvprof python big_tensor.py  --executor profiling --sizes $sizes                2>&1  | egrep '(CudaCodeGen|iter)'
done

echo "-----------------------------------------------------------------------------------------------------------"

for i in $(seq $N)
do
nvprof python big_tensor.py  --executor profiling --sizes $sizes              2>&1  | egrep '(CudaCodeGen|iter)'
nvprof python big_tensor.py  --executor profiling --sizes $sizes --big_tensor 2>&1  | egrep '(CudaCodeGen|iter)'
done
