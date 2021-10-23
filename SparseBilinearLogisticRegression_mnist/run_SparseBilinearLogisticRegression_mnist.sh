epochs=300
batch_size=100

r=5
alpha=0.0005
lam=0.001
beta1=0.9
function_name=SparseBilinearLogisticRegression_Mnist.py
 
echo "nworkers = 20, different beta"
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9 --r=$r --lam=$lam
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.8 --r=$r --lam=$lam
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.5 --r=$r --lam=$lam
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.2 --r=$r --lam=$lam
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.0 --r=$r --lam=$lam

echo "sync with different worker"

mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 3 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
#mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam

echo "async with different worker"
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 3 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --beta1=$beta1 --r=$r --lam=$lam

python plot_SparseBilinearLogisticRegression_Mnist.py

