epochs=300
batch_size=40
alpha=0.001
beta1=0.9
 
function_name=train_lenet_mnist.py

echo "nworkers = 6, select beta"
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.9
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.8
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.5
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.2
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.0
mpirun -np 6 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch --beta1=0.9

echo "Compare the parallel computing time with constant alpha and beta"
echo "sync"
mpirun -np 2  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 3  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=$beta1
#mpirun -np 6  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=$beta1
echo "async"
mpirun -np 2  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 3  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 6  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=$beta1
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=$beta1

echo "Compare the parallel computing time with constant alpah and diminishing beta"
echo "sync"
mpirun -np 2 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 3 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
#mpirun -np 6 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 11 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 21 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
echo "async"
mpirun -np 2 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 3 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 6 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 11 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1
mpirun -np 21 python $function_name  --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs  --communication=async --opt_name=inertial --param_type=beta_reduce --period_type=epoch  --beta1=$beta1

python plot_lenet_mnist.py
