epochs=100
batch_size=40
beta1=0.9

m=50000
d=20000
batch_size=100
alpha=0.00005

function_name=PhaseRetrieval_random_data_loop_diff_x.py

#echo "nworkers = 6, select alpha"
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=1e-6 --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=1e-5 --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=1e-4 --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=1e-3 --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=1e-2 --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9


echo "nworkers = 6, select beta"
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.8
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.5
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.2
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.0

mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=alpha_reduce --period_type=epoch  --beta1=0.0
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch --beta1=0.9
#mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=step --beta1=0.9

echo " "
echo "nworkers = 1"
mpirun -np 2 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 2"
mpirun -np 3 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 5"
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 10"
mpirun -np 11 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 20"
mpirun -np 21 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1

echo "nworkers = 1"
mpirun -np 2 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 2"
mpirun -np 3 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 5"
mpirun -np 6 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 10"
mpirun -np 11 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
echo "nworkers = 20"
mpirun -np 21 python $function_name --m=$m --d=$d --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=reduce --period_type=epoch  --beta1=$beta1
