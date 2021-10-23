batch_size=100
epochs=200
alpha=0.005

function_name=train_allcnn_cifar10.py
echo "use GPU to select beta"
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.9 --use_cuda=True
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.8 --use_cuda=True
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.5 --use_cuda=True
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.2 --use_cuda=True
mpirun -np 2 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --beta1=0.0 --use_cuda=True

echo " "
echo " only use CPU, compare computing time"
echo "sync"
mpirun -np 6  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.9
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.9
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=sync --opt_name=inertial --param_type=const --beta1=0.9
echo "async"
mpirun -np 6  python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=0.9
mpirun -np 11 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=0.9
mpirun -np 21 python $function_name --alpha=$alpha --train_batch_size=$batch_size --epochs=$epochs --communication=async --opt_name=inertial --param_type=const --beta1=0.9

python plot_allcnn_cifar10.py
