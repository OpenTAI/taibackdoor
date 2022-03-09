declare -a br_arr=(
  0.010
  0.025
  0.050
  0.075
  0.100
)

# declare -a type_arr=(
#   "sub_imagenet_isbba_rn18"
# )
#
# for br in "${br_arr[@]}"
#   do
#   for target in "${type_arr[@]}"
#     do
#       exp_path="experiments/backdoor/backdoor"_${br}
#       config_path="configs/backdoor/backdoor"_${br}
#       echo $exp_path $config_path $target
#       sbatch --partition deeplearn --qos gpgpudeeplearn    \
#              --mem=96G --gres=gpu:1 --cpus-per-task=4  \
#              --job-name $target --time=72:00:00        \
#              train.slurm $config_path $exp_path $target 0.001
#   done
# done


declare -a type_arr=(
  "cifar10_badnet_rn18"
  "cifar10_blend_rn18"
  "cifar10_cl_rn18"
  "cifar10_dfst_rn18"
  "cifar10_dynamic_rn18"
  "cifar10_sig_rn18"
  "cifar10_trojan_rn18"
)

for br in "${br_arr[@]}"
  do
  for target in "${type_arr[@]}"
    do
      exp_path="experiments/backdoor/backdoor"_${br}
      config_path="configs/backdoor/backdoor"_${br}
      echo $exp_path $config_path $target
      sbatch --partition gpgpu --qos gpgpumse    \
             --mem=16G --gres=gpu:1 --cpus-per-task=4  \
             --job-name $target --time=12:00:00        \
             train.slurm $config_path $exp_path $target 0.1
  done
done
