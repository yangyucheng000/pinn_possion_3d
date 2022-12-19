cd ..

config_dir="src/config"
save_dir="checkpoints/"
n_epoch=600

python train.py $config_dir/dataset_tetrahedron.yml $config_dir/model_3d.yml --save_dir=$save_dir --n_epoch=$n_epoch
python train.py $config_dir/dataset_cylinder.yml $config_dir/model_3d.yml --save_dir=$save_dir --n_epoch=$n_epoch
python train.py $config_dir/dataset_cone.yml $config_dir/model_3d.yml --save_dir=$save_dir --n_epoch=$n_epoch

