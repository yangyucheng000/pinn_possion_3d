cd ..

config_dir="src/config"
ckpt_dir="checkpoints_done"

python eval.py $config_dir/dataset_tetrahedron.yml $config_dir/model_3d.yml $ckpt_dir/possion_3d_tetrahedron.ckpt
python eval.py $config_dir/dataset_cylinder.yml $config_dir/model_3d.yml $ckpt_dir/possion_3d_cylinder.ckpt
python eval.py $config_dir/dataset_cone.yml $config_dir/model_3d.yml $ckpt_dir/possion_3d_cone.ckpt
