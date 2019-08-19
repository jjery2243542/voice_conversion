. vctk.config
python3 make_dataset_vctk.py $data_root_dir $h5py_path $train_proportion
python3 make_single_samples.py $h5py_path $index_path $n_samples $seg_len $speaker_used_path
