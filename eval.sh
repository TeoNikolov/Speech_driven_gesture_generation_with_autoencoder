#!/usr/bin/env bash

# You call use it by itself if the model is already trained
# Several aspects needs to be customized at config.txt

# Read the parameters for the scripts
source config.txt

model=datav1_m2_250ep

prefix=tst_2022_v1_00

# Create a folder to store produced gesture sequences
mkdir -p baseline_gestures

# Make predictions for all the test sequences
# (replace 1094 by 1093 for the dev sequences)
for seq in `seq 0 1 9`;
        do
        echo
                echo 'Predicting sequence' $seq
                # Step0: Encode audios
                cd data_processing/
                which python
                python encode_audio.py --input_audio=$audio_dir/${prefix}${seq}.wav --output_file=$audio_dir/tst_2022_v1_${seq}.npy
                cd ..
                # Step1: Predict representation
                echo 'Predicting representations'
                python predict.py $model.hdf5 $audio_dir/tst_2022_v1_${seq}.npy enc_${dim}_prediction${seq}.npy
                # Step2: Decode representation into motion
                echo 'Decoding representations'
                python motion_repr_learning/ae/decode.py --data_dir $data_dir -input_file enc_${dim}_prediction${seq}.npy -output_file baseline_gestures/gesture${seq}_array.npy --layer1_width=128 --batch_size=8
                echo 'Making BVH files'
                cd data_processing
                python features2bvh.py -feat ../baseline_gestures/gesture${seq}_array.npy -bvh ../baseline_gestures/${prefix}${seq}.bvh
                cd ../
        done
# Remove motion npy predictions
rm baseline_gestures/*.npy
# Remove encoded prediction
rm enc_*.npy

