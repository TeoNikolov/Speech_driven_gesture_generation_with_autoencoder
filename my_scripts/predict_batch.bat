@echo off
SET MODEL_NAME=full_model
SET INPUT_DIR=C:\Users\tniko\Documents\Work\WARA_Summer_School_2022\src\Speech_driven_gesture_generation_with_autoencoder\dataset\dev\my_audio
SET OUTPUT_DIR=C:\Users\tniko\Documents\Work\WARA_Summer_School_2022\src\Speech_driven_gesture_generation_with_autoencoder\output\%MODEL_NAME%
SET LAYER_DIM=128
SET DIR=%cd%

mkdir %OUTPUT_DIR% 1>NUL 2>NUL
cd %INPUT_DIR%
@ECHO ---
@ECHO Processing audio files from "%INPUT_DIR%"
@ECHO All intermediate and final files will be saved to "%OUTPUT_DIR%"
@ECHO ---
for %%i in (*) do call :Predict %%~ni
goto :End

:Predict
@echo off
set SAMPLE_IN_WAV_NAME=%1.wav
set SAMPLE_IN_ENCODED_NAME=%1.npy
set SAMPLE_OUT_ENCODED_NAME=%1_out_encoded.npy
set SAMPLE_OUT_DECODED_NAME=%1_out_decoded.npy
set SAMPLE_OUT_BVH_NAME=%1_out_decoded.bvh
cd "%DIR%\..\"

@ECHO Processing "%SAMPLE_IN_WAV_NAME%"
@ECHO   Copying raw audio...
copy "%INPUT_DIR%\%SAMPLE_IN_WAV_NAME%" "%OUTPUT_DIR%\%SAMPLE_IN_WAV_NAME%" 1>NUL 2>NUL

:: Encode the raw WAV audio data into audio features
@ECHO   Encoding raw audio into audio features...
cd data_processing
python encode_audio.py --input_audio="%INPUT_DIR%\%SAMPLE_IN_WAV_NAME%" --output_file="%OUTPUT_DIR%\%SAMPLE_IN_ENCODED_NAME%"

:: Predict (encoded) gestures given the encoded audio as input
@ECHO   Predicting encoded gestures from encoded audio features...
cd ..
python predict.py %MODEL_NAME%.hdf5 "%OUTPUT_DIR%\%SAMPLE_IN_ENCODED_NAME%" "%OUTPUT_DIR%\%SAMPLE_OUT_ENCODED_NAME%" 1>NUL 2>NUL

:: Decode the predicted gestures
@ECHO   Decoding predicted gestures...
python motion_repr_learning\ae\decode.py -input_file "%OUTPUT_DIR%\%SAMPLE_OUT_ENCODED_NAME%" -output_file "%OUTPUT_DIR%\%SAMPLE_OUT_DECODED_NAME%" --layer1_width %LAYER_DIM% --batch_size=8 1>NUL 2>NUL

:: Convert from decoded gestures to BVH animation data
@ECHO   Converting decoded gestures to BVH animation data...
cd data_processing
python features2bvh.py -feat "%OUTPUT_DIR%\%SAMPLE_OUT_DECODED_NAME%" -bvh "%OUTPUT_DIR%\%SAMPLE_OUT_BVH_NAME%" 1>NUL 2>NUL

:End
cd "%DIR%"