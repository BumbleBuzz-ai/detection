dirlist=$1

for curdir in $(cat $dirlist)
do
uv run process.py --save_audio_flac 1 --data_path /media/nfarrugi/My\ Passport/CREA/$curdir --name $curdir --l 5 --save_path /media/nfarrugi/My\ Passport/CREA/output --model_type Cnn14 --audio_format wav
done