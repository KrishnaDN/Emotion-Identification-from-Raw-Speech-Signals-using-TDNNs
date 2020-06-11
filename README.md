# Emotion-Identification-from-raw-speech-signals-using-TDNNs

This repo contains the implementation of the paper Emotion Identification from raw speech signals using DNNs" 
By Mousmita Sarma, Pegah Ghahremani, Daniel Povey, Nagendra Kumar Goel,Kandarpa Kumar Sarma, Najim Dehak in Pytorch
The paper is published in Interspeech 2018
Paper: https://danielpovey.com/files/2018_interspeech_emotion_id.pdf

## Installation

I suggest you to install Anaconda3 in your system. First download Anancoda3 from https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
```bash
bash Anaconda2-2019.03-Linux-x86_64.sh
```
## Clone the repo
```bash
https://github.com/KrishnaDN/x-vector-pytorch.git
```
Once you install anaconda3 successfully, install required packges using requirements.txt
```bash
pip iinstall -r requirements.txt
```

## Data preperation
This steps creates manifest files for training and testing
```
python dataset.py --pickle_filepath  /media/newhd/IEMOCAP_dataset/data_collected_full.pickle
                 --dataset_root /media/newhd/IEMOCAP_dataset/raw_data --store_meta meta/
```
If you want to add your dataset, take a look at datasets.py code and modify the code accordingly


## Training
This steps starts training the model.
```
python training_Emo_TDNN_StatPool.py --training_filepath meta/training.txt --testing_filepath meta/testing.txt
                             --input_dim 1 --num_classes 4 --batch_size 64 --use_gpu True --num_epochs 100
                             
```
Note that this model is based on raw waveform TDNN.

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)
