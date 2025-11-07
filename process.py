import os
import argparse
from tqdm import tqdm
import pandas as pd
from utils import metadata
from utils import dataloader
from utils.tagging_validation import tagging_validate,DEFAULT_MACRO_CAT
from models import PANNS_Model,inference
import json
def main():
    parser = argparse.ArgumentParser(description='Script to process sound files recorded by Audiomoth ')
    parser.add_argument('--data_path', default='example/audio/0002/', type=str, help='Path to wav files')
    parser.add_argument('--save_path', default='example/metadata/', type=str, help='Path to save meta data')
    parser.add_argument('--name', default='', type=str, help='name of measurement')
    parser.add_argument('--audio_format', default='wav', type=str, help='wav or flac')
    parser.add_argument('--l', default=10, type=int, help='Window length in seconds for audio tagging / must be more than 5 seconds')
    parser.add_argument('--model_type', default='MobileNetV2', type=str, help='Type of the model (e.g., ResNet22, MobileNetV2)')
    parser.add_argument('--save_audio_flac', default=1, type=int, help='Saving audio in flac format (needed to run visualization tool)')
    parser.add_argument('--multiprocessing', default=1, type=int, help='Number of processes to use for data loading')
    parser.add_argument('--batch_size', default=12, type=int, help='Size of the batch for data loading')
    parser.add_argument('--macro-json', default=None, type=str, help='Optional path to JSON file defining macro categories')
    args = parser.parse_args()

    AUDIO_FORMAT = args.audio_format
    LEN_AUDIO = args.l

    if LEN_AUDIO < 5:
        raise ValueError('With tagging, length_audio_segment must be more than 5')

    csvfile = os.path.join(args.save_path, f'indices_{args.name}.csv')
    audio_savepath = os.path.join(args.save_path, f'audio_{args.name}')
    if not os.path.exists(audio_savepath):
        os.makedirs(audio_savepath)

    # get json file 
    if args.macro_json is not None:
        with open(args.macro_json, 'r') as jf:
            macro_cat = json.load(jf)
        print(f'Loaded macro categories from {args.macro_json}')
    else:
        print('Using default macro categories')
        macro_cat = DEFAULT_MACRO_CAT

    # get meta data file
    df_files = metadata.metadata_generator(args.data_path, AUDIO_FORMAT)
    if len(df_files) == 0:
        raise('No audio file found')

    # get data loader
    dl = dataloader.get_dataloader_site(df_files, savepath = audio_savepath, len_audio_s  = LEN_AUDIO, save_audio=args.save_audio_flac, batch_size = args.batch_size, num_workers=args.multiprocessing)
    df_site = {'datetime':[], 'name':[],'flacfile':[], 'start':[]}
    df_site['clipwise_output'] =  []
    df_site['embedding'] = []  
    df_site['sorted_indexes'] = []
    df_site['dB'] = []

    ## Initialize audio tagging model 
    model = PANNS_Model.from_pretrained(f"nicofarr/panns_{args.model_type}")
    model.eval()

    for batch_idx, (inputs, info) in enumerate(tqdm(dl)):

        clipwise_output, labels, sorted_indexes, embedding = inference(model, inputs, usecuda=False)

        for idx, date_ in enumerate(info['date']):
            df_site['datetime'].append(str(date_)) 
            df_site['name'].append(str(info['name'][idx]))
            df_site['flacfile'].append(str(date_)+'.flac')
            df_site['start'].append(float(info['start'][idx]))

            df_site['clipwise_output'].append(clipwise_output[idx])
            df_site['sorted_indexes'].append(sorted_indexes[idx])
            df_site['embedding'].append(embedding[idx])

            for key in info['ecoac'].keys():
                df_site[key].append(float(info['ecoac'][key].numpy()[idx])) 

    Df_tagging = tagging_validate(df_site,macro_cat=macro_cat)


    ## Dataframe with only ecoacoustic indices and important metadata 

    Df_eco = pd.DataFrame()
    Df_eco['name'] = df_site['name']
    Df_eco['start'] = df_site['start']
    Df_eco['datetime'] = df_site['datetime']
    for key in info['ecoac'].keys():
        Df_eco[key] = df_site[key]

    ## Fusing with the dataframe containing only the ecoacoustic indices 
    Df_final = pd.merge(Df_tagging,Df_eco,on=['name','start','datetime'])

    Df_final.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)
    print(f'Saved indices to {csvfile}')

if __name__ == "__main__":
    main()