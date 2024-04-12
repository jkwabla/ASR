import librosa 
import os 
import json
import numpy as np
from pydub import AudioSegment
import IPython.display as ipd




# dataset_path = "dataset"
# json_path = "data.json"

SAMPLES_TO_CONSIDER =  22050 # Thats pretty much one second worth of sound based on librosa sample rates

def prepare_dataset(dataset_path,json_path, n_mfcc=13, hop_length=512 , n_fft=2048):
    
    #creating our data dictionary 

    data = {

        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files":[]
    }

    #loop through all the sub directories 

    for i, (dirpath, dir_names, filenames) in enumerate(os.walk(dataset_path)):
        
        # enusuring we not in at the root level
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1] #basically getting the Parent folder / sub folders with audio files
            data["mappings"].append(category)

            # print(f"Processing {category}")


            #loop through all filenames and pulls out MFCCs 

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)
                print(signal.shape,sample_rate)  

               
    #             # #ensure audio file is at least a second long

    #             if len(signal) > SAMPLES_TO_CONSIDER:

    #             #     #enforce 1 sec. long signal

    #                    signal = signal[:SAMPLES_TO_CONSIDER]

                #extract the MFCCs -  This gave me alot of wahala cos i didnt assisgn the positional arguement y=signal  mfcc will throw an erorr is thats not taken care of  

                MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    #           
    #     #store data 

                data["labels"].append(i-1) #the very first sub folder with audio which must be Zero in the array
                data["MFCCs"].append(MFCCs.T.tolist())
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))

    # #store in json file 

    try:
            with open(json_path, "w") as fp:
                json.dump(data, fp, indent=4)
            print("Data successfully written to", json_path)
    except Exception as e:
            print("Error occurred while writing to JSON file:", e)


if __name__ == "__main__":
    DATASET_PATH = "dataset"
    JSON_PATH = "data.json"
    prepare_dataset(DATASET_PATH, JSON_PATH)