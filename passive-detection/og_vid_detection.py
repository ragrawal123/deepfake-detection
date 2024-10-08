import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.model_zoo import load_url
import matplotlib as plt
from scipy.special import expit

import sys
sys.path.append('..')
from eval_utils.blazeface import FaceExtractor, BlazeFace, VideoReader
from eval_utils.architectures import fornet, weights
from eval_utils.isplutils import utils
import time

def main():
    '''
    Adapted from https://github.com/polimi-ispl/icpr2020dfdc/
    Set up model and face detector
    '''
    net_model = 'EfficientNetAutoAttB4ST'
    database = 'FFPP'
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

    face_policy = 'scale'
    face_size = 224
    frames_per_video = 128

    model_url = weights.weight_url['{:s}_{:s}'.format(net_model, database)]
    eval_model = getattr(fornet, net_model)().eval().to(device)
    eval_model.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    transf = utils.get_transformer(face_policy, face_size, eval_model.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("./eval_utils/blazeface/blazeface.pth")
    facedet.load_anchors("./eval_utils/blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video) 
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

    #Opening datasaving directory and root original video data
    ogdir = './data/ogdir/'
    if not os.path.exists(ogdir):
        os.makedirs(ogdir)
    rootdir = '/media/raunak/E380-1E91/Deepfake/End_To_End/og_may24/'
    checked_dict = dict()
    
    #Evaluating data
    i = 0
    for root, dirs, files in os.walk(rootdir):
        if len(dirs) == 0:
            participant = os.path.basename(os.path.dirname(root))
            
            #Check if data for participant already written
            #If so, load into a dictionary
            if os.path.exists(f"{ogdir}{participant}.txt"):
                with open(f"{ogdir}{participant}.txt") as file:
                    for line in file:
                        line.strip('\n')
                        id, data = line.split(':', 1)
                        checked_dict[id] = data
                    file.close()

            participant_file = open(f"{ogdir}{participant}.txt", 'a')
            
            #Walk through directory and evaluate .mp4 files
            for file in os.listdir(root):
                if file.endswith('.mp4') and file.startswith("p"):
                     #Check if mp4 file was already evaluated
                    path = f"{root}/{file}".split('/', 6)[6]
                    if path in checked_dict.keys():
                        print(f'{i}: {path}: Already In File')
                        i += 1
                        continue

                    #Get evaluation score
                    start_time = time.time()
                    score = eval(path=f"{root}/{file}", 
                                 model=eval_model, 
                                 device=device, 
                                 face_extractor=face_extractor,
                                 transf=transf)
                    end_time = time.time()

                    t_time = '{:.2f}'.format(end_time - start_time)
                    
                    #Write to datasaving file
                    participant_file.write(f"{path}:{score}:{t_time}\n")
                    print(f'{i}: Scored: {path}')
                    i += 1
            participant_file.close()
    
    #Remove miscallaneous data
    os.remove(f"{ogdir}/iphone.txt")


def eval(path, face_extractor, model, device, transf):
    '''
    Taken from:

    N. Bonettini, E. D. Cannas, S. Mandelli, L. Bondi, P. Bestagini and S. Tubaro, 
    "Video Face Manipulation Detection Through Ensemble of CNNs," 
    2020 25th International Conference on Pattern Recognition (ICPR)

    Github: https://github.com/polimi-ispl/icpr2020dfdc/
    '''
    vid_face = face_extractor.process_video(path)
    try:
        frame_face = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_face if len(frame['faces']) ] )
    except:
        print(f"Error with: {path}")

    with torch.no_grad():
        frame_face = model(frame_face.to(device)).cpu().numpy().flatten()
    
    return '{:.4f}'.format(expit(frame_face.mean()))



if __name__=='__main__':
    main()
