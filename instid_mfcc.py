# slid_mfcc_cnn.py

# September 2022

# Description: 
# This code performs a slid experiments with MFCC/CNN.

from genericpath import exists
import os
import time
import wave
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import librosa

from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dnn_models import MLP,flip
from dnn_models import CNN
from data_io import ReadList,read_conf_inp_mfcc,str_to_bool

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_waveform(waveform, sr, title="Waveform"):
  waveform = waveform.numpy()

  time_axis = torch.arange(0, len(waveform)) / sr

  figure, axes = plt.subplots(1, 1)
  axes.plot(time_axis, waveform, linewidth=1)
  axes.grid(True)

  figure.suptitle(title)
  plt.show(block=False)

def plot_spectrogram(specgram, title=None, ylabel='freq'):
    fig, axs = plt.subplots(1,1)
    axs.set_title(title or 'Spectrogram')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(specgram), origin='lower', aspect='auto')
    fig.colorbar(im, ax=axs)
    plt.show(block=False)



def create_batches_rnd(batch_size,data_list,N_snt,wlen,fact_amp):
    
 # defined sig_batch with 20,7 that is the size used for mfcc spec
 sig_batch=np.zeros([batch_size,40,18])
 lab_batch=np.zeros(batch_size)
  
 snt_id_arr=np.random.randint(N_snt, size=batch_size)
 
 rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

 for i in range(batch_size):
     
  # select a random sentence from the list 
  track_file = data_list['full_path'].iloc[snt_id_arr[i]]

  waveform, sr = torchaudio.load(track_file)

  # accesing to a random chunk
  snt_len=len(waveform[0])
  snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
  snt_end=snt_beg+wlen

  # print('WARNING: stereo to mono: '+track_file)
  waveform = waveform[0]

  # numpy_waveform = waveform[snt_beg:snt_end].numpy() if hasattr(waveform[snt_beg:snt_end], 'numpy') else np.asarray(waveform[snt_beg:snt_end])
  # y = numpy_waveform * rand_amp_arr[i]
  y = waveform[snt_beg:snt_end]*rand_amp_arr[i]

  mfcc = MFCC(y)

  # plot_waveform(y, sr)
  # plot_spectrogram(mfcc)
  # mfcc = librosa.feature.mfcc(y = y, sr = sample_rate)
  
  sig_batch[i,:]=mfcc
  lab_batch[i]=data_list['instrument_id'].iloc[snt_id_arr[i]]
  
 inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
 lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
  
 return inp,lab  



# Reading cfg file
options=read_conf_inp_mfcc('cfg/InstID_mfcc.cfg')

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt_x=list(map(int, options.cnn_len_filt_x.split(',')))
cnn_len_filt_y=list(map(int, options.cnn_len_filt_y.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))

#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)
test_size = float(options.test_size)

#[data]
output_folder=options.output_folder
pt_file=options.pt_file

# build base 
df = pd.read_csv(options.csv_path, sep=',')

X = df.drop(columns=['instrument'])
y = df['instrument']

train, test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed, stratify=y)

# training list
snt_tr=len(train.index)

# test list
snt_te=len(test.index)

instrument_id = test['instrument_id']

print('Test labels confusion matrix (for name):')
print(confusion_matrix(y_test, y_test))

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128


# Feature extractor CNN
CNN_arch = {'input_dim': [40, 18],
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt_x': cnn_len_filt_x,
          'cnn_len_filt_y': cnn_len_filt_y,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cuda()

DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()


if pt_file!='none' and exists(pt_file):
   checkpoint_load = torch.load(pt_file)
   CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
   DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
   DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 

# MFCC = torchaudio.transforms.MFCC(
#   sample_rate=fs, 
#   n_mfcc=40)

MFCC = torchaudio.transforms.MFCC(
  sample_rate=fs, 
  n_mfcc=40, 
  melkwargs={
    "n_fft": 2048,
    "n_mels": 256,
    "hop_length": 512,
    "mel_scale": "htk"
  })

for epoch in range(N_epochs):
  
  test_flag=0
  CNN_net.train()
  DNN1_net.train()
  DNN2_net.train()
 
  loss_sum=0
  err_sum=0

  batchs_start_time = time.time()

  for i in range(N_batches):

    [inp,lab]=create_batches_rnd(batch_size,train,snt_tr,wlen,0.2)
    pout=DNN2_net(DNN1_net(CNN_net(inp)))
    
    pred=torch.max(pout,dim=1)[1]
    loss = cost(pout, lab.long())
    err = torch.mean((pred!=lab.long()).float())
    
    optimizer_CNN.zero_grad()
    optimizer_DNN1.zero_grad() 
    optimizer_DNN2.zero_grad() 
    
    loss.backward()
    optimizer_CNN.step()
    optimizer_DNN1.step()
    optimizer_DNN2.step()
    
    loss_sum=loss_sum+loss.detach()
    err_sum=err_sum+err.detach()
 

  loss_tot=loss_sum/N_batches
  err_tot=err_sum/N_batches

  print("--- %s minutes for all batch---" % ((time.time() - batchs_start_time) / 60))
   
# Full Validation  new  
  if epoch%N_eval_epoch==0:
      
   CNN_net.eval()
   DNN1_net.eval()
   DNN2_net.eval()
   test_flag=1 
   loss_sum=0
   err_sum=0
   err_sum_snt=0
   
   with torch.no_grad():  

    labs = []
    preds = []

    test_start_time = time.time()

    for i in range(snt_te):
       
     track_file = test['full_path'].iloc[i]
     signal, _ = torchaudio.load(track_file)

    #  removed tensor format to perform mfcc feature extraction 
    #  before initializing variable
    #  signal=torch.from_numpy(signal).float()
    
      # print('WARNING: stereo to mono: '+track_file)
     signal = signal[0]

     lab_batch=test['instrument_id'].iloc[i]
    
     # split signals into chunks
     beg_samp=0
     end_samp=wlen
     
     N_fr=int((signal.shape[0]-wlen)/(wshift))
     
     # defined sig_arr with 20,7 that is the size used for mfcc spec
     sig_arr = np.zeros([Batch_dev,40,18])
     lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
     pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
     count_fr=0
     count_fr_tot=0
     while end_samp<signal.shape[0]:
         y = signal[beg_samp:end_samp]
         mfcc = MFCC(y)
         sig_arr[count_fr,:]=mfcc
         beg_samp=beg_samp+wshift
         end_samp=beg_samp+wlen
         count_fr=count_fr+1
         count_fr_tot=count_fr_tot+1
         if count_fr==Batch_dev:
             inp=Variable(torch.from_numpy(sig_arr).float().cuda().contiguous())
             pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
             count_fr=0
             sig_arr=np.zeros([Batch_dev,40,18])
   
     if count_fr>0:
      inp=Variable(torch.from_numpy(sig_arr[0:count_fr]).float().cuda().contiguous())
      pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))

    
     pred=torch.max(pout,dim=1)[1]
     loss = cost(pout, lab.long())
     err = torch.mean((pred!=lab.long()).float())
    
     [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
     err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
    
     preds.append(best_class.item())
     labs.append(lab[0].item())
    
     loss_sum=loss_sum+loss.detach()
     err_sum=err_sum+err.detach()
    
    err_tot_dev_snt=err_sum_snt/snt_te
    loss_tot_dev=loss_sum/snt_te
    err_tot_dev=err_sum/snt_te

   print("--- %s minutes for test---" % ((time.time() - test_start_time) / 60))  
  
   print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))
  
   print('Matriz de confusão')
   print(confusion_matrix(labs, preds))
  
   with open(output_folder+"/res.res", "a") as res_file:
    res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))   

   checkpoint={'CNN_model_par': CNN_net.state_dict(),
               'DNN1_model_par': DNN1_net.state_dict(),
               'DNN2_model_par': DNN2_net.state_dict(),
               }
   torch.save(checkpoint,output_folder+'/model_raw.pkl')
  
  else:
   print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))