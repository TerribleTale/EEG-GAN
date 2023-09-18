#Load EEG-GAN module
from eeggan import train_gan, visualize_gan, generate_samples, setup_tutorial

#Load other modules specific to this notebook
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import random as rnd
from scipy import signal
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import torch

#Create a print formatting class
class printFormat:
    bold = '\033[1m'
    italic = '\033[3m'
    end = '\033[0m'

#Setup 
#This function downloads tutorial-required files (e.g., datasets) from the GitHub. These files can also be found within the package itself, but Google Colab has difficulty accessing it.
#This function is only necessary when running the tutorial but it also creates three folders (data, trained_models, generated_samples) that are needed for the package, so still may be useful with your own data.
setup_tutorial()

#Load the data
empiricalHeaders = np.genfromtxt('data/gansEEGTrainingData.csv', delimiter=',', names=True).dtype.names
empiricalEEG = np.genfromtxt('data/gansEEGTrainingData.csv', delimiter=',', skip_header=1)

#Print the head of the data
print(printFormat.bold + 'Display Header and first few rows/columns of data\n \033[0m' + printFormat.end)
print(empiricalHeaders[:6])
print(empiricalEEG[0:3,:6])

#Print some information about the columns
print('\n------------------------------------------------------------------------------------------')
print(printFormat.bold + '\nNote the first three columns:' + printFormat.end +'\n    ParticipantID - Indicates different participants\n    Condition - Indicates the condition (WIN = 0, LOSE = 1) to be classified\n    Trial - Indicates the trial number for that participant and condition')
print('\nThe remaining columns are titled Time1 to Time100 - indicating 100 datapoints per sample.\nThe samples span from -200 to 1000ms around the onset of a feedback stimulus.\nThese are downsampled from the original data, which contained 600 datapoints per sample.')

#Print some meta-data
print('\n------------------------------------------------------------------------------------------')
print('\n' + printFormat.bold + 'Other characteristics of our data include:' + printFormat.end)
print('-We have ' + str(len(set(empiricalEEG[:,0]))) + ' participants in our training set')
print('-Participants have an average of ' + str(round(np.mean([np.max(empiricalEEG[empiricalEEG[:,0]==pID,2]) for pID in set(empiricalEEG[:,0])]))) + ' (SD: ' + str(round(np.std([np.max(empiricalEEG[empiricalEEG[:,0]==pID,2]) for pID in set(empiricalEEG[:,0])]))) + ')' + ' trials per outcome (win, lose)')

#Determine which rows are each condition
lossIndex = np.where(empiricalEEG[:,1]==1)
winIndex = np.where(empiricalEEG[:,1]==0)

#Grand average the waveforms for each condition
lossWaveform = np.mean(empiricalEEG[lossIndex,3:],axis=1)[0]
winWaveform = np.mean(empiricalEEG[winIndex,3:],axis=1)[0]

#Determine x axis of time
time = np.linspace(-200,1000,100)

#Setup figure
f, (ax1) = plt.subplots(1, 1, figsize=(6, 4))

#Plot each waveform
ax1.plot(time, lossWaveform, label = 'Loss')
ax1.plot(time, winWaveform, label = 'Win')

#Format plot
ax1.set_ylabel('Voltage (μμ\muV)')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Empirical', loc='left')
ax1.spines[['right', 'top']].set_visible(False)
ax1.legend(frameon=False)

#Train the GAN on CPUs

argv = dict(
    path_dataset="data/gansEEGTrainingData.csv",
    n_epochs=5
)

train_gan(argv)

#Train the GAN on GPUs
#Note, on Google Colab you can start a GPU runtime by going to Runtime > Change runtime type > Hardware accelerator > GPU
'''
argv = dict(
    ddp = True,
    path_dataset="data/gansEEGTrainingData.csv",
    n_epochs=5
)

train_gan(argv)
'''