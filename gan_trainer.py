import torch
import torch.multiprocessing as mp
import numpy as np
import random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import scale
from eeggan.helpers.trainer import Trainer
from eeggan.helpers import system_inputs
from eeggan.nn_architecture.models import TtsDiscriminator, TtsGenerator, TtsGeneratorFiltered
from eeggan.helpers.dataloader import Dataloader

from nn_architecture.models import AutoencoderDiscriminator

empiricalEEG = np.genfromtxt('data/gansEEGTrainingData.csv', delimiter=',', skip_header=1)


argv = []
if isinstance(argv,dict):
        args = []
        for arg in argv.keys():
            if argv[arg] == True: #If it's a boolean with True
                args.append(str(arg)) #Only include key if it is boolean and true
            elif argv[arg] == False: #If it's a boolean with False
                pass #We do not include the argument if it is turned false
            else: #If it's not a boolean
                args.append(str(arg) + "=" + str(argv[arg])) #Include the key and the value
        argv = args
default_args = system_inputs.parse_arguments(argv, file='Train_Gan.py')
opt = {
        'n_epochs': default_args['n_epochs'],
        'sequence_length': default_args['sequence_length'],
        'seq_len_generated': default_args['seq_len_generated'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['conditions']),
        'patch_size': default_args['patch_size'],
        'kw_timestep': default_args['kw_timestep_dataset'],
        'conditions': default_args['conditions'],
        'lambda_gp': 10,
        'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
        'latent_dim': 16,           # Dimension of the latent space
        'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_lstm': 2,                # number of lstm layers for lstm GAN
        'world_size': 0,            # number of processes for distributed training
    }
diff_data = False               # Differentiate data
std_data = False                # Standardize data
norm_data = True
dataloader = Dataloader(default_args['path_dataset'],
                            kw_timestep=default_args['kw_timestep_dataset'],
                            col_label=default_args['conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data)
dataset = dataloader.get_data(sequence_length=default_args['sequence_length'],
                                  windows_slices=default_args['windows_slices'], stride=5,
                                  pre_pad=default_args['sequence_length']-default_args['seq_len_generated'])

opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
opt['n_samples'] = dataset.shape[0]

    # keep randomly 30% of the data
    # dataset = dataset[np.random.randint(0, dataset.shape[0], int(dataset.shape[0]*0.3))]
padding = 0
if opt['sequence_length'] % opt['patch_size'] != 0:
    while opt['sequence_length'] % default_args['patch_size'] != 0:
        padding += 1
    opt['sequence_length'] += padding
dataset = torch.cat((dataset, torch.zeros(dataset.shape[0], padding)), dim=1)

if opt['seq_len_generated'] == -1:
    opt['seq_len_generated'] = opt['sequence_length']

discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size'], in_channels=1+opt['n_conditions'])  # TODO: Channel recovery: set in_channels to (number of channels)*2 in dataset
generator = TtsGenerator(seq_length=opt['seq_len_generated'],
                                 latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt['seq_len_generated'],
                                 patch_size=opt['patch_size'],
                                 channels=1)  # TODO: Channel recovery: set channels to number of channels in dataset
model = Trainer(generator, discriminator, opt)
model.load_checkpoint("/Users/tail/EEG-GAN/trained_models/gan_5ep_20230904_174007.pt")
##print(model.discriminator)


#Set seed for a bit of reproducibility

#This function averages trial-level empirical data for each participant and condition
def averageEEG(EEG):
    participants = np.unique(EEG[:,0])
    averagedEEG = []
    for participant in participants:
        for condition in range(2):
            averagedEEG.append(np.mean(EEG[(EEG[:,0]==participant)&(EEG[:,1]==condition),:], axis=0))
    return np.array(averagedEEG)

#Load test data to predict (data that neither the GAN nor the classifier will ever see in training)
EEGDataTest = np.genfromtxt('data/gansEEGValidationData.csv', delimiter=',', skip_header=1)
EEGDataTest = averageEEG(EEGDataTest)[:,1:]

#Extract test outcome and predictor data
y_test = EEGDataTest[:,0]
x_test = EEGDataTest[:,2:]
x_test = scale(x_test,axis = 1)

#Create participant by condition averages
Emp_train = averageEEG(empiricalEEG)[:,1:]

#Extract the outcomes
Emp_Y_train = Emp_train[:,0]

#Scale the predictors
Emp_X_train = scale(Emp_train[:,2:], axis=1)

#Shuffle the order of samples
trainShuffle = rnd.sample(range(len(Emp_X_train)),len(Emp_X_train))
Emp_Y_train = Emp_Y_train[trainShuffle]
Emp_X_train = Emp_X_train[trainShuffle,:]

#Setup tracking variable
predictionScores_SVM = []
ourmodel = model.discriminator

optimizer = optim.Adam(ourmodel.parameters(), lr=0.01, weight_decay=0.05)
criterion = nn.MSELoss()  # Change based on your task
num_epochs = 50

Emp_X_Train_tensor = torch.tensor(Emp_X_train, dtype=torch.float32)
EMP_Y_train_tensor = torch.tensor(Emp_Y_train, dtype=torch.float32)
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
from torch.utils.data import DataLoader

# Create dataset
dataset = CustomDataset(Emp_X_Train_tensor, EMP_Y_train_tensor)

# Create data loader
batch_size = 32  # adjust based on your needs
shuffle = True
dataloader = Dataloader(default_args['path_dataset'],
                            kw_timestep=default_args['kw_timestep_dataset'],
                            col_label=default_args['conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data)
dataset = dataloader.get_data()
dataset = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    print(epoch)
    # for each batch of data in your dataset:
    for batch in dataset:  # Replace 'dataloader' with your data loader

        print(batch)
        data_labels = batch[:, 0].unsqueeze(1)
        batch_data = batch[:, 1:]
        real_data = model.generator.autoencoder.encode(batch_data).reshape(-1, 1, model.discriminator.output_dim*model.discriminator.output_dim_2) if isinstance(model.discriminator, AutoencoderDiscriminator) and not model.discriminator.encode else batch_data
        real_data = torch.cat((real_data, data_labels.repeat(1, 1)), dim=-1)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = ourmodel(real_data)
        
        # Compute Loss
        loss = criterion(outputs, data_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()