
import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score
import preprocess_csv as pp
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import random_split
from sklearn.model_selection import KFold


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,subgraphs, adjacencies, molecular_sizes = inputs
        subgraphs=[t.cuda() for t in subgraphs]
        adjacencies=[t.cuda() for t in adjacencies]
      
        subgraphs = torch.cat(subgraphs)
        adjacencies = self.pad( adjacencies, 0)
        """GNN layer (update the subgraph vectors)."""
        subgraph_vectors = self.embed_fingerprint(subgraphs)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, subgraph_vectors, l)
            subgraph_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the subgraph vectors."""
        molecular_vectors = self.sum(subgraph_vectors, molecular_sizes)
        return Smiles,molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        a=data_batch[-1]
        a=[t.cuda() for t in a]
        correct_values = torch.cat(a)

        if train:
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            a=nn.L1Loss()
            loss = a(correct_values, predicted_values)
            return loss
        else:
            with torch.no_grad():
                Smiles,molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles,predicted_values, correct_values
    def forward_predict(self, data_batch):

            inputs = data_batch
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            
            return Smiles,predicted_values
            
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total
        
class Trainer_tf(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model
    def test_regressor(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values)
            Ys.append(predicted_values)
            
            SAE += sum(np.abs(predicted_values-correct_values))
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        MAEs = SAE / N  # mean absolute error.
        return MAEs,predictions
    def test_predict(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values) = self.model.forward_predict(
                                               data_batch)
            SMILES += ' '.join(Smiles) + ' '
            Ys.append(predicted_values)
        SMILES = SMILES.strip().split()
        Y = map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, Y)])
        return predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def split_dataset(dataset, ratio):
#   """Shuffle and split a dataset."""
    np.random.seed(1)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)
            
if __name__ == "__main__": 

    radius=1
    dim=48
    layer_hidden=10
    layer_output=10
    batch_train=32
    batch_test=8
    lr=2e-4
    lr_decay=0.99
    decay_interval=100
    iteration_tf=200
    N=5000
    path='./data/'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
   # print('The code uses a CPU!')
    
    dir_dataset = './data/'

    df = pd.read_csv(dir_dataset + "train_final.csv")
    df['RT'] = df['RT']*60
    labs = df['Lab'].unique()
    
  #  lab = labs[0]
    
    for lab in labs:
      print('labs ', lab)
      df_lab = df[df['Lab'] == lab]
  
      df_lab.to_csv(dir_dataset + "HILIC-train.csv", index=False)
  
      import datetime
      time1=str(datetime.datetime.now())[0:13]
      dataset= pp.transferlearning_dataset("HILIC-train.csv")
  
      print('Start training.')
  
      num_folds = 5
      kf = KFold(n_splits=num_folds, shuffle=True)
  
      for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        model= MolecularGraphNeuralNetwork( N, dim, layer_hidden, layer_output).to(device)
        file_model='./model/pre_GNN_model.h5'
        model.load_state_dict(torch.load(file_model,map_location=device ))
      
        trainer = Trainer_tf(model)
        tester = Tester(model)
        print(f"Fold: {fold + 1}")
        torch.cuda.empty_cache()
        # Create the training set based on the train indices
        dataset_train = [dataset[i] for i in train_index]
        # Create the validation set based on the validation indices
        dataset_val = [dataset[i] for i in val_index]
        
           
        start = timeit.default_timer()
        for epoch in tqdm(range(iteration_tf),ncols=50):
            epoch += 1
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            model.train()
            loss_train = trainer.train(dataset_train)
            MAE_tf_best=9999999
            model.eval()
            MAE_tf_train,predictions_train_tf = tester.test_regressor(dataset_train)
            MAE_tf_val = tester.test_regressor(dataset_val)[0]
            time = timeit.default_timer() - start
    
       #     results_tf = '\t'.join(map(str, [epoch, time, loss_train,MAE_tf_train,
       #                                   MAE_tf_val,]))
#            tester.save_MAEs(results_tf, file_MAEs)
    
            if MAE_tf_val <= MAE_tf_best:
                MAE_tf_best = MAE_tf_val
                MAE_tf_train_best = MAE_tf_train
                file_model = path+ 'all_models/' +lab + '/' + lab + '-Fold' + str(fold) +'_model'+'.h5'
                tester.save_model(model, file_model)
                #print(results_tf)
        print(f'time:{time:.2f},epoch:{epoch},MAE_tf_train:{MAE_tf_train_best},MAE_tf_val:{MAE_tf_best}')
    
              
    
    
    
  
  


















      
    
