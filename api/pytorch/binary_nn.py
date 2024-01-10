import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc 



class MyDataset():
    '''
        Dascription
    '''

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        





class BinaryNN(nn.Module):
    '''
        Class which offers methods to handle Neural Networks for Binary Classification tasks with at least 3 layers \
        (input, hidden and output) based on a dictionary of associations <hyperparameter_key,hyperparameter_value> \
        with the following keys: \n
        'input_units', 'hidden_units', 'learning_rate', 'momentum', 'input_activation', 'hidden_activation', 'output_activation', 'metrics'.
    '''


    def __init__(self, params: dict, monk_i: int, trial: int, n_hidden_layers: int):
        super(BinaryNN, self).__init__()

        # Sets the device used
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(device)

        # Passed values initializations
        self.params = params
        self.monk_i = monk_i
        self.trial = trial

        # Default's values initializations
        self.history: dict[list] = {'tr_accuracy':[], 'vl_accuracy':[], 'tr_loss':[], 'vl_loss':[], 'ts_accuracy':[]}
        self.mean_tr_accuracy    = 0
        self.mean_vl_accuracy    = 0
        self.ts_accuracy         = 0
        self.mean_tr_loss        = 0
        self.mean_vl_loss        = 0
        self.tr_batch_counter    = 0
        self.vl_batch_counter    = 0
        self.ts_batch_counter    = 0
        self.ts_loss             = 0
        self.f1_score            = 0
        self.f2_score            = 0
        self.recall_score        = 0
        self.precision_score     = 0
        self.tolerance           = 0.001
        self.patience            = 10
        self.y_predictions: torch.Tensor = None
        self.y_true:        torch.Tensor = None

        # Error case 
        if n_hidden_layers < 0: 
            raise ValueError 
        
        # Hidden activation function
        if self.params['hidden_activation'] == 'Tanh':
            activation = nn.ReLU()
        elif self.params['hidden_activation'] == 'ReLU':
            activation = nn.Tanh()
        else:
            raise ValueError
    
        # Input Layer
        '''self.layers.append(
            (
                nn.Linear(self.params['input_size'], self.params['hidden_size']),
                activation
            )
        )

        # Hidden Layers
        for _ in range(n_hidden_layers):
            self.layers.append(
                (
                    nn.Linear(self.params['hidden_size'], self.params['hidden_size']),
                    activation
                )
            )
        
        # Output Layer
        self.layers.append(
            (
                nn.Linear(self.params['hidden_size'], 1),
                nn.Sigmoid()
            )
        )'''
        self.layers = nn.Sequential(
            nn.Linear(self.params['input_size'], self.params['hidden_size']),
            activation,
            nn.Linear(self.params['hidden_size'], self.params['hidden_size']),
            activation,
            nn.Linear(self.params['hidden_size'], 1),
            nn.Sigmoid()
        )
        self.add_module('layers', self.layers)

        # Initialization of the Weights
        self.init_weights()

        # Loss Function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        if self.params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.params['learning_rate'],
                momentum=self.params['momentum'],
                weight_decay=self.params['weight_decay'],
                nesterov=True
            )
        elif self.params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.params['learning_rate'],
                #momentum=self.params['momentum'],
                weight_decay=self.params['weight_decay']
            )
        elif self.params['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.params['learning_rate'],
                momentum=self.params['momentum'],
                weight_decay=self.params['weight_decay']
            )
        else:
            raise ValueError

    
    def __str__(self) -> str:
        return \
            f" Monk:                     {self.monk_i}\n" + \
            f" Trial:                    {self.trial}\n" + \
            f" Hyperparameters:          {self.params}\n" + \
            f" Mean Training Loss:       {self.mean_tr_loss}\n" + \
            f" Mean Validation Loss:     {self.mean_vl_loss}\n" + \
            f" Test Loss:                {self.ts_loss}\n" + \
            f" Mean Training Accuracy:   {self.mean_tr_accuracy}\n" + \
            f" Mean Validation Accuracy: {self.mean_vl_accuracy}\n" + \
            f" Test Accuracy:            {self.ts_accuracy}\n" + \
            f" f1 score:                 {self.f1_score}\n" + \
            f" f2 score:                 {self.f2_score}\n" + \
            f" Prediction score:         {self.precision_score}\n" + \
            f" Recall score:             {self.recall_score}\n"
    

    def init_weights(self):
        '''
            Initializes the weights of the layers with default values.
        '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


    def print_plot(self):
        '''
            Prints the plot based on the history of the trained model.
        '''
        plt.figure()
        plt.plot(self.history['tr_loss'], label='Training Loss')
        plt.plot(self.history['vl_loss'], label='Validation Loss')
        plt.plot(self.history['tr_accuracy'], label='Training Accuracy')
        plt.plot(self.history['vl_accuracy'], label='Validation Accuracy')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.legend()
    

    def print_roc_curve(self, y_test):
        '''
            Prints the ROC curve graphic.
        '''
        # Calculate the ROC curve 
        fpr, tpr, thresholds = roc_curve(y_test, self.y_predictions) 
        
        # Calculate the Area Under the Curve (AUC) 
        roc_auc = auc(fpr, tpr) 
        
        # Plot the ROC curve 
        plt.figure(figsize=(8, 6)) 
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})') 
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        plt.title('Receiver Operating Characteristic (ROC) Curve') 
        plt.legend(loc='lower right')
        plt.show()


    def print_training_info(self):
        '''
            Prints the results of the Training Phase.
        '''
        print(
            f" Monk:                     {self.monk_i}\n" + \
            f" Trial:                    {self.trial}\n" + \
            f" Hyperparameters:          {self.params}\n" + \
            f" Mean Training Loss:       {self.mean_tr_loss}\n" + \
            f" Mean Validation Loss:     {self.mean_vl_loss}\n" + \
            f" Mean Training Accuracy:   {self.mean_tr_accuracy}\n" + \
            f" Mean Validation Accuracy: {self.mean_vl_accuracy}"
        )
    

    def print_confusion_matrix(self, y_test):
        '''
            Prints the confusion matrix based on the predictions made during the Testing Phase.
        '''

        # Error case
        if self.y_predictions == []:
            raise ValueError
        
        # Prints the Confusion Matrix as a DataFrame (alternative: tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_predictions).ravel())
        print(
            pd.DataFrame(
                data=confusion_matrix(y_true=y_test, y_pred=self.y_predictions),
                index=['Real_Class_0', 'Real_Class_1'],
                columns=['Predicted_Class_0', 'Predicted_Class_1']
            )
        )
    

    def forward(self, x):
        '''
            Execute the forwarding pass.
        '''
        return self.layers(x)

    
    def fit(self, x_train, y_train, x_val=None, y_val=None):
        '''
            Train the model based on the data passed as parameters and returns the history.\n
            - x_train: a NumPy array MxN dataset used for Training.\n
            - y_train: a NumPy Mx1 labels used for Training.\n
            - x_val: a NumPy array MxN dataset used for Validation.\n
            - y_val: a NumPy Mx1 labels used for Validation.
        '''

        train_dataset = MyDataset(
            torch.from_numpy(x_train).to(dtype=torch.float32),
            torch.from_numpy(y_train).to(dtype=torch.float32)
        )
        train_data = DataLoader(dataset=train_dataset, batch_size=int(self.params['batch_size']), shuffle=True)

        # Case of Retraining (Validation not necessary)
        if x_val is not None and y_val is not None:
            val_dataset = MyDataset(
                torch.from_numpy(x_val).to(dtype=torch.float32),
                torch.from_numpy(y_val).to(dtype=torch.float32)
            )
            val_data = DataLoader(dataset=val_dataset, batch_size=int(self.params['batch_size']), shuffle=True)
        else:
            val_dataset = None
            val_data = None
        
        # Counter for Early Stopping
        counter = 0

        # Save initial model weights
        initial_weights = copy.deepcopy(self.state_dict())

        # Epochs iteration
        for epoch in range(self.params['epochs']):

            # Batch iteration on TR set
            for batch_x, batch_y in train_data:
                
                # Resets gradients
                self.optimizer.zero_grad()

                # Forward pass
                tr_outputs: torch.Tensor = self(batch_x)

                # Cases of squeeze
                if tr_outputs.size() != torch.Size([1]):
                    # Case of squeeze don't needed (because we obtain a scalar)
                    if tr_outputs.size() == torch.Size([1,1]):
                        tr_outputs = tr_outputs[0]
                    # Case of squeeze don't needed (because we obtain a scalar)
                    elif tr_outputs.size() == torch.Size([1,1,1]):
                        tr_outputs = tr_outputs[0][0]
                    # Case of squeeze needed
                    else:
                        tr_outputs = tr_outputs.squeeze()

                # Compute Loss function
                loss = self.criterion(tr_outputs, batch_y)

                # Backward pass
                loss.backward()
                
                # Optimization
                self.optimizer.step()

                # Predictions
                batch_pred_y = torch.round(tr_outputs)
                correct_batch_pred_y = sum(
                    [1 for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
                )
                
                # Compute Accuracy and Loss
                tr_accuracy = float(correct_batch_pred_y / len(batch_pred_y))
                tr_loss     = loss.item()

                # Updates the history
                self.history['tr_accuracy'].append(tr_accuracy)
                self.history['tr_loss'].append(tr_loss)

                # Updates the mean of the Accuracy and the Loss on TR set
                self.mean_tr_accuracy = float((self.mean_tr_accuracy * self.tr_batch_counter + tr_accuracy) / (self.tr_batch_counter+1))
                self.mean_tr_loss     = float((self.mean_tr_loss * self.tr_batch_counter + tr_loss) / (self.tr_batch_counter+1))

                self.tr_batch_counter += 1
        
            # Case of Retraining (Validation not necessary)
            if val_data is None:
                return self.mean_tr_accuracy, self.mean_tr_loss, self.mean_vl_accuracy, self.mean_vl_loss
        
            # Evaluation on VL set
            with torch.no_grad():

                # Previous Mean Loss on VL set
                prev_mean_vl_loss = 0

                # Batch iteration on VL set
                for batch_x, batch_y in val_data:

                    # Forward pass
                    vl_outputs: torch.Tensor = self(batch_x)

                    # Cases of squeeze
                    if vl_outputs.size() != torch.Size([1]):
                        # Case of squeeze don't needed (because we obtain a scalar)
                        if vl_outputs.size() == torch.Size([1,1]):
                            vl_outputs = vl_outputs[0]
                        # Case of squeeze don't needed (because we obtain a scalar)
                        elif vl_outputs.size() == torch.Size([1,1,1]):
                            vl_outputs = vl_outputs[0][0]
                        # Case of squeeze needed
                        else:
                            vl_outputs = vl_outputs.squeeze()

                    # Compute Loss function
                    loss = self.criterion(vl_outputs, batch_y)

                    # Predictions
                    batch_pred_y = torch.round(vl_outputs)
                    correct_batch_pred_y = sum(
                        [float(1) for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
                    )
                    
                    # Compute Accuracy and Loss
                    vl_accuracy = correct_batch_pred_y / len(batch_pred_y)
                    vl_loss     = loss.item()

                    # Update history
                    self.history['vl_accuracy'].append(vl_accuracy)
                    self.history['vl_loss'].append(vl_loss)

                    # Updates the mean of the Accuracy and the Loss on TR set
                    self.mean_vl_accuracy = float((self.mean_vl_accuracy * self.vl_batch_counter + vl_accuracy) / (self.vl_batch_counter+1))
                    self.mean_vl_loss     = float((self.mean_vl_loss * self.vl_batch_counter + vl_loss) / (self.vl_batch_counter+1))

                    self.vl_batch_counter += 1
                
                # Check for Early Stopping counter's update
                if (self.mean_vl_loss - prev_mean_vl_loss) < self.tolerance:
                    print(counter)
                    counter += 1
                else:
                    counter = 0

                if counter == self.patience:
                    print(f'Early Stopping:\n\tpatience={self.patience} == counter={counter}\n\tmean_vl_loss-previous={self.mean_vl_loss-prev_mean_vl_loss}')
                    # Restore model weights to the initial state
                    self.load_state_dict(initial_weights)
                    break
                prev_mean_vl_loss = self.mean_vl_loss
                        

        # Returns the values computed
        return self.mean_tr_accuracy, self.mean_tr_loss, self.mean_vl_accuracy, self.mean_vl_loss


    def test(self, x_test, y_test):
        '''
            Evaluates the model on the Test set passed as parameter and returns a tuple of the following format: \
            (ts_loss, ts_accuracy)\n
            - x_test: a NumPy array MxN dataset used for Testing.\n
            - y_test: a NumPy array Mx1 labels used for Testing.
        '''

        test_dataset = MyDataset(
            torch.from_numpy(x_test).to(dtype=torch.float32),
            torch.from_numpy(y_test).to(dtype=torch.float32)
        )
        test_data = DataLoader(dataset=test_dataset, batch_size=int(self.params['batch_size']), shuffle=True)
        
        # Batch iteration on TS set
        for batch_x, batch_y in test_data:

            # Forward pass
            ts_outputs: torch.Tensor = self(batch_x)

            # Cases of squeeze
            if ts_outputs.size() != torch.Size([1]):
                # Case of squeeze don't needed (because we obtain a scalar)
                if ts_outputs.size() == torch.Size([1,1]):
                    ts_outputs = ts_outputs[0]
                # Case of squeeze don't needed (because we obtain a scalar)
                elif ts_outputs.size() == torch.Size([1,1,1]):
                    ts_outputs = ts_outputs[0][0]
                # Case of squeeze needed
                else:
                    ts_outputs = ts_outputs.squeeze()

            # Compute Loss function
            loss = self.criterion(ts_outputs, batch_y)
            
            # Optimization
            self.optimizer.step()

            # Predictions
            batch_pred_y = torch.round(ts_outputs)
            correct_batch_pred_y = sum(
                [1 for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
            )

            # Case of first assignment
            if self.y_true == None:
                self.y_true = batch_y
            # Case of concatenation of tensors
            else:
                self.y_true = self.y_true + batch_y
            
            # Case of first assignment
            if self.y_predictions == None:
                self.y_predictions = batch_pred_y
            # Case of concatenation of tensors
            else:
                self.y_predictions = self.y_predictions + batch_pred_y
            
            # Compute Accuracy and Loss
            ts_accuracy = float(correct_batch_pred_y / len(batch_pred_y))
            ts_loss     = loss.item()

            # Updates the history
            self.history['ts_accuracy'].append(ts_accuracy)
            self.history['ts_loss'].append(ts_loss)

            # Updates the mean of the Accuracy and the Loss on TR set
            self.mean_ts_accuracy = float((self.mean_ts_accuracy * self.ts_batch_counter + ts_accuracy) / (self.ts_batch_counter+1))
            self.mean_ts_loss     = float((self.mean_ts_loss * self.ts_batch_counter + ts_loss) / (self.ts_batch_counter+1))

            self.ts_batch_counter += 1
    
        return self.ts_loss, self.ts_accuracy

    
    def score(self):
        '''
            Evaluates the model computing the Beta1-score and the Beta2-score based on the Test set passed as parameter. \
            Returns a tuple of the following format: \
            (f1_score, f2_score)\n
            - x_test: a NumPy array MxN dataset used for Testing.\n
            - y_test: a NumPy array Mx1 labels used for Testing.
        '''

        # Compute Precision and Recall
        self.recall_score    = recall_score(y_true=self.y_true, y_pred=self.y_predictions)
        self.precision_score = precision_score(y_true=self.y_true, y_pred=self.y_predictions)

        # Compute the f1-score and f2-score
        self.f1_score = fbeta_score(y_true=self.y_true, y_pred=self.y_predictions, beta=1)
        self.f2_score = fbeta_score(y_true=self.y_true, y_pred=self.y_predictions, beta=2)

        return self.f1_score, self.f2_score


