import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc 



class MyDataset():

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        





class NN(nn.Module):
    '''
        Class which offers methods to handle Neural Networks for Binary Classification's and Regression's tasks with \
        customization of the number of layers, units, and whatever else based on a dictionary of associations like: \
        <hyperparameter_key,hyperparameter_value>.
    '''


    def __init__(self, params: dict, current_trial: int,  trials: int, monk_i=0):
        super(NN, self).__init__()

        # Sets the device used
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(device)

        # Passed values initializations
        self.params = params
        self.monk_i = monk_i
        self.trial  = current_trial
        self.trials = trials

        # Default's values initializations
        self.history: dict[list] = {'tr_metric':[], 'vl_metric':[], 'tr_loss':[], 'vl_loss':[], 'ts_metric':[], 'ts_loss':[]}
        self.mean_ts_metric      = 0
        self.mean_tr_metric      = 0
        self.mean_ts_loss        = 0
        self.mean_vl_metric      = 0
        self.ts_metric           = 0
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
        self.patience            = 10
        self.y_pred: torch.Tensor = None
        self.y_true: torch.Tensor = None
        
        # Choice of Hidden Activation Function
        if self.params['hidden_activation'] == 'Sigmoid':
            hidden_activation = nn.Sigmoid()
        elif self.params['hidden_activation'] == 'Tanh':
            hidden_activation = nn.Tanh()
        elif self.params['hidden_activation'] == 'ReLU':
            hidden_activation = nn.ReLU()
        elif self.params['hidden_activation'] == '':
            hidden_activation = None
        else:
            raise ValueError
        
        # Choice of Output Activation Function
        if self.params['output_activation'] == 'Sigmoid':
            output_activation = nn.Sigmoid()
        elif self.params['output_activation'] == 'Tanh':
            output_activation = nn.Tanh()
        elif self.params['output_activation'] == 'ReLU':
            output_activation = nn.ReLU()
        elif self.params['output_activation'] == 'Linear':
            output_activation = nn.Linear(in_features=self.params['hidden_size'], out_features=self.params['output_size'])
        elif self.params['output_activation'] == '':
            output_activation = None
        else:
            raise ValueError
    
        # Adding layers Dynamically from an empty MouleList
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.params['input_size'], self.params['hidden_size']))
        if hidden_activation != None:
            self.layers.append(hidden_activation)
        for _ in range(self.params['hidden_layers']-1):
            self.layers.append(nn.Linear(self.params['hidden_size'], self.params['hidden_size']))
            if hidden_activation != None:
                self.layers.append(hidden_activation)
        self.layers.append(nn.Linear(self.params['hidden_size'], self.params['output_size']))
        if output_activation != None:
            self.layers.append(output_activation)

        # Initialization of the Weights
        self.init_weights()

        # Loss Function
        self.criterion = nn.MSELoss()
        
        # Optimizer initialization
        if self.params['weight_decay'] == 0:
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.params['learning_rate'],
                momentum=self.params['momentum'],
                nesterov=self.params['nesterov']
            )
        else:
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.params['learning_rate'],
                momentum=self.params['momentum'],
                nesterov=self.params['nesterov'],
                weight_decay=self.params['weight_decay']
            )

    
    def __str__(self) -> str:
        if self.monk_i > 0:
            header = f" Monk:                     {self.monk_i}\n"
            tail = \
                f" f1 score:                 {self.f1_score}\n" + \
                f" f2 score:                 {self.f2_score}\n" + \
                f" Prediction score:         {self.precision_score}\n" + \
                f" Recall score:             {self.recall_score}\n"
        else:
            header = ''
            tail = ''
        
        metric = self.params['metrics']+':' if self.params['metrics'] == 'Accuracy' else str(self.params['metrics'] + ':     ')

        return header + \
            f" Trial:                    {self.trial}\n" + \
            f" Hyperparameters:          {self.params}\n" + \
            f" Mean Training Loss:       {self.mean_tr_loss}\n" + \
            f" Mean Validation Loss:     {self.mean_vl_loss}\n" + \
            f" Mean Test Loss:           {self.mean_ts_loss}\n" + \
            f" Mean Training {metric}   {self.mean_tr_metric}\n" + \
            f" Mean Validation {metric} {self.mean_vl_metric}\n" + \
            f" Mean Test {metric}       {self.mean_ts_metric}\n" + tail
    

    def init_weights(self):
        '''
            Initializes the weights of the layers with default values.
        '''
        if 'seed_init' in self.params.keys():
            torch.manual_seed(self.params['seed_init'])
        for module in self.layers:
            if isinstance(module, nn.Linear):
                #print(f"Before initialization - Layer {module}: {module.weight.shape}") # debug
                if(self.params["weight_init"] == "glorot_uniform"):
                    nn.init.xavier_uniform_(module.weight)
                elif(self.params["weight_init"] == "glorot_normal"):
                    nn.init.xavier_normal_(module.weight)
                elif(self.params["weight_init"] == "he_normal"):
                    nn.init.kaiming_normal_(module.weight)
                elif(self.params["weight_init"] == "he_uniform"):
                    nn.init.kaiming_uniform_(module.weight)
                else:
                    raise ValueError
                #print(f"After initialization - Layer {module}: {module.weight.shape}") # debug


    def print_acc_plot(self):
        '''
            Prints the plot based on the accuracy of the trained model.
        '''
        label = 'Accuracy' if self.params['metrics'] == 'Accuracy' else 'MEE'
            
        plt.figure()
        plt.plot(self.history['tr_metric'], label=f'Training {label}')
        plt.plot(self.history['vl_metric'], label=f'Validation {label}', linestyle='--')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend()
    

    def print_loss_plot(self):
        '''
            Prints the plot based on the loss of the trained model.
        '''
        plt.figure()
        plt.plot(self.history['tr_loss'], label='Training Loss')
        plt.plot(self.history['vl_loss'], label='Validation Loss', linestyle='--')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    

    def print_roc_curve(self):
        '''
            Prints the ROC curve's plot.
        '''
        # Calculate the ROC curve 
        fpr, tpr, thresholds = roc_curve(
            self.y_true.detach().numpy(),
            self.y_pred.detach().numpy()
        )
        
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
        if self.monk_i > 0:
            string = f" Monk:                     {self.monk_i}\n"
        else:
            string = ''
        
        metric = self.params['metrics']+':' if self.params['metrics'] == 'Accuracy' else str(self.params['metrics'] + ':     ')

        print(
            string + \
            f" Trial:                    {self.trial}/{self.trials}\n" + \
            f" Hyperparameters:          {self.params}\n" + \
            f" Mean Training Loss:       {self.mean_tr_loss}\n" + \
            f" Mean Validation Loss:     {self.mean_vl_loss}\n" + \
            f" Mean Training {metric}   {self.mean_tr_metric}\n" + \
            f" Mean Validation {metric} {self.mean_vl_metric}"
        )
    

    def print_confusion_matrix(self):
        '''
            Prints the confusion matrix based on the predictions made during the Testing Phase and on the true values.
        '''
        
        # Prints the Confusion Matrix as a DataFrame (alternative: tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_predictions).ravel())
        print(
            pd.DataFrame(
                data=confusion_matrix(y_true=self.y_true.detach().numpy(), y_pred=self.y_pred.detach().numpy()),
                index=['Real_Class_0', 'Real_Class_1'],
                columns=['Predicted_Class_0', 'Predicted_Class_1']
            )
        )
    
    
    def is_better_model_than(self, model):
        '''
            Returns True if this (self) model has better metrics (Accuracy or MEE) than the model passed as parameter, \
            otherwise False.
            - model: a model of the same type (class type NN) of this (self).
        '''
        # Case of metric used: Accuracy
        if self.params['metrics'] == 'Accuracy':

            # Case of self better model than model
            if (
                    self.mean_tr_metric+0.02 <= self.mean_vl_metric \
                    and self.mean_tr_metric-0.05 <= self.mean_vl_metric
                    and (
                        model.mean_vl_metric < self.mean_vl_metric \
                        or (
                            model.mean_vl_metric == self.mean_vl_metric \
                            and model.mean_tr_metric < self.mean_tr_metric
                        )
                    )
                ) or (
                    self.mean_vl_metric > model.mean_vl_metric \
                    and (
                        abs(self.mean_tr_metric - self.mean_vl_metric) < abs(model.mean_tr_metric - model.mean_vl_metric) \
                        or abs(self.mean_tr_metric - self.mean_vl_metric) < 0.02 or self.monk_i == 3
                    )
                ):
                return True
        
        # Case of metric used: MEE
        elif self.params['metrics'] == 'MEE':
            
            # Case of self better model than model
            if self.mean_vl_metric < model.mean_vl_metric \
                and abs(self.mean_tr_loss - self.mean_vl_loss) < 0.1:
                return True
        
        # Case of metric not supported
        else:
            raise ValueError(f"Metric '{self.params['metrics']}' not supported.")
            
        return False
    

    def forward(self, x):
        '''
            Executes the forwarding pass.
        '''
        #print(f"forward 0/{len(self.layers)}: " + str(x.size())) # debug
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            #print(f"forward {i+1}/{len(self.layers)}: " + str(x.size()) + f"  | Layer = {str(self.layers[i])}") # debug
        return x

    
    def fit(self, x_train, y_train, x_val=None, y_val=None):
        '''
            Train the model based on the data passed as parameters and returns the history.\n
            - x_train: a NumPy array MxN dataset used for Training.\n
            - y_train: a NumPy Mx1 labels used for Training.\n
            - x_val: a NumPy array MxN dataset used for Validation (if None will use only TR set).\n
            - y_val: a NumPy Mx1 labels used for Validation (if None will use only TR set).
        '''

        # Creates the Iterable dataset
        train_dataset = MyDataset(
            torch.from_numpy(x_train).to(dtype=torch.float32),
            torch.from_numpy(y_train).to(dtype=torch.float32)
        )
        train_data = DataLoader(dataset=train_dataset, batch_size=int(self.params['batch_size']), shuffle=True)

        # Case of Retraining (Validation not necessary)
        if x_val is not None and y_val is not None:

            # Creates the Iterable dataset
            val_dataset = MyDataset(
                torch.from_numpy(x_val).to(dtype=torch.float32),
                torch.from_numpy(y_val).to(dtype=torch.float32)
            )
            val_data = DataLoader(dataset=val_dataset, batch_size=int(self.params['batch_size']), shuffle=True)

        else:
            val_dataset = None
            val_data = None
        
        # Counter for Early Stopping
        earlystop_counter = 0

        # Epochs iteration
        for epoch in range(self.params['epochs']):

            # Batch iteration on TR set
            for batch_x, batch_y in train_data:
                
                # Resets gradients
                self.optimizer.zero_grad()

                # Forward pass
                tr_outputs: torch.Tensor = self(batch_x)
                if torch.any(torch.isnan(tr_outputs)):
                    print(f"NaN trovato DOPO forward")
                    #print(tr_outputs)
                    #print(batch_x)
                    return

                # Compute Loss function (MSE)
                loss = self.criterion(tr_outputs, batch_y)

                # Backward pass
                loss.backward()
                
                # Optimization
                self.optimizer.step()
                
                # Computes the TR loss
                tr_loss = loss.item()

                # Case of metric Accuracy
                if self.params['metrics'] == 'Accuracy':
                    batch_pred_y = torch.round(tr_outputs)
                    correct_batch_pred_y = sum(
                        [1 for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
                    )
                    tr_metric = float(correct_batch_pred_y / len(batch_pred_y))
                    #print(f'[TR] batch_pred_y.size()={batch_pred_y.size()} tr_metric={tr_metric} correct_batch_pred_y={correct_batch_pred_y} batch_pred_y={len(batch_pred_y)}')
                
                # Case of metric Mean Euclidian Error
                elif self.params['metrics'] == 'MEE':
                    batch_pred_y = tr_outputs
                    if torch.any(torch.isnan(batch_pred_y)):
                        batch_not_nan_indexes = ~torch.isnan(batch_pred_y)
                        batch_pred_y = batch_pred_y[batch_not_nan_indexes] # Remove NaN values
                        batch_y = batch_y[batch_not_nan_indexes] # Remove NaN values
                        #print("[TR] NaN trovato")
                    mee = torch.mean(
                        torch.norm(
                            batch_pred_y - batch_y,
                            dim=1
                        )
                    )
                    tr_metric = mee.item()

                # Case of error
                else:
                    raise ValueError('this metric is not supported. Try with "accuracy" or "MEE".')

                # Updates the mean of the Accuracy and the Loss on TR set
                self.mean_tr_metric = float((self.mean_tr_metric * self.tr_batch_counter + tr_metric) / (self.tr_batch_counter+1))
                self.mean_tr_loss   = float((self.mean_tr_loss * self.tr_batch_counter + tr_loss) / (self.tr_batch_counter+1))

                self.tr_batch_counter += 1
            
            # Updates the history
            self.history['tr_metric'].append(self.mean_tr_metric)
            self.history['tr_loss'].append(self.mean_tr_loss)
        
            # Case of Retraining (Validation not necessary)
            if val_data is None:
                return self.mean_tr_metric, self.mean_tr_loss, self.mean_vl_metric, self.mean_vl_loss
        
            # Evaluation on VL set
            with torch.no_grad():

                # Previous Mean Loss on VL set
                prev_mean_vl_loss = 0

                # Batch iteration on VL set
                for batch_x, batch_y in val_data:

                    # Forward pass
                    vl_outputs: torch.Tensor = self(batch_x)

                    # Compute Loss function (MSE)
                    loss = self.criterion(vl_outputs, batch_y)

                    # Computes the TR loss
                    vl_loss = loss.item()

                    # Case of metric Accuracy
                    if self.params['metrics'] == 'Accuracy':
                        batch_pred_y = torch.round(vl_outputs)
                        correct_batch_pred_y = sum(
                            [1 for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
                        )
                        vl_metric = float(correct_batch_pred_y / len(batch_pred_y))
                        #print(f'[VL] vl_metric={vl_metric} correct_batch_pred_y={correct_batch_pred_y} batch_pred_y={len(batch_pred_y)}')
                    
                    # Case of metric Mean Euclidian Error
                    elif self.params['metrics'] == 'MEE':
                        batch_pred_y = vl_outputs
                        if torch.any(torch.isnan(batch_pred_y)):
                            batch_not_nan_indexes = ~torch.isnan(batch_pred_y)
                            batch_pred_y = batch_pred_y[batch_not_nan_indexes] # Remove NaN values
                            batch_y = batch_y[batch_not_nan_indexes] # Remove NaN values
                            #print("[VL] NaN trovato")
                        mee = torch.mean(
                            torch.norm(
                                batch_pred_y - batch_y,
                                dim=1
                            )
                        )
                        vl_metric = mee.item()

                    # Case of error
                    else:
                        raise ValueError('this metric is not supported. Try with "accuracy" or "MEE".')

                    # Updates the mean of the Accuracy and the Loss on TR set
                    self.mean_vl_metric = float((self.mean_vl_metric * self.vl_batch_counter + vl_metric) / (self.vl_batch_counter+1))
                    self.mean_vl_loss   = float((self.mean_vl_loss * self.vl_batch_counter + vl_loss) / (self.vl_batch_counter+1))

                    self.vl_batch_counter += 1
                
                # Check for Early Stopping counter's update
                if (self.mean_vl_loss - prev_mean_vl_loss) < self.params['tolerance']:
                    earlystop_counter += 1
                else:
                    earlystop_counter = 0

                # Case of exit caused by Early Stopping
                if earlystop_counter == self.patience:
                    break
                
                prev_mean_vl_loss = self.mean_vl_loss
            
            # Update history
            self.history['vl_metric'].append(self.mean_vl_metric)
            self.history['vl_loss'].append(self.mean_vl_loss)

        # Returns the values computed
        return self.mean_tr_metric, self.mean_tr_loss, self.mean_vl_metric, self.mean_vl_loss
    

    def predict(self, x_test):
        '''
            Returns the predictions based on the given x_test dataset.
            - x_test: the x values of the dataset as NumPy array.
        '''
        test_dataset = MyDataset(
            torch.from_numpy(x_test).to(dtype=torch.float32),
            torch.tensor([1 for _ in range(x_test.shape[0])])
        )
        test_data = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        # Predictions list
        predictions = []
        
        # Batch iteration on TS set
        for x, _ in test_data:

            # Forward pass
            ts_outputs: torch.Tensor = self(x)
            
            # Optimization
            self.optimizer.step()

            # Append the predicted value
            predictions.append(ts_outputs[0].tolist())
        
        return predictions


    def test(self, x_test, y_test):
        '''
            Evaluates the model on the Test set passed as parameter.\n
            Returns a tuple of the following format: (ts_loss, ts_accuracy).\n
            - x_test: a NumPy array MxN dataset used for Testing.\n
            - y_test: a NumPy array Mx1 labels used for Testing.
        '''
        #self.y_true = torch.from_numpy(y_test).to(dtype=torch.float32)

        test_dataset = MyDataset(
            torch.from_numpy(x_test).to(dtype=torch.float32),
            torch.from_numpy(y_test).to(dtype=torch.float32)
        )
        test_data = DataLoader(dataset=test_dataset, batch_size=int(self.params['batch_size']), shuffle=True)
        
        # Batch iteration on TS set
        for batch_x, batch_y in test_data:

            # Forward pass
            ts_outputs: torch.Tensor = self(batch_x)

            # Compute Loss function
            loss = self.criterion(ts_outputs, batch_y)
            
            # Optimization
            self.optimizer.step()

            # Computes the TR loss
            ts_loss = loss.item()

            # Case of metric Accuracy
            if self.params['metrics'] == 'Accuracy':
                batch_pred_y = torch.round(ts_outputs)
                correct_batch_pred_y = sum(
                    [1 for batch_pred_y_i, batch_y_i in zip(batch_pred_y, batch_y) if batch_pred_y_i == batch_y_i]
                )
                ts_metric = float(correct_batch_pred_y / len(batch_pred_y))
                #print(f'[TS] ts_metric={ts_metric} correct_batch_pred_y={correct_batch_pred_y} batch_pred_y={len(batch_pred_y)}')
            
            # Case of metric Mean Euclidian Error
            elif self.params['metrics'] == 'MEE':
                batch_pred_y = ts_outputs
                if torch.any(torch.isnan(batch_pred_y)):
                    batch_not_nan_indexes = ~torch.isnan(batch_pred_y)
                    batch_pred_y = batch_pred_y[batch_not_nan_indexes] # Remove NaN values
                    batch_y = batch_y[batch_not_nan_indexes] # Remove NaN values
                    #print("[TR] NaN trovato")
                mee = torch.mean(
                    torch.norm(
                        batch_pred_y - batch_y,
                        dim=1
                    )
                )
                ts_metric = mee.item()

            # Case of error
            else:
                raise ValueError('this metric is not supported. Try with "accuracy" or "MEE".')

            # Updates the mean of the Accuracy and the Loss on TR set
            self.mean_ts_metric = float((self.mean_ts_metric * self.ts_batch_counter + ts_metric) / (self.ts_batch_counter+1))
            self.mean_ts_loss   = float((self.mean_ts_loss * self.ts_batch_counter + ts_loss) / (self.ts_batch_counter+1))

            # Case of first assignment
            if self.y_true == None:
                self.y_true = batch_y
            # Case of concatenation of tensors
            else:
                self.y_true = torch.concat([self.y_true, batch_y], dim=0)
            
            # Case of first assignment
            if self.y_pred == None:
                self.y_pred = batch_pred_y
            # Case of concatenation of tensors
            else:
                self.y_pred = torch.concat([self.y_pred, batch_pred_y], dim=0)

            # Updates the history
            self.history['ts_metric'].append(ts_metric)
            self.history['ts_loss'].append(ts_loss)

            self.ts_batch_counter += 1
    
        return self.mean_ts_loss, self.mean_ts_metric

    
    def score(self):
        '''
            Evaluates the model computing the Beta1-score and the Beta2-score based on the predictions and on the true values.\n
            Returns a tuple of the following format: (f1_score, f2_score)
        '''

        # Compute Precision and Recall
        self.recall_score    = recall_score(y_true=self.y_true.detach().numpy(), y_pred=self.y_pred.detach().numpy())
        self.precision_score = precision_score(y_true=self.y_true.detach().numpy(), y_pred=self.y_pred.detach().numpy())

        # Compute the f1-score and f2-score
        self.f1_score = fbeta_score(y_true=self.y_true.detach().numpy(), y_pred=self.y_pred.detach().numpy(), beta=1)
        self.f2_score = fbeta_score(y_true=self.y_true.detach().numpy(), y_pred=self.y_pred.detach().numpy(), beta=2)

        return self.f1_score, self.f2_score


