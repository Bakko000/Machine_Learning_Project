from keras import Model, Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from keras.initializers import glorot_uniform
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc 



class BinaryNN():
    '''
        Class which offers methods to handle Neural Networks for Binary Classification tasks with at least 3 layers \
        (input, hidden and output) based on a dictionary of associations <hyperparameter_key,hyperparameter_value> \
        with the following keys: \n
        'input_units', 'hidden_units', 'learning_rate', 'momentum', 'activation', 'output_activation', 'metrics'.
    '''


    def __init__(self, params: dict, monk_i: int, trial: int):

        # Passed values initializations
        self.params = params
        self.monk_i = monk_i
        self.trial = trial

        # Default's values initializations
        self.vlacc_variance   = 0
        self.vlacc_devstd     = 0
        self.tracc_variance   = 0
        self.tracc_devstd     = 0
        self.vl_variance      = 0 
        self.vl_devstd        = 0
        self.tr_variance      = 0 
        self.tr_devstd        = 0
        self.mean_tr_accuracy = 0
        self.mean_vl_accuracy = 0
        self.ts_accuracy      = 0
        self.mean_tr_loss     = 0
        self.mean_vl_loss     = 0
        self.k_fold_counter   = 0
        self.ts_loss          = 0
        self.f1_score         = 0
        self.f2_score         = 0
        self.recall_score     = 0
        self.precision_score  = 0
        self.y_predictions    = []
        self.tr_losses        = [] 
        self.vl_losses        = []
        self.vl_accuracies    = [] 
        self.tr_accuracies    = [] 
        self.mean_tr_loss_list = []
        self.mean_vl_loss_list = []
        self.mean_tr_acc_list = []
        self.mean_vl_acc_list = []
        self.model            = None
        self.history          = None

    
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
    

    
    def mean_euclidean_error(self, y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


    def print_plot(self):
        '''
            Prints the plot based on the history of the trained model.
        '''
        
        # Error case
        if self.history is None:
            raise ValueError

        # Print of the Plot
        plt.figure()
        plt.plot(self.mean_tr_loss_list, label='Training MSE')
        plt.plot(self.mean_vl_loss_list, label='Validation MSE', linestyle='--')
        plt.title('Model MSE')
        plt.xlabel('Epoch')
        plt.legend()

        # Print of the Plot
        plt.figure()
        plt.plot(self.mean_tr_acc_list, label='Training MEE')
        plt.plot(self.mean_vl_acc_list, label='Validation MEE', linestyle='--')
        plt.title('Model MEE')
        plt.xlabel('Epoch')
        plt.legend()
    


    def print_training_info(self):
        '''
            Prints the results of the Training Phase.
        '''
        print(
            f" Monk:                          {self.monk_i}\n" + \
            f" Trial:                         {self.trial}\n" + \
            f" Hyperparameters:               {self.params}\n" + \
            f" Mean Training MSE:             {self.mean_tr_loss}\n" + \
            f" Mean Validation MSE:           {self.mean_vl_loss}\n" + \
            f" Mean Training MEE:             {self.mean_tr_accuracy}\n" + \
            f" Mean Validation MEE:           {self.mean_vl_accuracy}\n" + \
            #f" Standard Deviation TR Accuracy:    {self.tracc_devstd}\n" + \
            f" Standard Deviation VL MEE:     {self.vlacc_devstd}\n" + \
            #f" Variance TR Accuracy:              {self.tracc_variance}\n" + \
            f" Variance VL MEE:               {self.vlacc_variance}\n" + \
            f" Standard Deviation VL MEE:     {self.vl_devstd}\n" + \
            #f" Standard Deviation TR Loss:        {self.tr_devstd}\n" + \
            f" Variance VL MEE:               {self.vl_variance}\n"
            #f" Variance TR Loss:             {self.tr_variance}"
        )
    


    def create_model(self, n_hidden_layers: int) -> Model:
        '''
            Returns a Sequential Keras model with "n_hidden_layers" layers (input, hidden, output) created with the parameters \
            passed to the object. Returns the model created.\n
            - n_hidden_layers: number of hidden layers (bigger than 0).
        '''

        # Error case
        if n_hidden_layers < 0:
            raise ValueError

        # Build the sequential model
        model = Sequential()

        # Hidden Layers
        for _ in range(n_hidden_layers):
            model.add(
                Dense(
                    units=self.params['hidden_units'],
                    activation=self.params['activation'],
                    kernel_regularizer=l2(self.params['weight_decay']),
                    kernel_initializer=glorot_uniform(seed=15),
                    use_bias=True
                )
            )
        
        # Output Layer
        model.add(Dense(units=3, activation=self.params['output_activation'], use_bias=True))

        # Sets the Loss Function, the Optimizer (Stochastic Gradient Descent) and the Metrics used for evaluation
        model.compile(
            loss='mean_squared_error',
            optimizer=SGD(
                learning_rate=self.params['learning_rate'],
                momentum=self.params['momentum'],
                nesterov=self.params['nesterov']
            ),
            metrics=[self.mean_euclidean_error]
        )

        # Saving the model
        self.model = model

        return self.model

    
    def fit(self, x_train, y_train, x_val=None, y_val=None, retraining=False):
        '''
            Train the model based on the data passed as parameters and returns the history.\n
            - x_train: a NumPy array MxN dataset used for Training.\n
            - y_train: a NumPy Mx1 labels used for Training.\n
            - x_val: a NumPy array MxN dataset used for Validation.\n
            - y_val: a NumPy Mx1 labels used for Validation.
        '''

        # Error case
        if self.model is None:
            raise ValueError
        
        if retraining:
        
            # Training of the model on entire dataset for retraining
            self.history = self.model.fit(
                    x=x_train,
                    y=y_train,
                    epochs=self.params['epochs'],
                    batch_size=self.params['batch_size'],
                    verbose=0,
                    )
        
        # Training of the model with only TR set
        if x_val is None and y_val is None:
            self.history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=self.params["patience"], restore_best_weights=True)],
                verbose=0,
                shuffle=True
            )
        
        # Training of the model with TR set and VL set (already splitted)
        elif x_val is not None and y_val is not None:
            self.history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=(x_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=self.params["patience"], restore_best_weights=True)],
                verbose=0,
                shuffle=True
            )
            # Save the training history for this trial
            self.mean_vl_loss_list += self.history.history["val_loss"]
            self.mean_vl_acc_list += self.history.history["val_mean_euclidean_error"]
        
        # Error case
        else:
            raise ValueError

         # Save the training history for this trial
        self.mean_tr_loss_list += self.history.history["loss"]
        self.mean_tr_acc_list += self.history.history["mean_euclidean_error"]

        # Returns the history
        return self.history
    

    def evaluate(self, x_train, y_train, x_val=None, y_val=None):
        '''
            Evaluates the model on the Training set passed as parameter and returns a tuple of the following format: \
            (tr_loss, tr_accuracy, vl_loss, vl_accuracy)\n
            - x_train: a NumPy array MxN dataset used for Training.\n
            - y_train: a NumPy array Mx1 labels used for Training.\n
            - x_val: a NumPy array MxN dataset used for Validation.\n
            - y_val: a NumPy array Mx1 labels used for Validation.
        '''

        # Error case
        if self.history is None or self.model is None:
            raise ValueError
        
        # Evaluation on TR set
        tr_loss, tr_accuracy = self.model.evaluate(x=x_train, y=y_train, verbose=0)
        self.mean_tr_accuracy = float((self.mean_tr_accuracy * self.k_fold_counter + tr_accuracy) / (self.k_fold_counter + 1))
        self.mean_tr_loss = float((self.mean_tr_loss * self.k_fold_counter + tr_loss) / (self.k_fold_counter + 1))
        self.tr_losses.append(tr_loss)
        self.tr_accuracies.append(tr_accuracy)
        self.tr_variance = np.var(self.tr_losses)
        self.tr_devstd = np.std(self.tr_losses)
        self.tracc_variance = float(np.var(self.tr_accuracies))
        self.tracc_devstd = float(np.std(self.tr_accuracies))

        # Evaluation on VL set
        if x_val is not None and y_val is not None:
            vl_loss, vl_accuracy = self.model.evaluate(x=x_val, y=y_val, verbose=0)
            self.mean_vl_accuracy = float((self.mean_vl_accuracy * self.k_fold_counter + vl_accuracy) / (self.k_fold_counter + 1))
            self.mean_vl_loss = float((self.mean_vl_loss * self.k_fold_counter + vl_loss) / (self.k_fold_counter + 1))
            self.vl_losses.append(vl_loss)
            self.vl_accuracies.append(vl_accuracy)
            self.vl_variance = float(np.var(self.vl_losses))
            self.vl_devstd = float(np.std(self.vl_losses))
            self.vlacc_variance = float(np.var(self.vl_accuracies))
            self.vlacc_devstd = float(np.std(self.vl_accuracies))

        # Update of the trials
        self.k_fold_counter += 1

        # Return in case of evaluation on TR and VL set
        if x_val is not None and y_val is not None:
            return (tr_loss, tr_accuracy, vl_loss, vl_accuracy)
        
        # Return in case of evaluation on TR set only
        else:
            return (tr_loss, tr_accuracy)
        

    def predict(self, x_its, y_its):
        # predict on internal test set
        y_ipred = self.model.predict(x_its, verbose=1)
        score = self.mean_euclidean_error(y_its, y_ipred)
        return score, y_ipred


    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R-squared for each output variable and average R-squared.

        Parameters:
        - y_true: True labels (numpy array with shape [n_samples, n_outputs])
        - y_pred: Predicted values (numpy array with shape [n_samples, n_outputs])

        Returns:
        - r2_scores: List of R-squared scores for each output variable
        - average_r2: Average R-squared across all output variables
        """
        # Calculate R-squared for each output variable
        r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

        # Average R-squared across all output variables
        average_r2 = np.mean(r2_scores)

        return r2_scores, average_r2

