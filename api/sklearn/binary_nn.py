from keras import Model, Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc 
from sklearn.base import BaseEstimator, ClassifierMixin







class BinaryNN(BaseEstimator, ClassifierMixin):
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
    


    def get_params(self, deep=True):
        # Return the parameters of your estimator as a dictionary
        # Make sure to include all hyperparameters that affect the model
        return self.params

    def print_plot(self):
        '''
            Prints the plot based on the history of the trained model.
        '''
        
        # Error case
        if self.history is None:
            raise ValueError

        # Print of the Plot
        plt.figure()
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.legend()
    

    def print_roc_curve(self, y_test):
        '''
            Prints
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


    def create_model(self, n_hidden_layers: int) -> Model:
        '''
            Returns a sklearn model with "n_hidden_layers" layers (input, hidden, output) created with the parameters \
            passed to the object. Returns the model created.\n
            - n_hidden_layers: number of hidden layers (bigger than 0).
        '''

        if n_hidden_layers < 0:
            raise ValueError("Number of hidden layers must be non-negative.")

        if n_hidden_layers == 0:
            # Single-layer perceptron (no hidden layers)
            self.model = MLPClassifier(
                hidden_layer_sizes=(),
                #??????????????????activation=self.params['input_activation'],
                alpha=self.params['weight_decay'],
                learning_rate_init=self.params['learning_rate'],
                momentum=self.params['momentum'],
                solver='sgd',
                early_stopping=True, 
                nesterovs_momentum=True,
                max_iter=self.params['epochs'],  # Number of epochs
                batch_size=self.params["batch_size"], 
                tol=1e-4,         # Tolerance to control the improvement in the val loss
                n_iter_no_change=10  # patience
            )
        else:
            # Create a list to hold the hidden layer sizes
            hidden_layer_sizes = [self.params['hidden_units']] * n_hidden_layers

            # Build the MLPClassifier model with hidden layers
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=self.params['hidden_activation'],
                alpha=self.params['weight_decay'],
                learning_rate_init=self.params['learning_rate'],
                momentum=self.params['momentum'],
                solver='sgd',
                early_stopping=True, 
                nesterovs_momentum=True,
                max_iter=self.params['epochs'],  # Number of epochs
                batch_size=self.params["batch_size"], 
                tol=1e-4,         # Tolerance to control the improvement in the val loss
                n_iter_no_change=10  # patience
            )

        return self.model

    
    def fit(self, x_train, y_train):
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
        
        # Training of the model with only TR set
        self.model.set_params(validation_fraction=0.0)
        self.history = self.model.fit(
            x=x_train,  
            y=y_train
            )

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

        # Evaluation on VL set
        if x_val is not None and y_val is not None:
            vl_loss, vl_accuracy = self.model.evaluate(x=x_val, y=y_val, verbose=0)
            self.mean_vl_accuracy = float((self.mean_vl_accuracy * self.k_fold_counter + vl_accuracy) / (self.k_fold_counter + 1))
            self.mean_vl_loss = float((self.mean_vl_loss * self.k_fold_counter + vl_loss) / (self.k_fold_counter + 1))

        # Update of the trials
        self.k_fold_counter += 1

        # Return in case of evaluation on TR and VL set
        if x_val is not None and y_val is not None:
            return (tr_loss, tr_accuracy, vl_loss, vl_accuracy)
        
        # Return in case of evaluation on TR set only
        else:
            return (tr_loss, tr_accuracy)


    def test(self, x_test, y_test):
        '''
            Evaluates the model on the Test set passed as parameter and returns a tuple of the following format: \
            (ts_loss, ts_accuracy)\n
            - x_test: a NumPy array MxN dataset used for Testing.\n
            - y_test: a NumPy array Mx1 labels used for Testing.
        '''

        # Error case
        if self.model is None:
            raise ValueError
        
        # Testing of the model
        self.ts_loss, self.ts_accuracy = self.model.evaluate(
            x=x_test,
            y=y_test,
            batch_size=self.params['batch_size'],
            verbose=0
        )

        return (self.ts_loss, self.ts_accuracy)

    
    def score(self, x_test, y_test):
        '''
            Evaluates the model computing the Beta1-score and the Beta2-score based on the Test set passed as parameter. \
            Returns a tuple of the following format: \
            (f1_score, f2_score)\n
            - x_test: a NumPy array MxN dataset used for Testing.\n
            - y_test: a NumPy array Mx1 labels used for Testing.
        '''

        # Error case
        if self.model is None:
            raise ValueError
        
        # Predictions probability of the model
        y_predictions_prob = self.model.predict(
            x=x_test,
            batch_size=self.params['batch_size'],
            verbose=0
        )

        # Converting the Probabilities into Categorized values
        self.y_predictions = [round(prediction[0]) for prediction in y_predictions_prob]

        # Compute Precision and Recall
        self.recall_score    = recall_score(y_true=y_test, y_pred=self.y_predictions)
        self.precision_score = precision_score(y_true=y_test, y_pred=self.y_predictions)

        # Compute the f1-score and f2-score
        self.f1_score = fbeta_score(y_true=y_test, y_pred=self.y_predictions, beta=1)
        self.f2_score = fbeta_score(y_true=y_test, y_pred=self.y_predictions, beta=2)

        return (self.f1_score, self.f2_score)


