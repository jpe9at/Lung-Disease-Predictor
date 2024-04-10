import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from Module import CNNLungs

class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs, batch_size = 8, early_stopping_patience=6, min_delta = 0.0007, num_gpus=0):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.num_epochs_no_improve = 0
        self.min_delta = min_delta
        assert num_gpus == 0, 'No GPU support yet'
    

    def prepare_training_data(self, data_train, data_val, batch_size):
        self.train_dataloader = data_train.train_dataloader(batch_size)
        self.val_dataloader = data_val.val_dataloader(batch_size)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, data_train, data_val):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(data_train, data_val, self.batch_size)
        self.prepare_model(model)
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, val_loss = self.fit_epoch()
            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss: .4f}, LR = {self.model.scheduler.get_last_lr() if self.model.scheduler is not None else self.model.learning_rate}')
            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            #########################################
            #Early Stopping Monitor
            #instead, we can also use the early stopping monitor class below. 
            if (self.best_val_loss - val_loss) > self.min_delta:
                self.best_val_loss = val_loss
                self.num_epochs_no_improve = 0
            else:
                self.num_epochs_no_improve += 1
                if self.num_epochs_no_improve == self.early_stopping_patience:
                    print("Early stopping at epoch", epoch)
                    break
            ########################################

            ########################################
            #Scheduler for adaptive learning rate
            if self.model.scheduler is not None:
                self.model.scheduler.step(val_loss)
            ########################################


    def fit_epoch(self):
        train_loss = 0.0
        for x_batch, y_batch in self.train_dataloader:
            output = self.model(x_batch)
            loss = self.model.loss(output, y_batch)
            self.model.optimizer.zero_grad()
            loss.backward()
            
            ######################################
            #L1 Loss
            if self.model.l1_rate != 0: 
                loss = self.model.l1_regularization(self.model.l2_rate)
            ######################################
            
            ######################################
            #Gradient Clipping
            if self.model.clip_val !=0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip_val)  # Gradient clipping
            ######################################

            self.model.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(self.train_dataloader.dataset)
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.val_dataloader:
                val_output = self.model(x_batch)
                loss = self.model.loss(val_output, y_batch)
                val_loss += loss.item() * x_batch.size(0) #why multiplication with 0?
            val_loss /= len(self.val_dataloader.dataset)
        return train_loss, val_loss

    def test(self, model, data_test):
        model.eval()
        self.test_loss = 0.0
        self.test_dataloader = data_test.get_dataloader(self.batch_size)
        y_hat_total = torch.zeros(1,3)
        y_total = torch.zeros(1,3)
        with torch.no_grad():
            for X,y in self.test_dataloader:
                y_hat = torch.argmax(model(X), dim=1)  # Choose the class with highest probability
                loss = self.calculate_accuracy(y_hat, y)
                #loss = self.model.metric(y_hat, y)
                self.test_loss += loss * X.size(0)
        self.test_loss /= len(self.test_dataloader.dataset)

        return self.test_loss

    def calculate_accuracy(self,predictions, labels):
        # Get the predicted classes by selecting the index with the highest probabilityc
        _, predicted_classes = torch.max(predictions, 0)
        # Compare predictions with ground truth
        correct_predictions = torch.eq(predicted_classes, labels).sum().item()
        # Calculate accuracy
        accuracy = correct_predictions / labels.size(0)
    
        return accuracy
    @classmethod
    def Optuna_objective(cls, trial,train_data, val_data):
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        learning_r = trial.suggest_float("learning_rate", 1e-6, 1e-4)
        batch_size = trial.suggest_categorical("batch_size", [8])
        hidden_size = trial.suggest_categorical('hidden_size',[32,64])
        param_init = trial.suggest_categorical('weight_initialisation', [None,(None, 'Xavier'), (None, 'He')])
        l2_rate = trial.suggest_categorical('l2_rate', [0.15,0.1,0.01, 0.07])

        cnn_model = CNNLungs(hidden_size, learning_rate = learning_r, l2 = l2_rate, scheduler = 'OnPlateau', param_initialisation = param_init, optimizer = optimizer)
        trainer = cls(50,  batch_size)
        trainer.fit(cnn_model, train_data, val_data)

        return  trainer.val_loss_values[-1]

    @classmethod
    def hyperparameter_optimization(cls, train_data, val_data):
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: cls.Optuna_objective(trial, train_data, val_data)
        study.optimize(objective_func, n_trials=15)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value

        return best_params, best_accuracy

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss



