from torch import nn, optim
import torch
import torch.nn.init as init
import torch.nn.functional as F


class CNNLungs(nn.Module):
    def __init__(self, hidden_size= 128,  output_size = 3, optimizer = 'Adam', learning_rate = 0.0001,loss_function = 'CEL', l1 = 0.0, l2 = 0.0, clip_val=0, scheduler = None, param_initialisation = None):
        super(CNNLungs, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 20 * 20, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )

        self._init_weights = 'Default'

        self.optimizer = self.get_optimizer(optimizer, learning_rate, l2)
        self.loss = self.get_loss(loss_function)
        self.learning_rate = learning_rate
        self.l1_rate = l1
        self.l2_rate = l2
        self.metric = self.get_metric()
        self.clip_val = clip_val
        self.scheduler = self.get_scheduler(scheduler, self.optimizer)
        
        if param_initialisation is not None: 
            layer, init_method = param_initialisation 
            self.initialize_weights(layer, init_method)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 20 * 20)
        x = self.classifier(x)
        return x

    def get_optimizer(self, optimizer, learning_rate, l2):
        Optimizers = {'Adam':optim.Adam(self.parameters(), lr=learning_rate, weight_decay = l2), 'SGD': optim.SGD(self.parameters(), lr=learning_rate, momentum=0.09, weight_decay = l2)}
        return Optimizers[optimizer]

    def l1_regularization(self, loss):
        l1_reg = sum(p.abs().sum() * self.l1_rate for p in self.parameters())
        loss += l1_reg
        return loss

    def get_loss(self,loss_function):
        Loss_Functions = {'CEL': nn.CrossEntropyLoss(), 'MSE': nn.MSELoss()}
        return Loss_Functions[loss_function]

    def get_metric(self):
        return torch.nn.L1Loss()
    
    def get_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            return None
        schedulers = {'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.00001)}
        return schedulers[scheduler]

    def initialize_weights(self, layer_init = None, initialisation = 'Normal'):
        init_methods = {'Xavier': init.xavier_uniform_, 'Uniform': lambda x : init.uniform_(x, a=-0.1, b=0.1), 'Normal': lambda x: init.normal_(x, mean=0, std=0.01), 'He': lambda x: init.kaiming_normal_( x , mode='fan_in', nonlinearity='relu') }

        self._init_weights = init_methods[initialisation]

        if layer_init is None:
            print('no layer specified')
            parameters = self.named_parameters()
            print(f'{initialisation} initialization for all weights')
        else: 
            parameters = layer_init.named_parameters()
            print(f'{initialisation} initialization for {layer_init}')
        for name, param in parameters:
            if 'weight' in name:
                self._init_weights(param)
            elif 'bias' in name:
                # Initialize biases to zeros
                nn.init.constant_(param, 0)
        
        def clip_gradients(self, clip_value):
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-clip_value, clip_value)

