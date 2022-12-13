import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

class CNN(torch.nn.Module):

    def __init__(self, nInputChannels, nOutputClasses, learningRate=1E-2, nEpochs=100):
        super().__init__()

        self.nInputChannels = nInputChannels
        self.nOutputClasses = nOutputClasses
        self.trainingDevice = 'cpu'
        
        self.nEpochs = nEpochs
        self.learningRate = learningRate

        # Image input is nSamples x 3 x 28 x 28
        # Convolutional architecture
        self.conv = torch.nn.Sequential(
            
            # Layer 1
            torch.nn.Conv2d(in_channels=nInputChannels, out_channels=32, kernel_size=3, padding="valid"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            # Layer 2
            torch.nn.Conv2d(32, 32, kernel_size=3, padding="valid"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            torch.nn.Conv2d(32, 64, kernel_size=3, padding="valid"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            # Layer 4
            torch.nn.Conv2d(64, 64, kernel_size=3, padding="valid"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        
            # Layer 5
            torch.nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            # Layer 6
            torch.nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # Fully connected architecture
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(32, self.nOutputClasses)
        )
        
    def forward(self, x):

        # Calculating convolutional features
        x = self.conv(x)

        # Flattening to feed to fully connected layers
        x = x.view(x.size(0), -1)

        # Making predictions
        x = self.fc(x)

        return x
    

    def trainModel(self, trainLoader, validationLoader, classWeights=None):

        nTrainingSamples = len(trainLoader.dataset)
        
        # Moving to training device
        device = torch.device(self.trainingDevice)
        model = self
        model.to(device=device)

        # Optimization
        loss = torch.nn.CrossEntropyLoss(weight=classWeights)                         # Loss; default weight is None
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learningRate)        # Optimization approach
        #lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1) # Learning rate decay

        # Training
        writer = SummaryWriter() # Initialize logger
        for epoch in range(self.nEpochs):
            print(f'Epoch: {epoch}, LR: {optimizer.param_groups[0]["lr"]}')
            
            # -> Training mode
            model.train()
            epochLoss = 0
            epochAccuracy = 0
            for batch, (inputs, targets) in enumerate(trainLoader):

                # Making targets a vector with labels instead of a matrix
                targets = targets.to(torch.long).view(targets.size(0))
                
                # Zero gradients
                optimizer.zero_grad()

                # Forward, backpropogate, and update parameters
                outputs = model(inputs)
                batchLoss = loss(outputs, targets)
                batchLoss.backward()
                optimizer.step()
                
                # Evaluation
                y_pred = torch.max(outputs, 1)[1].data.squeeze()
                correct = (targets == y_pred).float()
                batchAccuracy = correct.sum() / correct.numel()
                
                # Update average epoch loss and accuracy
                epochLoss += batchLoss.data * inputs.shape[0] / nTrainingSamples
                epochAccuracy += batchAccuracy * 100 * inputs.shape[0] / nTrainingSamples
            
            # Step lr_decay
            #lr_decay.step()

            # -> Validation mode
            self.train(False)
            y_true = torch.tensor([])
            y_pred = torch.tensor([])
            for inputs, targets in validationLoader:
                targets = targets.to(torch.long).view(targets.size(0))
                outputs = model(inputs)
                y_true = torch.cat((y_true, targets), 0)
                y_pred = torch.cat((y_pred, torch.max(outputs, 1)[1].data.squeeze()), 0)
            # Evaluate validation
            correct = (y_true == y_pred).float()
            validationAccuracy = correct.sum() / correct.numel()
            
            # Log information
            writer.add_scalar('Epoch Loss', epochLoss, epoch)
            writer.add_scalar('Epoch Accuracy', epochAccuracy, epoch)
            writer.add_scalar('Validation Accuracy', validationAccuracy, epoch)
            
        # Stop logger    
        writer.close()
        
        
    def save(self, path):
        torch.save(self, path)
        
        
    def predict(self, dataLoader):
        device = torch.device(self.trainingDevice)
        model = self
        
        # -> Evaluation mode
        model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        y_pred = torch.tensor([])
        with torch.no_grad():
            for inputs, targets in dataLoader:
                targets = targets.to(torch.long).view(targets.size(0))
                outputs = model(inputs).softmax(dim=-1)
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
                y_pred = torch.cat((y_pred, torch.max(outputs, 1)[1].data.squeeze()), 0)
                
        #y_true = y_true.numpy()
        #y_score = y_score.detach().numpy()
        return y_true, y_score, y_pred
