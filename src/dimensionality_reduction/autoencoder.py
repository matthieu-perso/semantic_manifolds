import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # Add an additional layer before the latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(latent_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, dataloader, criterion, optimizer, epochs=500):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def evaluate_autoencoder(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            outputs = model(inputs)
            loss = mean_squared_error(inputs.numpy(), outputs.numpy())
            total_loss += loss
    avg_loss = total_loss / len(dataloader)
    logger.info(f'Evaluation Loss: {avg_loss:.4f}')
    return avg_loss

def run_autoencoder(embeddings, hidden_dims=[256, 128], latent_dim=16, batch_size=64, epochs=500, learning_rate=0.001, test_size=0.2, random_state=42):
    """
    Train and evaluate an autoencoder on the given embeddings.

    Parameters:
    - embeddings: numpy array of high-dimensional embeddings
    - hidden_dims: list of int, number of neurons in each hidden layer
    - latent_dim: int, number of neurons in the latent layer
    - batch_size: int, size of each batch for training
    - epochs: int, number of epochs to train the model
    - learning_rate: float, learning rate for the optimizer
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for train-test split

    Returns:
    - model: trained Autoencoder model
    - train_loss: float, average training loss
    - test_loss: float, average test loss
    """
    input_dim = embeddings.shape[1]

    X_train, X_test = train_test_split(embeddings, test_size=test_size, random_state=random_state)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Autoencoder(input_dim, hidden_dims, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = train_autoencoder(model, train_loader, criterion, optimizer, epochs=epochs)
    test_loss = evaluate_autoencoder(model, test_loader)

    return model, train_loss, test_loss
