import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SphericalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(SphericalAutoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
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
        encoded = self._project_to_sphere(encoded)
        decoded = self.decoder(encoded)
        return decoded

    def _project_to_sphere(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)

def train_spherical_autoencoder(model, dataloader, criterion, optimizer, epochs=500):
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

def evaluate_spherical_autoencoder(model, dataloader):
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

if __name__ == "__main__":
    # Example usage
    input_dim = 784  # Example for MNIST dataset
    hidden_dims = [128, 64]
    latent_dim = 32

    model = SphericalAutoencoder(input_dim, hidden_dims, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load your dataset here
    # For example, using random data
    data = np.random.rand(1000, input_dim).astype(np.float32)
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data)), batch_size=32, shuffle=False)

    train_spherical_autoencoder(model, train_loader, criterion, optimizer, epochs=50)
    evaluate_spherical_autoencoder(model, test_loader)
