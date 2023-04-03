import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Sampling(nn.Module):
    def __init__(self, batch_size, latent_dim, epsilon_std=1.0):
        super(Sampling, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

    def forward(self, z_mean, z_log_sigma): # [64,100]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size , latent_dim = z_mean.shape
        # epsilon = torch.randn(self.batch_size, self.latent_dim) * self.epsilon_std
        epsilon = (torch.randn(batch_size, latent_dim)* self.epsilon_std).to(device)
        return z_mean + z_log_sigma * epsilon


class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim, batch_size, epsilon_std=1.0):
        super(LSTMVAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, intermediate_dim, batch_first=True)
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_sigma = nn.Linear(intermediate_dim, latent_dim)
        self.sampling = Sampling(batch_size, latent_dim, epsilon_std)

    def forward(self, x): # [64,100,25]
        _, (h, _) = self.lstm(x) # [64,100,32]
        h = h.squeeze(0) #
        z_mean = self.z_mean(h)
        z_log_sigma = self.z_log_sigma(h)
        z = self.sampling(z_mean, z_log_sigma)
        return z_mean, z_log_sigma, z


class LSTMVAEDecoder(nn.Module):
    def __init__(self, latent_dim, win_size, timesteps, intermediate_dim, input_dim):
        super(LSTMVAEDecoder, self).__init__()
        self.win_size = win_size
        self.timesteps = timesteps
        self.intermediate_dim = intermediate_dim
        self.repeat_vector = nn.Linear(latent_dim, timesteps * intermediate_dim)
        self.lstm_h = nn.LSTM(intermediate_dim, intermediate_dim, batch_first=True)
        self.lstm_mean = nn.LSTM(intermediate_dim, input_dim, batch_first=True)
        self.ffn = nn.Linear(timesteps, win_size)

    def forward(self, z): #[64,100]
        h_decoded = self.repeat_vector(z) #[64,1000*32]
        h_decoded = h_decoded.view(-1, self.timesteps, self.intermediate_dim)#[64,1000,32]
        h_decoded, _ = self.lstm_h(h_decoded) # [64,1000,32]
        x_decoded_mean, _ = self.lstm_mean(h_decoded) # [64,1000,25]
        x_out = x_decoded_mean.permute(0, 2, 1) #[64,25,1000]
        x_out = self.ffn(x_out).permute(0,2,1) # [64,100,25]
        return x_out


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, win_size, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std=1.0):
        super(LSTMVAE, self).__init__()
        self.encoder = LSTMVAEEncoder(input_dim, intermediate_dim, latent_dim, batch_size, epsilon_std)
        self.decoder = LSTMVAEDecoder(latent_dim, win_size, timesteps, intermediate_dim, input_dim)

    def forward(self, x):
        z_mean, z_log_sigma, z = self.encoder(x) # [64,100]
        x_decoded_mean = self.decoder(z)
        return x_decoded_mean, z_mean, z_log_sigma


def vae_loss(x, x_decoded_mean, z_mean, z_log_sigma):
    xent_loss = F.mse_loss(x, x_decoded_mean)
    kl_loss = -0.5 * torch.mean(1 + z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
    loss = xent_loss + kl_loss
    return loss
    
    
def create_lstm_vae(input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std=1.0):
    vae = LSTMVAE(input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std)
    optimizer = optim.RMSprop(vae.parameters(), lr=0.001)

    # Encoder, from inputs to latent space
    encoder = LSTMVAEEncoder(input_dim, intermediate_dim, latent_dim, batch_size, epsilon_std)

    # Generator, from latent space to reconstructed inputs
    generator = LSTMVAEDecoder(latent_dim, timesteps, intermediate_dim, input_dim)

    return vae, optimizer, encoder, generator, vae_loss

if __name__ == "__main__":
    decoder = LSTMVAEDecoder(latent_dim=100, win_size=100, timesteps=1000, intermediate_dim=32, input_dim=25)
    vae = LSTMVAE(input_dim=25, win_size=100, timesteps=1000, batch_size=64, intermediate_dim=32, latent_dim=100, epsilon_std=1.0)
    x = torch.rand(64,100,25)
    x_decoded_mean, z_mean, z_log_sigma = vae(x)
    # x_decoder = decoder(x)
    # print(x_decoder.shape)
    loss = vae_loss(x, x_decoded_mean, z_mean, z_log_sigma)
    print(x_decoded_mean.shape)
    print(loss)

