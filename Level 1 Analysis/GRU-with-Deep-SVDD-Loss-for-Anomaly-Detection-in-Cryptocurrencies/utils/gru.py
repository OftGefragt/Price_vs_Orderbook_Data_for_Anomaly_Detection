import torch
import torch.nn as nn
import torch.optim as optim

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, n_layers=1, dropout=0.0):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=embedding_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        _, h_n = self.gru(x)
        embedding = h_n[-1]
        return embedding

def train_deep_svdd(gru_encoder, X, epochs=50, batch_size=64, lr=1e-3, lr_c=0.01, device='cpu'):
    gru_encoder.to(device)
    X = X.to(device)
    # initialize center c as zero vector
    embedding_dim = gru_encoder.embedding_dim
    c = torch.zeros(embedding_dim, device=device) # center of the hypersphere

    optimizer = optim.Adam(gru_encoder.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (batch,) in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            # forward pass
            embeddings = gru_encoder(batch)
            # compute loss
            diff = embeddings - c
            loss = torch.mean(torch.sum(diff**2, dim=1))
            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            # update center c
            batch_center_update = torch.mean(diff.detach(), dim=0)
            c -= lr_c * batch_center_update
            # accumulate loss
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    return c
