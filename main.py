import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Importar a barra de progresso

# Dataset dinâmico para gerar os dados sob demanda
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, input_size, device):
        self.num_samples = num_samples
        self.input_size = input_size
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = torch.rand(self.input_size, device=self.device)
        y = torch.sum(X ** 2).unsqueeze(0)  # Soma dos quadrados como saída
        return X, y

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 2000
    hidden_size = 4096
    num_layers = 6
    output_size = 1
    num_samples = 5_000_000
    batch_size = 2048
    num_epochs = 10

    # Criar dataset e dataloader
    dataset = SyntheticDataset(num_samples, input_size, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modelo complexo
    class ComplexModel(nn.Module):
        def __init__(self):
            super(ComplexModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ) for _ in range(num_layers - 2)
                ],
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    model = ComplexModel().to(device)

    # Função de perda e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    # Treinamento
    print("Treinando modelo complexo...")
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())  # Exibir a perda atual no lote
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
