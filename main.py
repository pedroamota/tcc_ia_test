import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# Dataset dinâmico para gerar os dados sob demanda
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = torch.rand(self.input_size, dtype=torch.float32)  # Usar float32
        y = torch.sum(X ** 2).unsqueeze(0)
        return X, y

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 1000  # Reduzir o tamanho de entrada
    hidden_size = 2048  # Reduzir número de neurônios
    num_layers = 5  # Reduzir número de camadas
    output_size = 1
    num_samples = 1_000_000  # Reduzir o número de amostras
    batch_size = 512  # Reduzir o batch size
    num_epochs = 10

    # Criar dataset e dataloader
    dataset = SyntheticDataset(num_samples, input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    # Modelo extremamente complexo
    class ExtremeModel(nn.Module):
        def __init__(self):
            super(ExtremeModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                        nn.Dropout(0.1)
                    ) for _ in range(num_layers - 2)
                ],
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    model = ExtremeModel().to(device).to(torch.float32)  # Garantir float32 no modelo

    # Função de perda e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Treinamento
    print("Treinando modelo extremo...")
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
