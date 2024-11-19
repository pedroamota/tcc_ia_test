import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        y = torch.sum(X ** 2).unsqueeze(0)
        return X, y

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 2000
    hidden_size = 4096
    num_layers = 4  # Reduzido para acelerar
    output_size = 1
    num_samples = 5_000_000
    batch_size = 8192  # Aumentado para uso mais intenso da GPU
    num_epochs = 1

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
                nn.Dropout(0.1),  # Menor dropout para aumentar desempenho
                *[
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ) for _ in range(num_layers - 2)
                ],
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    model = ComplexModel().to(device)

    # Função de perda e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Usar mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Treinamento
    print("Treinando modelo complexo...")
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                optimizer.zero_grad()

                # Forward e backward com mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
