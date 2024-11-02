import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Pré-processamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertendo para tensores
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Definindo o modelo da árvore de decisão
class SimpleDecisionTree(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleDecisionTree, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # Camada linear para simular a decisão

    def forward(self, x):
        return self.fc(x)

# Configurando o modelo, a função de perda e o otimizador
model = SimpleDecisionTree(input_size=4, output_size=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento do modelo
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Avaliação do modelo
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    _, predicted_labels = torch.max(predictions, 1)
    accuracy = accuracy_score(y_test.cpu(), predicted_labels.cpu())
    print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
