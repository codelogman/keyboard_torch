import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# mapeo del teclado
key_to_idx = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
    't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25,
    '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35,
    'space': 36, 'enter': 37, 'shift': 38, 'ctrl': 39, 'alt': 40, 'tab': 41, 'caps_lock': 42,
    'backspace': 43, 'esc': 44, 'left_arrow': 45, 'right_arrow': 46, 'up_arrow': 47, 'down_arrow': 48,
    'f1': 49, 'f2': 50, 'f3': 51, 'f4': 52, 'f5': 53, 'f6': 54, 'f7': 55, 'f8': 56, 'f9': 57,
    'f10': 58, 'f11': 59, 'f12': 60, 'insert': 61, 'delete': 62, 'home': 63, 'end': 64,
    'page_up': 65, 'page_down': 66, 'num_lock': 67, 'print_screen': 68, 'scroll_lock': 69,
    'pause': 70, 'semicolon': 71, 'equal': 72, 'comma': 73, 'minus': 74, 'period': 75, 'slash': 76,
    'grave': 77, 'left_bracket': 78, 'backslash': 79, 'right_bracket': 80, 'quote': 81
    # Añadir más teclas si es necesario
}

# Definir el dataset
class KeypressDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        
        # Convertir la secuencia de teclas en una secuencia de índices
        sequence_idx = [key_to_idx[key] for key in sequence]
        return torch.tensor(sequence_idx, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Ejemplo de datos: secuencias de teclas y sus etiquetas correspondientes
sequences = [
    ['a', 's', 'd', 'f'],  # Secuencia de teclas
    ['q', 'w', 'e', 'r'],
    # Añadir más secuencias aquí...
]
labels = [0, 1]  # Etiquetas correspondientes

# Crear el dataset y el dataloader
dataset = KeypressDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Definir el modelo
class KeypressModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KeypressModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x.unsqueeze(-1))  # Agregar la dimensión de características
        out = self.fc(hn[-1])
        return out

# Parámetros del modelo
input_size = 1  # Longitud de la secuencia de entrada es 1 (una característica por tecla)
hidden_size = 128
output_size = 2  # clases

# Crear el modelo, la función de pérdida y el optimizador
model = KeypressModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenando el modelo
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print("Entrenamiento completado")

