import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_dataloaders(data_path, seq_len, pred_len, batch_size):

    #  Wczytanie
    df = pd.read_csv(data_path)
    features_df = df.drop(['date'], axis=1)
    target_idx = features_df.columns.get_loc('OT')
    
    # 12k godzin, do zmiany potem na caly dataset

    data_raw = features_df.values[:12000]

    # Trening\test to bedzie 80\20 tutaj, można zrobić 70/15/15 z walidacyjnym ale nie jest to tu potrzebne

    split_idx = int(0.8 * len(data_raw))
    train_data = data_raw[:split_idx]
    test_data = data_raw[split_idx:]

    # skalujemy i robimy fit() na train

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data) # Tylko transform!

    # Okna przesuwne dla GRU
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len : i + seq_len + pred_len, target_idx])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    X_test, y_test = create_sequences(test_scaled)

    # Dataloadery PyTorcha
    train_loader = DataLoader(ETDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ETDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler