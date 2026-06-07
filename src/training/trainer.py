import torch
import torch.nn as nn

def train_model(model, train_loader, device, epochs, lr, save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Rozpoczynam trenowanie na urządzeniu: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoka [{epoch+1}/{epochs}] | Strata Treningowa (MSE): {avg_loss:.4f}')
        

    # Zapis wag po skończeniu uczenia
    torch.save(model.state_dict(), save_path)
    print(f"Zapisano wagi modelu w: {save_path}")