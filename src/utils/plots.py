import torch
import matplotlib.pyplot as plt

def evaluate_and_plot(model, test_loader, device):
    """Szybka funkcja do narysowania wyników na zbiorze testowym"""
    model.eval()
    with torch.no_grad():
        # jedna paczka (batch) ze zbioru testowego
        sample_X, sample_y = next(iter(test_loader))
        sample_X = sample_X.to(device)
        
        pred_y = model(sample_X).cpu().numpy()
        true_y = sample_y.numpy()

    # pierwszy przykład z paczki
    idx = 0 
    plt.figure(figsize=(10, 5))
    plt.plot(true_y[idx], label='Prawdziwa Temp (OT)', marker='o', color='blue')
    plt.plot(pred_y[idx], label='Predykcja GRU', marker='x', color='red', linestyle='--')
    plt.title("Predykcja na 24 godziny w przód (GRU)")
    plt.legend()
    plt.grid(True)
    
    # Zapis do folderu logs
    plt.savefig('logs/predykcja_gru.png')
    print("Zapisano wykres predykcji w logs/predykcja_gru.png")