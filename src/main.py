import torch
import matplotlib.pyplot as plt

# moduły
from configs import config
from src.data.dataset import get_dataloaders
from src.models.gru_model import GRUForecaster
from src.training.trainer import train_model
from src.utils.plots import evaluate_and_plot
from src.utils.prediction import run_custom_prediction


if __name__ == "__main__":
    print("--- ETDataset Time-Series Project ---")
    
    #  Ładowanie Danych
    train_loader, test_loader, scaler = get_dataloaders(
        data_path=config.DATA_PATH, 
        seq_len=config.SEQ_LEN, 
        pred_len=config.PRED_LEN, 
        batch_size=config.BATCH_SIZE
    )
    
    # Inicjalizacja Modelu GRU
    model = GRUForecaster(
        input_dim=7, 
        hidden_dim=config.HIDDEN_DIM, 
        output_dim=config.PRED_LEN, 
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    
    # Trening
    train_model(
        model=model, 
        train_loader=train_loader, 
        device=config.DEVICE, 
        epochs=config.EPOCHS, 
        lr=config.LEARNING_RATE, 
        save_path=config.MODEL_SAVE_PATH
    )
    
    # Ocena i Wykresy
    evaluate_and_plot(model, test_loader, config.DEVICE)
    run_custom_prediction(model, scaler, config.SEQ_LEN, config.DEVICE)