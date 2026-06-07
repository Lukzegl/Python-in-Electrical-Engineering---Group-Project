import torch

# Parametry czasowe
SEQ_LEN = 384        # 16 dni historii
PRED_LEN = 24       # 1 dzień predykcji

# Hiperparametry modelu i uczenia
BATCH_SIZE = 64
EPOCHS = 20         # 10 epok na uczenie - można zmienić tylko będzie więcej mielić
LEARNING_RATE = 0.001 # tak samo tu, sprawdzimy i poprawimy
HIDDEN_DIM = 64
NUM_LAYERS = 2     # Dwie warstwy GRU dla lepszego wyłapywania wzorców

# Ścieżki
DATA_PATH = "data/raw/ETTh2.csv"
MODEL_SAVE_PATH = "saved_models/gru_forecaster.pth"

# wybor urzadzenia do obliczen - cpu lub gpu jezeli dziala na nim cuda
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')