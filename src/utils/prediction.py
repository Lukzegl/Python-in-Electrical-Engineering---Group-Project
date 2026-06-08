import numpy as np
import torch

FEATURE_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def format_input_row(raw_text):
    values = [value.strip() for value in raw_text.split(',') if value.strip()]
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Wymagane {len(FEATURE_NAMES)} wartości oddzielonych przecinkami: "
            + ", ".join(FEATURE_NAMES)
        )
    return np.array([float(value) for value in values], dtype=np.float32)


def build_input_sequence(raw_input, seq_len):
    if raw_input.ndim == 1:
        if raw_input.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"Pojedynczy wiersz musi mieć {len(FEATURE_NAMES)} wartości."
            )
        return np.tile(raw_input, (seq_len, 1))
    if raw_input.ndim == 2:
        if raw_input.shape != (seq_len, len(FEATURE_NAMES)):
            raise ValueError(
                f"Jeśli podajesz sekwencję, musi mieć {seq_len} wierszy i {len(FEATURE_NAMES)} kolumn."
            )
        return raw_input
    raise ValueError("Dane wejściowe muszą mieć wymiar 1D lub 2D.")


def predict_from_input(model, scaler, input_sequence, seq_len, device):
    model.eval()
    num_features = len(FEATURE_NAMES)
    if input_sequence.ndim == 2 and input_sequence.shape[1] != num_features:
        raise ValueError(f"Dane muszą mieć {num_features} cech w każdej kolumnie.")

    input_sequence = build_input_sequence(input_sequence, seq_len)
    scaled_sequence = scaler.transform(input_sequence)
    input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy().flatten()

    return predictions


def prompt_custom_input(seq_len):
    prompt = (
        "Wprowadź wartości dla jednej próbki jako 7 liczb oddzielonych przecinkami\n"
        "(kolejność: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT):\n"
    )
    raw_text = input(prompt)
    row = format_input_row(raw_text)
    return build_input_sequence(row, seq_len)


def run_custom_prediction(model, scaler, seq_len, device):
    print("\nOpcja predykcji na podstawie własnych danych.")
    if input("Czy chcesz podać dane wejściowe do modelu? [t/N]: ").strip().lower() != 't':
        return

    try:
        user_sequence = prompt_custom_input(seq_len)
        predictions = predict_from_input(model, scaler, user_sequence, seq_len, device)
        print("Predykcja modelu (24 wartości):")
        print(np.array2string(predictions, precision=3, separator=', '))
    except Exception as err:
        print(f"Nie udało się przeprowadzić predykcji: {err}")
