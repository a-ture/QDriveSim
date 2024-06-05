import pandas as pd

# Leggere il file CSV
file_path = 'episode_metrics_traning_1camera.csv'
data = pd.read_csv(file_path)

# Controllare se la colonna 'duration' esiste
if 'duration' in data.columns:
    # Sommare i valori della colonna 'duration'
    total_duration = data['duration'].sum()
    print(f"La somma totale della colonna 'duration' è: {total_duration}")
else:
    print("La colonna 'duration' non esiste nel file CSV.")



