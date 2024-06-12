import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_episode_metrics(file_path):
    # Carica i dati
    data = pd.read_csv(file_path)

    # Configura la figura e gli assi
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # Grafico 1: Reward per episodio
    axs[0, 0].plot(data['episode'], data['reward'], marker='o')
    axs[0, 0].set_title('Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')

    # Grafico 2: Durata per episodio
    axs[0, 1].plot(data['episode'], data['duration'], marker='o')
    axs[0, 1].set_title('Duration per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Duration (s)')

    # Grafico 3: Collisioni per episodio
    axs[1, 0].bar(data['episode'], data['collisions'], color='orange')
    axs[1, 0].set_title('Collisions per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Number of Collisions')

    # Grafico 4: Invasioni di corsia per episodio
    axs[1, 1].bar(data['episode'], data['lane_invasions'], color='red')
    axs[1, 1].set_title('Lane Invasions per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Number of Lane Invasions')

    # Grafico 5: Velocità media per episodio
    axs[2, 0].plot(data['episode'], data['avg_speed'], marker='o')
    axs[2, 0].set_title('Average Speed per Episode')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Average Speed')

    # Grafico 6: Distanza totale per episodio
    axs[2, 1].plot(data['episode'], data['total_distance'], marker='o')
    axs[2, 1].set_title('Total Distance per Episode')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Total Distance')

    # Regola il layout
    plt.tight_layout()
    plt.show()


def plot_emissions_metrics(emissions_file_path):
    # Carica i dati
    emissions_data = pd.read_csv(emissions_file_path)

    # Grafico a Linee: Emissioni nel tempo
    plt.figure(figsize=(10, 6))
    plt.plot(emissions_data['timestamp'], emissions_data['emissions'], marker='o')
    plt.title('Emissions Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Emissions (CO2eq)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Grafico a Barre: Emissioni per esecuzione
    plt.figure(figsize=(10, 6))
    plt.bar(emissions_data['run_id'], emissions_data['emissions'], color='green')
    plt.title('Emissions per Run')
    plt.xlabel('Run ID')
    plt.ylabel('Emissions (CO2eq)')
    plt.xticks(rotation=90)
    plt.show()

    # Scatter Plot: Durata vs Emissioni
    plt.figure(figsize=(10, 6))
    plt.scatter(emissions_data['duration'], emissions_data['emissions'])
    plt.title('Duration vs Emissions')
    plt.xlabel('Duration (s)')
    plt.ylabel('Emissions (CO2eq)')
    plt.grid(True)
    plt.show()

    # Istogramma delle Emissioni
    plt.figure(figsize=(10, 6))
    plt.hist(emissions_data['emissions'], bins=30, edgecolor='black')
    plt.title('Distribution of Emissions')
    plt.xlabel('Emissions (CO2eq)')
    plt.ylabel('Frequency')
    plt.show()

    # Box Plot delle Emissioni
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=emissions_data['emissions'])
    plt.title('Box Plot of Emissions')
    plt.xlabel('Emissions (CO2eq)')
    plt.show()

    # Heatmap delle Correlazioni delle Metriche Energetiche
    plt.figure(figsize=(12, 8))
    correlation_matrix = emissions_data[
        ['emissions', 'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Energy Metrics')
    plt.show()


# Esempio di utilizzo
plot_emissions_metrics('emissions.csv')

# Esempio di utilizzo
plot_episode_metrics('episode_metrics1000.csv')
