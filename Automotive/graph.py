import pandas as pd
import matplotlib.pyplot as plt

# Leggi i dati dal file CSV
df = pd.read_csv('episode_metrics.csv')

# Elimina la colonna 'speed' se esiste
if 'speed' in df.columns:
    df = df.drop(columns=['speed'])

# Plotting the graphs
plt.figure(figsize=(12, 8))

# Total Reward over Episodes
plt.subplot(2, 3, 1)
plt.plot(df['episode'], df['reward'], marker='o')
plt.title('Total Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Duration per Episode
plt.subplot(2, 3, 2)
plt.bar(df['episode'], df['duration'])
plt.title('Duration per Episode')
plt.xlabel('Episode')
plt.ylabel('Duration')

# Collisions per Episode
plt.subplot(2, 3, 3)
plt.bar(df['episode'], df['collisions'], color='r')
plt.title('Collisions per Episode')
plt.xlabel('Episode')
plt.ylabel('Collisions')

# Lane Invasions per Episode
plt.subplot(2, 3, 4)
plt.bar(df['episode'], df['lane_invasions'], color='g')
plt.title('Lane Invasions per Episode')
plt.xlabel('Episode')
plt.ylabel('Lane Invasions')

# Average Speed per Episode
plt.subplot(2, 3, 5)
plt.plot(df['episode'], df['avg_speed'], marker='o', color='m')
plt.title('Average Speed per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Speed (km/h)')

# Waypoint Distance (Similarity) per Episode
plt.subplot(2, 3, 6)
plt.plot(df['episode'], df['waypoint_similarity'], marker='o', color='c')
plt.title('Waypoint Distance per Episode')
plt.xlabel('Episode')
plt.ylabel('Waypoint Distance')

plt.tight_layout()
plt.savefig('Griglia_Subplot_Metriche_Simulazione_CARLA.png')
plt.show()
