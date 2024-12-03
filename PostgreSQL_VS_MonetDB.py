import os
import pandas as pd
import psycopg2
from pymonetdb import connect
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
os.environ['KAGGLE_CONFIG_DIR'] = '/Users/arthurmorand/Desktop/BDD/Projet/Comparateur/config'
from kaggle.api.kaggle_api_extended import KaggleApi

# Spécifiez le chemin absolu du fichier kaggle.json
print(os.environ.get('KAGGLE_CONFIG_DIR'))

# Initialiser l'API Kaggle avec les informations d'identification
api = KaggleApi()
api.authenticate()


# Téléchargement avec une barre de progression
def download_dataset(dataset_name):
    with tqdm(total=1, desc="Téléchargement du dataset", unit="dataset") as pbar:
        api.dataset_download_files(dataset_name, path='./data', unzip=True)
        pbar.update(1)
    print(f"Dataset téléchargé dans : ./data")


# Connexion à PostgreSQL
def connect_postgres(db_name, user, password, host="localhost", port=5432):
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connexion à PostgreSQL réussie.")
        return conn
    except Exception as e:
        print(f"Erreur de connexion à PostgreSQL : {e}")
        return None


# Connexion à MonetDB
def connect_monetdb(db_name, user, password, host="localhost", port=50000):
    try:
        conn = connect(
            username=user,
            password=password,
            hostname=host,
            database=db_name
        )
        print("Connexion à MonetDB réussie.")
        return conn
    except Exception as e:
        print(f"Erreur de connexion à MonetDB : {e}")
        return None


# Création de tables dans les deux bases
def create_tables(postgres_conn, monet_conn):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS taxi_data (
        id SERIAL PRIMARY KEY,
        pickup_datetime TIMESTAMP,
        dropoff_datetime TIMESTAMP,
        passenger_count INT,
        trip_distance FLOAT,
        total_amount FLOAT
    );
    """

    try:
        # Création de la table dans PostgreSQL
        with postgres_conn.cursor() as cursor:
            cursor.execute(create_table_query)
            postgres_conn.commit()
        print("Table créée dans PostgreSQL.")

        # Création de la table dans MonetDB
        with monet_conn.cursor() as cursor:
            cursor.execute(create_table_query)
        print("Table créée dans MonetDB.")
    except Exception as e:
        print(f"Erreur lors de la création des tables : {e}")


# Chargement du fichier CSV avec barre de progression
def load_csv(file_path, max_rows=100000):
    try:
        with tqdm(total=max_rows, desc="Chargement du CSV", unit="lignes") as pbar:
            data = pd.read_csv(file_path).head(max_rows)
            pbar.update(max_rows)
        data_tuples = list(data.itertuples(index=False, name=None))
        print("Données chargées avec succès.")
        return data_tuples
    except Exception as e:
        print(f"Erreur lors du chargement des données CSV : {e}")
        return None


# Insertion progressive des données avec barre de progression
def insert_and_measure(conn, query, data, batch_sizes, num_tests=3):
    batch_times = {batch_size: [] for batch_size in batch_sizes}

    for batch_size in batch_sizes:
        for _ in range(num_tests):
            batch_data = data[:batch_size]  # Limiter la taille du batch
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion {batch_size} lignes", unit="lignes") as pbar:
                    with conn.cursor() as cursor:
                        for row in batch_data:
                            cursor.execute(query, row)
                            pbar.update(1)
                        conn.commit()
                elapsed_time = time.time() - start_time
                batch_times[batch_size].append(elapsed_time)
                print(f"Inséré {batch_size} lignes en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur lors de l'insertion des données : {e}")

    # Calcul des moyennes et écarts-types
    avg_times = {batch_size: np.mean(times) for batch_size, times in batch_times.items()}
    std_times = {batch_size: np.std(times) for batch_size, times in batch_times.items()}

    return avg_times, std_times


# Visualisation des résultats
def plot_results(batch_sizes, postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times):
    plt.figure(figsize=(10, 6))

    # Tracer les moyennes
    plt.plot(batch_sizes, postgres_avg_times, label="PostgreSQL (Moyenne)", marker='o')
    plt.plot(batch_sizes, monet_avg_times, label="MonetDB (Moyenne)", marker='o')

    # Tracer les barres d'erreur pour l'écart-type
    plt.errorbar(batch_sizes, postgres_avg_times, yerr=postgres_std_times, fmt='o', color='blue', alpha=0.5, capsize=5)
    plt.errorbar(batch_sizes, monet_avg_times, yerr=monet_std_times, fmt='o', color='orange', alpha=0.5, capsize=5)

    plt.xlabel("Nombre de lignes insérées")
    plt.ylabel("Temps d'insertion (s)")
    plt.title("Performance d'insertion : PostgreSQL vs MonetDB")
    plt.legend()
    plt.grid()
    plt.show()


# Programme principal avec barre de progression globale
def main():
    with tqdm(total=6, desc="Progression globale", unit="étapes") as pbar:
        dataset_name = "elemento/nyc-yellow-taxi-trip-data"

        # 1. Télécharger les données
        download_dataset(dataset_name)
        pbar.update(1)

        # 2. Détecter le fichier CSV
        file_path = None
        for root, dirs, files in os.walk("./data"):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    break

        if not file_path:
            print("Fichier CSV introuvable dans le dossier de données.")
            return
        print(f"Fichier CSV trouvé : {file_path}")
        pbar.update(1)

        # 3. Charger le fichier CSV
        data_tuples = load_csv(file_path)
        if data_tuples is None:
            return
        pbar.update(1)

        # 4. Connexions aux bases de données
        postgres_conn = connect_postgres("postgres_db", "postgres_user", "password")
        monet_conn = connect_monetdb("monetdb", "monetdb", "monetdb")
        if not postgres_conn or not monet_conn:
            print("Erreur de connexion à l'une des bases de données.")
            return
        pbar.update(1)

        # 5. Création des tables
        create_tables(postgres_conn, monet_conn)
        pbar.update(1)

        # 6. Tests d'insertion
        insert_query = """
        INSERT INTO taxi_data (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
        VALUES (%s, %s, %s, %s, %s);
        """
        batch_sizes = [1000, 5000, 10000, 20000]
        postgres_avg_times, postgres_std_times = insert_and_measure(postgres_conn, insert_query, data_tuples,
                                                                    batch_sizes, num_tests=5)
        monet_avg_times, monet_std_times = insert_and_measure(monet_conn, insert_query, data_tuples, batch_sizes,
                                                              num_tests=5)
        pbar.update(1)

        # Visualisation des résultats
        plot_results(batch_sizes, postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times)

        # Fermeture des connexions
        postgres_conn.close()
        monet_conn.close()


if __name__ == "__main__":
    main()