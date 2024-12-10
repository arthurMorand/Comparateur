import os
import pandas as pd
import psycopg2
from pymonetdb import connect
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import configparser

chemin_courant = os.getcwd()
chemin_config = chemin_courant + '/config'
os.environ['KAGGLE_CONFIG_DIR'] = chemin_config
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
def connect_postgres(db_name, user, password, host, port):
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
def connect_monetdb(db_name, user, password, host, port):
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

# Supprime une table dans PostgreSQL si elle existe.
def drop_table_postgres(postgres_conn, table_name):
    try:
        with postgres_conn.cursor() as cursor:
            # Vérifier si la table existe
            check_table_query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = '{table_name}'
            );
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0]

            if table_exists:
                # Supprimer la table
                drop_table_query = f"DROP TABLE {table_name} CASCADE;"
                cursor.execute(drop_table_query)
                print(f"Table '{table_name}' supprimée avec succès dans PostgreSQL.")

                # Supprimer la séquence associée si elle existe
                drop_sequence_query = f"DROP SEQUENCE IF EXISTS {table_name}_id_seq;"
                cursor.execute(drop_sequence_query)
                print(f"Séquence '{table_name}_id_seq' supprimée avec succès dans PostgreSQL.")

                postgres_conn.commit()
            else:
                print(f"La table '{table_name}' n'existe pas dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors de la suppression de la table '{table_name}' dans PostgreSQL : {e}")
        postgres_conn.rollback()

# Crée une nouvelle table dans PostgreSQL.
def create_table_postgres(postgres_conn, table_name):
    try:
        # Supprimer la table si elle existe déjà
        drop_table_postgres(postgres_conn, table_name)

        # Créer la table
        create_table_query = f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            pickup_datetime TIMESTAMP,
            dropoff_datetime TIMESTAMP,
            passenger_count INT,
            trip_distance FLOAT,
            total_amount FLOAT
        );
        """
        with postgres_conn.cursor() as cursor:
            cursor.execute(create_table_query)
            postgres_conn.commit()
            print(f"Table '{table_name}' créée avec succès dans PostgreSQL.")

    except Exception as e:
        print(f"Erreur lors de la création de la table {table_name} dans PostgreSQL : {e}")
        postgres_conn.rollback()

# Supprime une table dans MonetDB si elle existe.
def drop_table_monetdb(monet_conn, table_name):
    try:
        with monet_conn.cursor() as cursor:
            # Vérifier si la table existe
            check_table_query = f"""
            SELECT name
            FROM sys.tables
            WHERE name = '{table_name}';
            """
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()

            if table_exists:
                # Supprimer la table
                drop_table_query = f"DROP TABLE {table_name};"
                cursor.execute(drop_table_query)
                monet_conn.commit()
                print(f"Table '{table_name}' supprimée avec succès dans MonetDB.")
            else:
                print(f"La table '{table_name}' n'existe pas dans MonetDB.")

    except Exception as e:
        print(f"Erreur lors de la suppression de la table '{table_name}' dans MonetDB : {e}")
        monet_conn.rollback()

# Crée une nouvelle table dans MonetDB.
def create_table_monetdb(monet_conn, table_name):
    """
    Supprime la table si elle existe déjà, puis crée une nouvelle table dans MonetDB.
    """
    try:
        # Supprimer la table si elle existe déjà
        drop_table_monetdb(monet_conn, table_name)

        # Créer la table avec une séquence pour l'id
        create_table_query = f"""
        CREATE TABLE {table_name} (
            id INT PRIMARY KEY DEFAULT NEXT VALUE FOR {table_name}_id_seq,
            pickup_datetime TIMESTAMP,
            dropoff_datetime TIMESTAMP,
            passenger_count INT,
            trip_distance FLOAT,
            total_amount FLOAT
        );
        """

        # Créer une séquence pour l'id (sans IF NOT EXISTS)
        create_sequence_query = f"""
        CREATE SEQUENCE {table_name}_id_seq;
        """

        with monet_conn.cursor() as cursor:
            # Vérifier si la séquence existe déjà
            check_sequence_query = f"""
            SELECT name
            FROM sys.sequences
            WHERE name = '{table_name}_id_seq';
            """
            cursor.execute(check_sequence_query)
            sequence_exists = cursor.fetchone()

            if not sequence_exists:
                cursor.execute(create_sequence_query)
                print(f"Séquence '{table_name}_id_seq' créée dans MonetDB.")
            else:
                print(f"Séquence '{table_name}_id_seq' existe déjà dans MonetDB.")

            # Créer la table
            cursor.execute(create_table_query)
            monet_conn.commit()
            print(f"Table '{table_name}' créée avec succès dans MonetDB.")

    except Exception as e:
        print(f"Erreur lors de la création ou de la suppression de la table {table_name} dans MonetDB : {e}")
        monet_conn.rollback()

def create_table(postgres_conn, monet_conn, table_name):
    print()
    print("Création des tables dans PostgreSQL et MonetDB")
    create_table_postgres(postgres_conn, table_name)
    create_table_monetdb(monet_conn, table_name)

# Insertion des données dans postgres
def insert_data_postgres(postgres_conn, data, table_name, max_rows=None):
    insert_query_postgres = f"""
    INSERT INTO {table_name} (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
    VALUES (%s, %s, %s, %s, %s)
    """

    if max_rows is not None:
        data = data[:max_rows]

    data_postgres = [
        (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
        for (_, pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount, *rest) in data
    ]

    try:
        with postgres_conn.cursor() as cursor:
            cursor.executemany(insert_query_postgres, data_postgres)
            postgres_conn.commit()
    except Exception as e:
        print(f"Erreur lors de l'insertion des données dans PostgreSQL : {e}")

# Insertion des données dans MonetDB
def insert_data_monet(monet_conn, data, table_name, max_rows=None):
    insert_query_monet = f"""
    INSERT INTO {table_name} (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
    VALUES (%s, %s, %s, %s, %s)
    """

    if max_rows is not None:
        data = data[:max_rows]

    # Inclure l'id dans les données insérées
    data_monet = [
        (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
        for (_, pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount, *rest) in data
    ]

    try:
        with monet_conn.cursor() as cursor:
            cursor.executemany(insert_query_monet, data_monet)
            monet_conn.commit()
    except Exception as e:
        print(f"Erreur lors de l'insertion des données dans MonetDB : {e}")
        monet_conn.rollback()  # ROLLBACK en cas d'erreur


def experiment_storage_engine(postgres_conn, monet_conn, data, batch_sizes, table_name, num_tests=3):
    # Créer les tables dans PostgreSQL et MonetDB
    print(f"Inssertion des {batch_sizes} données dans PostgreSQL et MonetDB pour {table_name}")
    create_table(postgres_conn, monet_conn, table_name)

    # Dictionnaires pour stocker les temps d'insertion pour chaque base de données
    postgres_batch_times = {batch_size: [] for batch_size in batch_sizes}
    monet_batch_times = {batch_size: [] for batch_size in batch_sizes}

    # Dictionnaires pour stocker les temps de suppression pour chaque base de données
    postgres_drop_times = {batch_size: [] for batch_size in batch_sizes}
    monet_drop_times = {batch_size: [] for batch_size in batch_sizes}

    for batch_size in batch_sizes:
        for _ in range(num_tests):
            # Limiter les données à insérer à la taille du batch
            batch_data = data[:batch_size]

            # Insertion dans PostgreSQL
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion PostgreSQL {batch_size} lignes",
                          unit="lignes") as pbar:
                    insert_data_postgres(postgres_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                postgres_batch_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur PostgreSQL : {e}")

            # Insertion dans MonetDB
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion MonetDB {batch_size} lignes", unit="lignes") as pbar:
                    insert_data_monet(monet_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                monet_batch_times[batch_size].append(elapsed_time)
                print(f"MonetDB : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur MonetDB : {e}")

            # Suppression des tables après l'insertion
            start_time = time.time()
            try:
                # Supprimer la table dans PostgreSQL
                drop_table_postgres(postgres_conn, table_name)
                elapsed_time = time.time() - start_time
                postgres_drop_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : Table {table_name} supprimée en {elapsed_time:.2f} secondes.")

                # Supprimer la table dans MonetDB
                start_time = time.time()
                drop_table_monetdb(monet_conn, table_name)
                elapsed_time = time.time() - start_time
                monet_drop_times[batch_size].append(elapsed_time)
                print(f"MonetDB : Table {table_name} supprimée en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur lors de la suppression des tables : {e}")

    # Calcul des moyennes et écarts-types pour PostgreSQL et MonetDB
    postgres_avg_times = {}
    postgres_std_times = {}
    monet_avg_times = {}
    monet_std_times = {}

    postgres_avg_drop_times = {}
    postgres_std_drop_times = {}
    monet_avg_drop_times = {}
    monet_std_drop_times = {}

    for batch_size in batch_sizes:
        if len(postgres_batch_times[batch_size]) > 0:
            postgres_avg_times[batch_size] = np.mean(postgres_batch_times[batch_size])
            postgres_std_times[batch_size] = np.std(postgres_batch_times[batch_size])
        else:
            postgres_avg_times[batch_size] = np.nan
            postgres_std_times[batch_size] = np.nan

        if len(monet_batch_times[batch_size]) > 0:
            monet_avg_times[batch_size] = np.mean(monet_batch_times[batch_size])
            monet_std_times[batch_size] = np.std(monet_batch_times[batch_size])
        else:
            monet_avg_times[batch_size] = np.nan
            monet_std_times[batch_size] = np.nan

        if len(postgres_drop_times[batch_size]) > 0:
            postgres_avg_drop_times[batch_size] = np.mean(postgres_drop_times[batch_size])
            postgres_std_drop_times[batch_size] = np.std(postgres_drop_times[batch_size])
        else:
            postgres_avg_drop_times[batch_size] = np.nan
            postgres_std_drop_times[batch_size] = np.nan

        if len(monet_drop_times[batch_size]) > 0:
            monet_avg_drop_times[batch_size] = np.mean(monet_drop_times[batch_size])
            monet_std_drop_times[batch_size] = np.std(monet_drop_times[batch_size])
        else:
            monet_avg_drop_times[batch_size] = np.nan
            monet_std_drop_times[batch_size] = np.nan

    print(f"Temps moyens PostgreSQL (insertion) : {postgres_avg_times}")
    print(f"Temps écarts-types PostgreSQL (insertion) : {postgres_std_times}")
    print(f"Temps moyens MonetDB (insertion) : {monet_avg_times}")
    print(f"Temps écarts-types MonetDB (insertion) : {monet_std_times}")

    print(f"Temps moyens PostgreSQL (suppression) : {postgres_avg_drop_times}")
    print(f"Temps écarts-types PostgreSQL (suppression) : {postgres_std_drop_times}")
    print(f"Temps moyens MonetDB (suppression) : {monet_avg_drop_times}")
    print(f"Temps écarts-types MonetDB (suppression) : {monet_std_drop_times}")

    return postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times, postgres_avg_drop_times, postgres_std_drop_times, monet_avg_drop_times, monet_std_drop_times


def experiment_on_increasing_rows(postgres_conn, monet_conn, data, num_tests=3):
    # Calculer les tailles de lots en fonction du nombre de données
    num_data = len(data)
    step_size = num_data // 10  # Taille de lot estimée pour 10 expériences

    # Générer des tailles de lots croissantes
    batch_sizes = [step_size * i for i in range(1, 11)]  # De step_size à 10*step_size

    # Dictionnaires pour stocker les résultats
    postgres_avg_times = {}
    postgres_std_times = {}
    monet_avg_times = {}
    monet_std_times = {}

    postgres_avg_drop_times = {}
    postgres_std_drop_times = {}
    monet_avg_drop_times = {}
    monet_std_drop_times = {}

    # Pour chaque taille de lot, on effectue l'expérience
    for batch_size in batch_sizes:
        print(f"Expérience pour {batch_size} lignes:")

        # Générer un nom unique de table pour chaque taille de lot (par exemple: taxi_trips_100, taxi_trips_200, etc.)
        table_name = f"taxi_trips_{batch_size}"

        # Appel de l'expérience pour une taille de lot donnée
        avg_postgres_time, std_postgres_time, avg_monet_time, std_monet_time, \
        avg_postgres_drop_time, std_postgres_drop_time, avg_monet_drop_time, std_monet_drop_time = experiment_storage_engine(
            postgres_conn, monet_conn, data, [batch_size], table_name, num_tests=num_tests)

        # Stocker les résultats
        postgres_avg_times[batch_size] = avg_postgres_time[batch_size]
        postgres_std_times[batch_size] = std_postgres_time[batch_size]
        monet_avg_times[batch_size] = avg_monet_time[batch_size]
        monet_std_times[batch_size] = std_monet_time[batch_size]

        postgres_avg_drop_times[batch_size] = avg_postgres_drop_time[batch_size]
        postgres_std_drop_times[batch_size] = std_postgres_drop_time[batch_size]
        monet_avg_drop_times[batch_size] = avg_monet_drop_time[batch_size]
        monet_std_drop_times[batch_size] = std_monet_drop_time[batch_size]

    # Affichage des résultats
    print(f"Temps moyens PostgreSQL (insertion) : {postgres_avg_times}")
    print(f"Écarts-types PostgreSQL (insertion) : {postgres_std_times}")
    print(f"Temps moyens MonetDB (insertion) : {monet_avg_times}")
    print(f"Écarts-types MonetDB (insertion) : {monet_std_times}")

    print(f"Temps moyens PostgreSQL (suppression) : {postgres_avg_drop_times}")
    print(f"Écarts-types PostgreSQL (suppression) : {postgres_std_drop_times}")
    print(f"Temps moyens MonetDB (suppression) : {monet_avg_drop_times}")
    print(f"Écarts-types MonetDB (suppression) : {monet_std_drop_times}")

    return postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times, postgres_avg_drop_times, postgres_std_drop_times, monet_avg_drop_times, monet_std_drop_times

def plot_insertion_results(batch_sizes, postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times):
    plt.figure(figsize=(10, 6))

    # Tracer les temps d'insertion
    plt.plot(batch_sizes, list(postgres_avg_times.values()), label="PostgreSQL (Insertion Moyenne)", marker='o')
    plt.plot(batch_sizes, list(monet_avg_times.values()), label="MonetDB (Insertion Moyenne)", marker='o')

    plt.errorbar(batch_sizes, list(postgres_avg_times.values()), yerr=list(postgres_std_times.values()), fmt='o', color='blue', alpha=0.5, capsize=5)
    plt.errorbar(batch_sizes, list(monet_avg_times.values()), yerr=list(monet_std_times.values()), fmt='o', color='orange', alpha=0.5, capsize=5)

    plt.xlabel("Nombre de lignes insérées")
    plt.ylabel("Temps (s)")
    plt.title("Temps d'insertion : PostgreSQL vs MonetDB")
    plt.legend()
    plt.grid()
    plt.show()

def plot_deletion_results(batch_sizes, postgres_avg_drop_times, postgres_std_drop_times, monet_avg_drop_times, monet_std_drop_times):
    plt.figure(figsize=(10, 6))

    # Tracer les temps de suppression
    plt.plot(batch_sizes, list(postgres_avg_drop_times.values()), label="PostgreSQL (Suppression Moyenne)", marker='s')
    plt.plot(batch_sizes, list(monet_avg_drop_times.values()), label="MonetDB (Suppression Moyenne)", marker='s')

    plt.errorbar(batch_sizes, list(postgres_avg_drop_times.values()), yerr=list(postgres_std_drop_times.values()), fmt='s', color='green', alpha=0.5, capsize=5)
    plt.errorbar(batch_sizes, list(monet_avg_drop_times.values()), yerr=list(monet_std_drop_times.values()), fmt='s', color='red', alpha=0.5, capsize=5)

    plt.xlabel("Nombre de lignes supprimées")
    plt.ylabel("Temps (s)")
    plt.title("Temps de suppression : PostgreSQL vs MonetDB")
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
        # Charger les informations de configuration depuis bdd.ini
        config = configparser.ConfigParser()
        config.read(chemin_config + "/bdd.ini")

        # Récupérer les informations pour PostgreSQL
        postgres_db = config["postgres"]["database"]
        postgres_user = config["postgres"]["user"]
        postgres_password = config["postgres"]["password"]
        postgres_host = config["postgres"]["host"]
        postgres_port = config["postgres"]["port"]

        # Récupérer les informations pour MonetDB
        monet_db = config["monetdb"]["database"]
        monet_user = config["monetdb"]["user"]
        monet_password = config["monetdb"]["password"]
        monet_host = config["monetdb"]["host"]
        monet_port = config["monetdb"]["port"]

        postgres_conn = connect_postgres(postgres_db, postgres_user, postgres_password, postgres_host, postgres_port)
        monet_conn = connect_monetdb(monet_db, monet_user, monet_password, monet_host, monet_port)

        if not postgres_conn or not monet_conn:
            print("Erreur de connexion à l'une des bases de données.")
            return
        pbar.update(1)

        # # 5. Création des tables
        # create_tables(postgres_conn, monet_conn, "taxi_trips")
        # pbar.update(1)
        #
        # # 6. Insertion des données dans PostgreSQL et MonetDB
        # insert_data(postgres_conn, monet_conn, data_tuples, "taxi_trips")  # Appel à insert_data pour insérer les données
        # pbar.update(1)

        # 7. Tests de performance pour les insertions
        postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times, postgres_avg_drop_times, postgres_std_drop_times, monet_avg_drop_times, monet_std_drop_times = experiment_on_increasing_rows(postgres_conn, monet_conn, data_tuples, num_tests=3)

        # 8. Visualisation des résultats
        plot_insertion_results(postgres_avg_times.keys(), postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times)
        plot_deletion_results(postgres_avg_drop_times.keys(),postgres_avg_drop_times, postgres_std_drop_times, monet_avg_drop_times,monet_std_drop_times)

        # 9. Fermeture des connexions
        postgres_conn.close()
        monet_conn.close()


if __name__ == "__main__":
    main()