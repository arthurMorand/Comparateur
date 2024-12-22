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
chemin_config = chemin_courant+'/config'
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

def create_tables(postgres_conn, monet_conn, table_name):
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

def insert_data(postgres_conn, monet_conn, data, table_name):
    print()
    print("Insertion des données dans les tables dans PostgreSQL et MonetDB")
    insert_data_monet(postgres_conn, data, table_name)
    insert_data_postgres(monet_conn, data, table_name)

def plot_results(batch_sizes, postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times):
    plt.figure(figsize=(10, 6))

    # Vérifiez que les longueurs correspondent
    print(f"batch_sizes: {batch_sizes}")
    print(f"postgres_avg_times: {postgres_avg_times}")
    print(f"monet_avg_times: {monet_avg_times}")

    # Tracer les moyennes
    plt.plot(batch_sizes, list(postgres_avg_times.values()), label="PostgreSQL (Moyenne)", marker='o')
    plt.plot(batch_sizes, list(monet_avg_times.values()), label="MonetDB (Moyenne)", marker='o')

    # Tracer les barres d'erreur pour l'écart-type
    plt.errorbar(batch_sizes, list(postgres_avg_times.values()), yerr=list(postgres_std_times.values()), fmt='o', color='blue', alpha=0.5, capsize=5)
    plt.errorbar(batch_sizes, list(monet_avg_times.values()), yerr=list(monet_std_times.values()), fmt='o', color='orange', alpha=0.5, capsize=5)

    plt.xlabel("Nombre de lignes insérées")
    plt.ylabel("Temps d'insertion (s)")
    plt.title("Performance d'insertion : PostgreSQL vs MonetDB")
    plt.legend()
    plt.grid()
    plt.show()

# Fonction pour exécuter une requête et mesurer le temps d'exécution
def execute_query(conn, query, db_name):
    try:
        start_time = time.time()
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        elapsed_time = time.time() - start_time
        print(f"{db_name} : Requête exécutée en {elapsed_time:.4f} secondes.")
        return elapsed_time, results
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête dans {db_name} : {e}")
        return None, []

# Fonction pour tester plusieurs requêtes
def test_queries(postgres_conn, monet_conn):
    queries = {
        "Simple Select": "SELECT * FROM taxi_trips LIMIT 100;",
        "Aggregation": "SELECT passenger_count, AVG(total_amount) FROM taxi_trips GROUP BY passenger_count;",
        "Filter": "SELECT * FROM taxi_trips WHERE trip_distance > 5.0;",
        "Join (Self-Join)": """
            SELECT t1.id, t2.id
            FROM taxi_trips t1 JOIN taxi_trips t2
            ON t1.passenger_count = t2.passenger_count
            WHERE t1.id < t2.id
            LIMIT 100;
        """
    }

    postgres_times = {}
    monet_times = {}

    for query_name, query in queries.items():
        print(f"Test de la requête : {query_name}")

        # Test sur PostgreSQL
        postgres_time, _ = execute_query(postgres_conn, query, "PostgreSQL")
        postgres_times[query_name] = postgres_time

        # Test sur MonetDB
        monet_time, _ = execute_query(monet_conn, query, "MonetDB")
        monet_times[query_name] = monet_time

    return postgres_times, monet_times

# Visualisation des résultats de performance des requêtes
def plot_query_results(postgres_times, monet_times):
    query_names = list(postgres_times.keys())
    postgres_values = list(postgres_times.values())
    monet_values = list(monet_times.values())

    x = np.arange(len(query_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, postgres_values, 0.4, label='PostgreSQL', color='blue')
    plt.bar(x + 0.2, monet_values, 0.4, label='MonetDB', color='orange')

    plt.xlabel('Requêtes')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Comparaison des performances des requêtes : PostgreSQL vs MonetDB')
    plt.xticks(x, query_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Intégration dans le programme principal
def main():
    with tqdm(total=8, desc="Progression globale", unit="étapes") as pbar:
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
        config = configparser.ConfigParser()
        config.read(chemin_config + "/bdd.ini")

        # PostgreSQL config
        postgres_db = config["postgres"]["database"]
        postgres_user = config["postgres"]["user"]
        postgres_password = config["postgres"]["password"]
        postgres_host = config["postgres"]["host"]
        postgres_port = config["postgres"]["port"]

        # MonetDB config
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
        
        # 5. Création des tables
        create_tables(postgres_conn, monet_conn, "taxi_trips")
        pbar.update(1)
        
        # 6. Insertion des données dans PostgreSQL et MonetDB
        insert_data(postgres_conn, monet_conn, data_tuples, "taxi_trips")  # Appel à insert_data pour insérer les données
        pbar.update(1)

        # 7. Tester les requêtes
        postgres_times, monet_times = test_queries(postgres_conn, monet_conn)
        pbar.update(1)

        # 8. Visualisation des résultats
        plot_query_results(postgres_times, monet_times)
        pbar.update(1)

        # Fermeture des connexions
        postgres_conn.close()
        monet_conn.close()

if __name__ == "__main__":
    main()
