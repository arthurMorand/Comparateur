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


def delete_table_postgres(postgres_conn, table_name):
    """
    Supprime les lignes d'une table PostgreSQL en fonction de la condition spécifiée.

    Args:
    - postgres_conn : connexion à la base de données PostgreSQL.
    - table_name : nom de la table dans laquelle les lignes doivent être supprimées.
    - condition : condition à appliquer pour supprimer les lignes (par exemple, 'id = 10').
    """
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
                # Requête pour supprimer les lignes selon la condition spécifiée
                delete_query = f"DELETE FROM {table_name};"

                # Exécuter la suppression
                cursor.execute(delete_query)
                postgres_conn.commit()

                print(f"Lignes supprimées avec succès dans la table '{table_name}'.")
            else:
                print(f"La table '{table_name}' n'existe pas dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors de la suppression des données dans la table '{table_name}' : {e}")
        postgres_conn.rollback()


# Crée une nouvelle table dans PostgreSQL.
def create_table_postgres_2_attibus(postgres_conn, table_name):
    try:
        # Supprimer la table si elle existe déjà
        drop_table_postgres(postgres_conn, table_name)

        # Créer la table
        create_table_query = f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            pickup_datetime TIMESTAMP
        );
        """
        with postgres_conn.cursor() as cursor:
            cursor.execute(create_table_query)
            postgres_conn.commit()
            print(f"Table '{table_name}' créée avec succès dans PostgreSQL.")

    except Exception as e:
        print(f"Erreur lors de la création de la table {table_name} dans PostgreSQL : {e}")
        postgres_conn.rollback()
        
# Crée une nouvelle table dans PostgreSQL.
def create_table_postgres_6_attibus(postgres_conn, table_name):
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

def delete_table_monetdb(monet_conn, table_name):
    """
    Supprime les lignes d'une table MonetDB en fonction de la condition spécifiée.

    Args:
    - monet_conn : connexion à la base de données MonetDB.
    - table_name : nom de la table dans laquelle les lignes doivent être supprimées.
    - condition : condition à appliquer pour supprimer les lignes (par exemple, 'id = 10').
    """
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
                # Requête pour supprimer les lignes selon la condition spécifiée
                delete_query = f"DELETE FROM {table_name};"

                # Exécuter la suppression
                cursor.execute(delete_query)
                monet_conn.commit()

                print(f"Lignes supprimées avec succès dans la table '{table_name}'.")
            else:
                print(f"La table '{table_name}' n'existe pas dans MonetDB.")
    except Exception as e:
        print(f"Erreur lors de la suppression des données dans la table '{table_name}' : {e}")
        monet_conn.rollback()
        
# Crée une nouvelle table dans MonetDB.
def create_table_monetdb_2_attibus(monet_conn, table_name):
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
            pickup_datetime TIMESTAMP
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

# Crée une nouvelle table dans MonetDB.
def create_table_monetdb_6_attibus(monet_conn, table_name):
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
        
def create_table_2_attibus(postgres_conn, monet_conn, table_name):
    print()
    print("Création des tables dans PostgreSQL et MonetDB")
    create_table_postgres_2_attibus(postgres_conn, table_name)
    create_table_monetdb_2_attibus(monet_conn, table_name)
    
def create_table_6_attibus(postgres_conn, monet_conn, table_name):
    print()
    print("Création des tables dans PostgreSQL et MonetDB")
    create_table_postgres_6_attibus(postgres_conn, table_name)
    create_table_monetdb_6_attibus(monet_conn, table_name)

# Insertion des données dans postgres
def insert_data_postgres_2_attibus(postgres_conn, data, table_name, max_rows=None):
    insert_query_postgres = f"""
    INSERT INTO {table_name} (pickup_datetime)
    VALUES (%s)
    """

    if max_rows is not None:
        data = data[:max_rows]

    data_postgres = [
        (pickup_datetime)
        for (_, pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount, *rest) in data
    ]

    try:
        with postgres_conn.cursor() as cursor:
            cursor.executemany(insert_query_postgres, data_postgres)
            postgres_conn.commit()
    except Exception as e:
        print(f"Erreur lors de l'insertion des données dans PostgreSQL : {e}")

# Insertion des données dans postgres
def insert_data_postgres_6_attibus(postgres_conn, data, table_name, max_rows=None):
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
def insert_data_monet_2_attibus(monet_conn, data, table_name, max_rows=None):
    insert_query_monet = f"""
    INSERT INTO {table_name} (pickup_datetime, dropoff_datetime, passenger_count, trip_distance, total_amount)
    VALUES (%s)
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

# Insertion des données dans MonetDB
def insert_data_monet_6_attibus(monet_conn, data, table_name, max_rows=None):
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

def experiment_storage_engine_2_attibus(postgres_conn, monet_conn, data, batch_sizes, table_name, num_tests):
    # Créer les tables dans PostgreSQL et MonetDB
    print()
    print()
    print(f"Inssertion des {batch_sizes} données dans PostgreSQL et MonetDB pour {table_name} avec 2 attributs")
    create_table_2_attibus(postgres_conn, monet_conn, table_name)
    delete_table_postgres(postgres_conn, table_name)
    delete_table_monetdb(monet_conn, table_name)

    # Dictionnaires pour stocker les temps d'insertion pour chaque base de données
    i_postgres_batch_times = {batch_size: [] for batch_size in batch_sizes}
    i_monet_batch_times = {batch_size: [] for batch_size in batch_sizes}
    d_postgres_batch_times = {batch_size: [] for batch_size in batch_sizes}
    d_monet_batch_times = {batch_size: [] for batch_size in batch_sizes}

    for batch_size in batch_sizes:
        for _ in range(num_tests):
            # Limiter les données à insérer à la taille du batch
            batch_data = data[:batch_size]

            # Insertion dans PostgreSQL
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion PostgreSQL {batch_size} lignes", unit="lignes") as pbar:
                    insert_data_postgres_2_attibus(postgres_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                i_postgres_batch_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur insertion PostgreSQL : {e}")

            # Insertion dans MonetDB
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion MonetDB {batch_size} lignes", unit="lignes") as pbar:
                    insert_data_monet_2_attibus(monet_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                i_monet_batch_times[batch_size].append(elapsed_time)
                print(f"MonetDB : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                    print(f"Erreur insertion MonetDB : {e}")

            # Suppression dans PostgreSQL
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Suppression PostgreSQL {batch_size} lignes", unit="lignes") as pbar:
                    delete_table_postgres(postgres_conn, table_name)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                d_postgres_batch_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : {batch_size} lignes supprimées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur suppression PostgreSQL : {e}")

            # Suppression dans MonetDB
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Suppression MonetDB {batch_size} lignes", unit="lignes") as pbar:
                    delete_table_monetdb(monet_conn, table_name)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                d_monet_batch_times[batch_size].append(elapsed_time)
                print(f"MonetDB : {batch_size} lignes supprimées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur suppression MonetDB : {e}")

    # Calcul des moyennes et écarts-types pour PostgreSQL et MonetDB
    i_postgres_avg_times = {}
    i_postgres_std_times = {}
    i_monet_avg_times = {}
    i_monet_std_times = {}
    d_postgres_avg_times = {}
    d_postgres_std_times = {}
    d_monet_avg_times = {}
    d_monet_std_times = {}

    for batch_size in batch_sizes:
        if len(i_postgres_batch_times[batch_size]) > 0:
            i_postgres_avg_times[batch_size] = np.mean(i_postgres_batch_times[batch_size])
            i_postgres_std_times[batch_size] = np.std(i_postgres_batch_times[batch_size])
        else:
            i_postgres_avg_times[batch_size] = np.nan
            i_postgres_std_times[batch_size] = np.nan

        if len(i_monet_batch_times[batch_size]) > 0:
            i_monet_avg_times[batch_size] = np.mean(i_monet_batch_times[batch_size])
            i_monet_std_times[batch_size] = np.std(i_monet_batch_times[batch_size])
        else:
            i_monet_avg_times[batch_size] = np.nan
            i_monet_std_times[batch_size] = np.nan

        if len(d_postgres_batch_times[batch_size]) > 0:
            d_postgres_avg_times[batch_size] = np.mean(d_postgres_batch_times[batch_size])
            d_postgres_std_times[batch_size] = np.std(d_postgres_batch_times[batch_size])
        else:
            d_postgres_avg_times[batch_size] = np.nan
            d_postgres_std_times[batch_size] = np.nan

        if len(d_monet_batch_times[batch_size]) > 0:
            d_monet_avg_times[batch_size] = np.mean(d_monet_batch_times[batch_size])
            d_monet_std_times[batch_size] = np.std(d_monet_batch_times[batch_size])
        else:
            d_monet_avg_times[batch_size] = np.nan
            d_monet_std_times[batch_size] = np.nan

    print("Insertion 2 attributs")
    print(f"Temps moyens PostgreSQL : {i_postgres_avg_times}")
    print(f"Temps écarts-types PostgreSQL : {i_postgres_std_times}")
    print(f"Temps moyens MonetDB : {i_monet_avg_times}")
    print(f"Temps écarts-types MonetDB : {i_monet_std_times}")

    print("suppression 2 attributs")
    print(f"Temps moyens PostgreSQL : {d_postgres_avg_times}")
    print(f"Temps écarts-types PostgreSQL : {d_postgres_std_times}")
    print(f"Temps moyens MonetDB : {d_monet_avg_times}")
    print(f"Temps écarts-types MonetDB : {d_monet_std_times}")

    return i_postgres_avg_times, i_postgres_std_times, i_monet_avg_times, i_monet_std_times, d_postgres_avg_times, d_postgres_std_times, d_monet_avg_times, d_monet_std_times

def experiment_storage_engine_6_attibus(postgres_conn, monet_conn, data, batch_sizes, table_name, num_tests):
    # Créer les tables dans PostgreSQL et MonetDB
    print()
    print()
    print(f"Inssertion des {batch_sizes} données dans PostgreSQL et MonetDB pour {table_name} avec 6 attributs")
    create_table_6_attibus(postgres_conn, monet_conn, table_name)
    delete_table_postgres(postgres_conn, table_name)
    delete_table_monetdb(monet_conn, table_name)

    # Dictionnaires pour stocker les temps d'insertion pour chaque base de données
    i_postgres_batch_times = {batch_size: [] for batch_size in batch_sizes}
    i_monet_batch_times = {batch_size: [] for batch_size in batch_sizes}
    d_postgres_batch_times = {batch_size: [] for batch_size in batch_sizes}
    d_monet_batch_times = {batch_size: [] for batch_size in batch_sizes}

    for batch_size in batch_sizes:
        for _ in range(num_tests):
            # Limiter les données à insérer à la taille du batch
            batch_data = data[:batch_size]

            # Insertion dans PostgreSQL
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion PostgreSQL {batch_size} lignes", unit="lignes") as pbar:
                    insert_data_postgres_6_attibus(postgres_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                i_postgres_batch_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur insertion PostgreSQL : {e}")

            # Insertion dans MonetDB
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Insertion MonetDB {batch_size} lignes", unit="lignes") as pbar:
                    insert_data_monet_6_attibus(monet_conn, batch_data, table_name, batch_size)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                i_monet_batch_times[batch_size].append(elapsed_time)
                print(f"MonetDB : {batch_size} lignes insérées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                    print(f"Erreur insertion MonetDB : {e}")

            # Suppression dans PostgreSQL
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Suppression PostgreSQL {batch_size} lignes", unit="lignes") as pbar:
                    delete_table_postgres(postgres_conn, table_name)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                d_postgres_batch_times[batch_size].append(elapsed_time)
                print(f"PostgreSQL : {batch_size} lignes supprimées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur suppression PostgreSQL : {e}")

            # Suppression dans MonetDB
            start_time = time.time()
            try:
                with tqdm(total=len(batch_data), desc=f"Suppression MonetDB {batch_size} lignes", unit="lignes") as pbar:
                    delete_table_monetdb(monet_conn, table_name)
                    pbar.update(len(batch_data))
                elapsed_time = time.time() - start_time
                d_monet_batch_times[batch_size].append(elapsed_time)
                print(f"MonetDB : {batch_size} lignes supprimées en {elapsed_time:.2f} secondes.")
            except Exception as e:
                print(f"Erreur suppression MonetDB : {e}")

    # Calcul des moyennes et écarts-types pour PostgreSQL et MonetDB
    i_postgres_avg_times = {}
    i_postgres_std_times = {}
    i_monet_avg_times = {}
    i_monet_std_times = {}
    d_postgres_avg_times = {}
    d_postgres_std_times = {}
    d_monet_avg_times = {}
    d_monet_std_times = {}

    for batch_size in batch_sizes:
        if len(i_postgres_batch_times[batch_size]) > 0:
            i_postgres_avg_times[batch_size] = np.mean(i_postgres_batch_times[batch_size])
            i_postgres_std_times[batch_size] = np.std(i_postgres_batch_times[batch_size])
        else:
            i_postgres_avg_times[batch_size] = np.nan
            i_postgres_std_times[batch_size] = np.nan

        if len(i_monet_batch_times[batch_size]) > 0:
            i_monet_avg_times[batch_size] = np.mean(i_monet_batch_times[batch_size])
            i_monet_std_times[batch_size] = np.std(i_monet_batch_times[batch_size])
        else:
            i_monet_avg_times[batch_size] = np.nan
            i_monet_std_times[batch_size] = np.nan

        if len(d_postgres_batch_times[batch_size]) > 0:
            d_postgres_avg_times[batch_size] = np.mean(d_postgres_batch_times[batch_size])
            d_postgres_std_times[batch_size] = np.std(d_postgres_batch_times[batch_size])
        else:
            d_postgres_avg_times[batch_size] = np.nan
            d_postgres_std_times[batch_size] = np.nan

        if len(d_monet_batch_times[batch_size]) > 0:
            d_monet_avg_times[batch_size] = np.mean(d_monet_batch_times[batch_size])
            d_monet_std_times[batch_size] = np.std(d_monet_batch_times[batch_size])
        else:
            d_monet_avg_times[batch_size] = np.nan
            d_monet_std_times[batch_size] = np.nan

    print("Insertion 6 attributs")
    print(f"Temps moyens PostgreSQL : {i_postgres_avg_times}")
    print(f"Temps écarts-types PostgreSQL : {i_postgres_std_times}")
    print(f"Temps moyens MonetDB : {i_monet_avg_times}")
    print(f"Temps écarts-types MonetDB : {i_monet_std_times}")

    print("suppression 6 attributs")
    print(f"Temps moyens PostgreSQL : {d_postgres_avg_times}")
    print(f"Temps écarts-types PostgreSQL : {d_postgres_std_times}")
    print(f"Temps moyens MonetDB : {d_monet_avg_times}")
    print(f"Temps écarts-types MonetDB : {d_monet_std_times}")

    return i_postgres_avg_times, i_postgres_std_times, i_monet_avg_times, i_monet_std_times, d_postgres_avg_times, d_postgres_std_times, d_monet_avg_times, d_monet_std_times

def experiment_on_increasing_rows(postgres_conn, monet_conn, data, num_tests):

    # Générer des tailles de lots croissantes
    batch_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    # Dictionnaires pour stocker les résultats
    i_postgres_avg_times_2_attibus = {}
    i_postgres_std_times_2_attibus = {}
    i_monet_avg_times_2_attibus = {}
    i_monet_std_times_2_attibus = {}
    d_postgres_avg_times_2_attibus = {}
    d_postgres_std_times_2_attibus = {}
    d_monet_avg_times_2_attibus = {}
    d_monet_std_times_2_attibus = {}

    i_postgres_avg_times_6_attibus = {}
    i_postgres_std_times_6_attibus = {}
    i_monet_avg_times_6_attibus = {}
    i_monet_std_times_6_attibus = {}
    d_postgres_avg_times_6_attibus = {}
    d_postgres_std_times_6_attibus = {}
    d_monet_avg_times_6_attibus = {}
    d_monet_std_times_6_attibus = {}

    # Pour chaque taille de lot, on effectue l'expérience
    for batch_size in batch_sizes:
        print(f"Expérience pour {batch_size} lignes:")

        # Générer un nom unique de table pour chaque taille de lot (par exemple: taxi_trips_100, taxi_trips_200, etc.)
        table_name = f"taxi_trips_{batch_size}"

        # Appel de l'expérience pour une taille de lot donnée
        i_p_avg_times_2, i_p_std_times_2, i_m_avg_times_2, i_m_std_times_2, d_p_avg_times_2, d_p_std_times_2, d_m_avg_times_2, d_m_std_times_2 = experiment_storage_engine_2_attibus(
            postgres_conn, monet_conn, data, [batch_size], table_name, num_tests)
        i_p_avg_times_6, i_p_std_times_6, i_m_avg_times_6, i_m_std_times_6, d_p_avg_times_6, d_p_std_times_6, d_m_avg_times_6, d_m_std_times_6 = experiment_storage_engine_6_attibus(
            postgres_conn, monet_conn, data, [batch_size], table_name, num_tests)

        # Stocker les résultats dans les dictionnaires
        i_postgres_avg_times_2_attibus.update(i_p_avg_times_2)
        i_postgres_std_times_2_attibus.update(i_p_std_times_2)
        i_monet_avg_times_2_attibus.update(i_m_avg_times_2)
        i_monet_std_times_2_attibus.update(i_m_std_times_2)
        d_postgres_avg_times_2_attibus.update(d_p_avg_times_2)
        d_postgres_std_times_2_attibus.update(d_p_std_times_2)
        d_monet_avg_times_2_attibus.update(d_m_avg_times_2)
        d_monet_std_times_2_attibus.update(d_m_std_times_2)

        i_postgres_avg_times_6_attibus.update(i_p_avg_times_6)
        i_postgres_std_times_6_attibus.update(i_p_std_times_6)
        i_monet_avg_times_6_attibus.update(i_m_avg_times_6)
        i_monet_std_times_6_attibus.update(i_m_std_times_6)
        d_postgres_avg_times_6_attibus.update(d_p_avg_times_6)
        d_postgres_std_times_6_attibus.update(d_p_std_times_6)
        d_monet_avg_times_6_attibus.update(d_m_avg_times_6)
        d_monet_std_times_6_attibus.update(d_m_std_times_6)

    return i_postgres_avg_times_2_attibus, i_postgres_std_times_2_attibus, i_monet_avg_times_2_attibus, i_monet_std_times_2_attibus, d_postgres_avg_times_2_attibus, d_postgres_std_times_2_attibus, d_monet_avg_times_2_attibus, d_monet_std_times_2_attibus, i_postgres_avg_times_6_attibus, i_postgres_std_times_6_attibus, i_monet_avg_times_6_attibus, i_monet_std_times_6_attibus, d_postgres_avg_times_6_attibus, d_postgres_std_times_6_attibus, d_monet_avg_times_6_attibus, d_monet_std_times_6_attibus

def plot_results(batch_sizes, postgres_avg_times, postgres_std_times, monet_avg_times, monet_std_times, xlabel, ylabel, title, label1, label2):
    plt.figure(figsize=(10, 6))

    # Tracer les moyennes
    plt.plot(batch_sizes, list(postgres_avg_times.values()), label=label1, marker='o')
    plt.plot(batch_sizes, list(monet_avg_times.values()), label=label2, marker='o')

    # Tracer les barres d'erreur pour l'écart-type
    plt.errorbar(batch_sizes, list(postgres_avg_times.values()), yerr=list(postgres_std_times.values()), fmt='o', color='blue', alpha=0.5, capsize=5)
    plt.errorbar(batch_sizes, list(monet_avg_times.values()), yerr=list(monet_std_times.values()), fmt='o', color='orange', alpha=0.5, capsize=5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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

        # 7. Tests de performance pour les insertions
        i_postgres_avg_times_2, i_postgres_std_times_2, i_monet_avg_times_2, i_monet_std_times_2, d_postgres_avg_times_2, d_postgres_std_times_2, d_monet_avg_times_2, d_monet_std_times_2, i_postgres_avg_times_6, i_postgres_std_times_6, i_monet_avg_times_6, i_monet_std_times_6, d_postgres_avg_times_6, d_postgres_std_times_6, d_monet_avg_times_6, d_monet_std_times_6 = experiment_on_increasing_rows(postgres_conn, monet_conn, data_tuples, 5)
        pbar.update(1)

        # 8. Visualisation des résultats
        plot_results(i_postgres_avg_times_2.keys(), i_postgres_avg_times_2, i_postgres_std_times_2, i_monet_avg_times_2, i_monet_std_times_2, "Nombre de lignes insérées", "Temps d'insertion (s)", "Graphique 1 : Performance d'insertion - PostgreSQL vs MonetDB avec 2 attributs", "PostgreSQL", "MonetDB")
        plot_results(d_postgres_avg_times_2.keys(), d_postgres_avg_times_2, d_postgres_std_times_2, d_monet_avg_times_2, d_monet_std_times_2, "Nombre de lignes supprimées", "Temps de suppression (s)", "Graphique 2 : Performance de suppression - PostgreSQL vs MonetDB avec 2 attributs", "PostgreSQL", "MonetDB")
        plot_results(i_postgres_avg_times_6.keys(), i_postgres_avg_times_6, i_postgres_std_times_6, i_monet_avg_times_6, i_monet_std_times_6, "Nombre de lignes insérées", "Temps d'insertion (s)", "Graphique 3 : Performance d'insertion - PostgreSQL vs MonetDB avec 6 attributs", "PostgreSQL", "MonetDB")
        plot_results(d_postgres_avg_times_6.keys(), d_postgres_avg_times_6, d_postgres_std_times_6, d_monet_avg_times_6, d_monet_std_times_6, "Nombre de lignes supprimées", "Temps de suppression (s)", "Graphique 4 : Performance de suppression - PostgreSQL vs MonetDB avec 6 attributs", "PostgreSQL", "MonetDB")
        plot_results(i_postgres_avg_times_6.keys(), i_postgres_avg_times_2, i_postgres_std_times_2, i_postgres_avg_times_6, i_postgres_std_times_6, "Nombre de lignes insérées", "Temps d'insertion (s)", "Graphique 5 : Performance d'insertion PostgreSQL - 2 vs 6 attributs", "2 attributs", "6 attributs")
        plot_results(d_postgres_avg_times_6.keys(), d_postgres_avg_times_2, d_postgres_std_times_2, d_postgres_std_times_6, d_postgres_std_times_6, "Nombre de lignes supprimées", "Temps de suppression (s)", "Graphique 6 : Performance de suppression PostgreSQL - 2 vs 6 attributs", "2 attributs", "6 attributs")
        plot_results(i_monet_avg_times_6.keys(), i_monet_avg_times_2, i_monet_std_times_2, i_monet_avg_times_6, i_monet_std_times_6, "Nombre de lignes insérées", "Temps d'insertion (s)", "Graphique 7 : Performance d'insertion MonetDB - 2 vs 6 attributs", "2 attributs", "6 attributs")
        plot_results(d_monet_avg_times_6.keys(), d_monet_avg_times_2, d_monet_std_times_2, d_monet_avg_times_6, d_monet_std_times_6, "Nombre de lignes supprimées", "Temps de suppression (s)", "Graphique 8 : Performance de suppression MonetDB - 2 vs 6 attributs", "2 attributs", "6 attributs")

        # 9. Fermeture des connexions
        postgres_conn.close()
        monet_conn.close()
        pbar.update(1)

if __name__ == "__main__":
    main()