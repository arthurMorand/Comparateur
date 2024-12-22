# Configuration du Projet

Ce document vous guide à travers la configuration nécessaire pour exécuter le projet correctement. Vous devez préparer un dossier de configuration contenant les fichiers requis pour l'authentification et les connexions aux bases de données.

---

## Étape 1 : Créer le dossier de configuration

1. À la racine du projet, créez un dossier nommé `config`.

   ```bash
   mkdir config
    ```
   
## Étape 2 : Ajouter les fichiers de configuration

Le dossier `config` doit contenir les fichiers suivants :

---

### 1. `kaggle.json`

Ce fichier est utilisé pour l'authentification au service kaggle.

1. Créez un fichier nommé `kaggle.json` dans le dossier `config` ou téléchargez ce fichier depuis l'onglet `settings` de votre compte [Kaggle](https://www.kaggle.com/settings).
2. Ajoutez-y le contenu suivant :

   ```json
   {
     "username": "votre_nom_d_utilisateur",
     "key": "votre_cle"
   }
### 2. `bdd.ini`

Ce fichier configure les connexions aux bases de données.

1. Créez un fichier nommé `bdd.ini` dans le dossier `config`.
   2. Ajoutez-y le contenu suivant :

      ```ini
      [postgres]
      database = database_name
      user = username
      password = password
      host = host_address
      port = port_number

      [monetdb]
      database = database_name
      user = username
      password = password
      host = host_address
      port = port_number
       ```
   ### 3. Permissions d'accès à `kaggle.json` et `bdd.ini`
   ```bash
       chmod 600 config/kaggle.json config/bdd.ini
    ```
