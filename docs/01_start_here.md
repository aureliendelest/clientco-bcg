# ClientCo : démarrer ici

> **Rôle** : Point d'entrée : le cas, la mission, les échéances, où on en est, les règles d'équipe, l'ordre de lecture.
> **Pour** : Toute personne qui rejoint l'équipe (20 minutes de lecture).
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. En 60 secondes

- **Le cas.** Étude de cas CentraleSupélec × BCG (M2AD). Un distributeur fictif de matériaux de construction, **ClientCo** (15,5 Md€ de CA, > 1 000 agences, > 450 k clients), a une nouvelle Chief Data Officer (CDO) qui doit convaincre un CEO et un ExCom réticents de la valeur de la data.
- **Notre mission.** Lister 8–10 use cases data, en prioriser 5 dans une feuille de route 2027, **traiter le n°1 en profondeur (avec un modèle)**, définir les besoins en données d'un 2ᵉ use case, et demander un budget.
- **Ce qu'on a choisi.** Le n°1 est la **prévention du churn** : prédire quels clients vont arrêter d'acheter, pour que les commerciaux les appellent en priorité.
- **L'idée qui structure tout.** Le but n'est pas de prédire qui part, mais de **déclencher une action commerciale qui change le résultat, et de le prouver**. Prédire n'est pas prévenir. On classe donc les clients par **euros espérés protégés**, pas par probabilité seule.
- **Où on en est.** Le code est gelé et le test final a été lancé le 26/11 : avec une liste de **700 clients**, le modèle capte **51,5 %** du CA perdu des clients réguliers (26,1 % de tout le churn). Détail : [03_evidence.md](03_evidence.md), § 11. Écarts connus : [07 § 7](07_risks_and_open_questions.md).
- **Échéances.** 26/11 à 12h : code. 26/11 à 17h : slides. 27/11 : présentation au CEO.

## 2. Le cas : mission, livrables, règles du jeu

### Livrables attendus (brief)

| Quoi | Quand |
| --- | --- |
| Liste des use cases, priorisation, issue tree, première structure de l'algorithme (Case Team Meeting + tutorat BCG) | **20/11** |
| **Modèles algorithmiques** (code) | **26/11, 12h** |
| **Deck de 5 à 10 planches** (+ annexes) | **26/11, 17h** |
| Présentation au CEO : 45 min + 15 min de débrief, devant 3 jurys | **27/11** |

Le deck doit contenir : les use cases classés avec leur valeur, la feuille de route 2027 du top 5, l'analyse du use case n°1 (modèle et plan de mise en œuvre), les besoins en données et la solution d'un 2ᵉ use case, et le budget et les ressources 2027.

### Ce que le jury attend

De la structure, des hypothèses fondées sur le business, un modèle qui répond au problème, des plans actionnables et réalistes, une histoire claire (storyline), de la créativité.

### Règles du brief à ne pas oublier

- **Ne livrer que ce que l'on sait expliquer et défendre.** Pas de boîte noire, pas de formule dont on ne sait pas refaire le calcul.
- **Vérifier chaque chiffre** et l'étiqueter : **Confirmé**, **Source unique**, **Déduit** ou **Manquant** (le fichier de priorisation ajoute « Hypothèse équipe »).
- **Confidentialité.** Le brief est confidentiel. On ne colle pas les comptes rendus ni les données dans des outils d'IA non approuvés. Une IA peut inventer des chiffres ou des sources, et elle a tendance à nous donner raison : relire et recalculer.
- **Agir comme dans la vraie vie** : on peut faire des hypothèses, mais on les dit.

### Le matériel du cas

- **Données** : un seul fichier de transactions, `dataset_sales_2026.csv` (~4 Go, ~63 M de lignes). Colonnes : `date_order`, `date_invoice`, `product_id`, `client_id`, `sales_net`, `quantity`, `order_channel`, `branch_id`. Il ne couvre qu'**un sous-ensemble** des agences et des clients, sur 2 ans.
- **8 comptes rendus d'interviews** (CDO, Head of Strategy, Sales Manager, Supply Chain, équipe data, ancien client BCG « OffSupp », Head of Sales, Commercial Director) et des Key Facts, dans `docs/subject/` (non versionné, voir section 6). `Financials.xlsx` est dans `data/`.
- Ces documents sont dans `docs/subject/` (**non versionné**, voir section 6).

## 3. Les enjeux

### Enjeux business

- **Prouver la valeur vite.** Budget serré, ExCom réticent, CEO sans cap clair. La Commercial Director cite la mesure du succès comme son « deep pain point ».
- **Le goulot est la capacité commerciale.** \~200 commerciaux, une visite coûte 3 h + 1 h de suivi, le téléphone est jugé « le meilleur » canal. Le seul indicateur utile est la qualité des K clients que l'on peut réellement traiter.
- **L'objectif est le chiffre d'affaires.** On classe donc par euros espérés protégés, pas par probabilité seule.
- **L'adoption est le premier risque de déploiement.** OffSupp : la plupart des équipes n'ont pas adopté les nouveaux processus immédiatement. Le Sales Manager doute que la data « résume » une relation client, et le CRM est peu rempli.
- **Le chiffrage actuel est fragile.** Le +1 % de revenus de la CDO est une source unique, d'une autre entreprise.

### Enjeux techniques

- **Churn non contractuel.** Aucune résiliation : le client cesse d'acheter. La définition de la perte est un choix, pas une donnée.
- **Base très hétérogène.** Occasionnels, PME régulières, clients sous contrat, institutions publiques, grands comptes : un seul modèle et un seul seuil ne tiennent pas.
- **Peu de données.** Transactions seulement, sur un sous-ensemble d'agences et de clients, deux ans d'historique.
- **Corrélation, pas causalité.** Un client à risque n'est pas forcément un client « persuadable ». Sans groupe témoin, aucun CA « sauvé » n'est vérifiable.

### Ce que disent les interviews

| Interlocuteur | Position sur le churn |
| --- | --- |
| CDO | Le juge le plus faisable : données de ventes accessibles, peu d'effectifs. Avance +1 % de revenus (expérience passée, source unique). |
| Head of Sales | « Hot topic depuis des années », jamais de solution efficace ; veut embarquer ses équipes. |
| Commercial Director | Impact financier potentiellement majeur, mais doute que la data soit la priorité. |
| Sales Manager | Sceptique (« impédictible »), prêt à essayer ; préfère un outil de cross-sell. |
| OffSupp (benchmark) | Quick win facile à piloter ; le défi est le déploiement auprès des équipes locales. |
| Supply Chain | Ruptures et délais seraient les premières causes de départ (non testable sans données de stock). |

## 4. La démarche, étape par étape

```
Chaîne de valeur  →  9–10 use cases  →  priorisation  →  churn en n°1  →  exploration des données
                                                                              ↓
 slides ← simulation d'impact ← évaluation en CA ← modèles ← photos+labels ← cadrage de la méthode
```

### Étape 1 – Cartographier la chaîne de valeur

ClientCo fait sa marge (≈ 25 % du CA) en achetant bien et en revendant via son réseau. On a repéré les pain points cités dans les interviews sur six maillons : achats, supply chain et stocks, assortiment et prix, marketing et promotions, ventes et canaux, relation client. Le support correspondant n'est pas dans le dépôt.

### Étape 2 – Lister et prioriser les use cases

Tableur de priorisation (feuilles « Priorisation » et « Chiffrage Top 5 », hors dépôt). Chaque use case est noté sur :

| Critère | Poids |
| --- | --- |
| Impact financier (marge brute annuelle) | 40 % |
| Faisabilité technique | 30 % |
| Délai d'implémentation | 15 % |
| Soutien interne (interviews) | 15 % |

Résultat : **top 5** = 1) prévention du churn, 2) ciblage de l'effort commercial, 3) cross-sell, 4) prévision de la demande et stocks, 5) ROI des promotions. Les promotions sont le candidat pour le **2ᵉ use case** (définir les données manquantes). La segmentation client et le socle data sont jugés indirects : ils servent de briques ou de prérequis.

### Étape 3 – Choisir le churn comme deep dive

Raisons : c'est le **seul use case dont les données sont disponibles dès le premier jour** (les transactions suffisent), c'est le plus cité dans les interviews (5 sur 8), la CDO le juge le plus faisable, et un modèle de classification classique suffit. C'est aussi un « quick win » mesurable, ce dont la CDO a besoin devant son ExCom.

### Étape 4 – Explorer les données (notebooks `01` à `03`)

On y a vérifié la qualité des données, décrit les clients et leur rythme d'achat, mesuré l'enjeu et testé des signaux avant-coureurs. Constats principaux : [04_data_findings.md](04_data_findings.md).

### Étape 5 – Cadrer la méthode et l'architecture (conversation du 17–18/11)

Une série de décisions, résumées dans [02_modeling_decisions.md](02_modeling_decisions.md) : définition de la cible, score en valeur espérée, évaluation en CA capté, découpage temporel strict, architecture du code.

### Étape 5 bis – Fixer le périmètre et les labels (24–25/11, notebooks `04` à `08`)

Périmètre des clients réguliers, churn total contre partiel, vérité à 180 jours, deux labels d'entraînement comparés, capacité de 500 appels par semaine. Les décisions sont dans [02_modeling_decisions.md](02_modeling_decisions.md) ; les preuves dans [03_evidence.md](03_evidence.md).

### Étape 6 – Écrire le code (fait le 25/11)

Paquet Python `src/churn/` + configurations YAML + tests anti-fuite : voir [06_engineering.md](06_engineering.md).

### Étape 7 – Expériences, gel, test final (25-26/11)

Backtest des modèles, comparaison des jeux de variables, réglage Optuna, gel de la configuration, test du 22/03/2026. Résultats : [03_evidence.md](03_evidence.md), § 11. Reste : les slides.

## 5. État d'avancement au 26/11

| Chantier | État |
| --- | --- |
| Priorisation des use cases, chiffrage du top 5 | Fait (chiffrage du churn à mettre à jour, voir plus bas) |
| Exploration, périmètre et labels (notebooks `01` à `09`) | Fait |
| Code (`data`, `snapshots`, `models`, `pipeline`, `evaluation`, `scoring`, `impact`, `comparison`, `simulation`) | Fait ; la suite de tests donne 198 OK, 2 skippés |
| Configuration finale | Gelée : variables `basic`, label A, XGBoost réglé (`configs/experiments/final.yaml`) |
| Test final (22/03/2026) | Lancé le 26/11 |
| Tableau de bord | Page HTML locale de rejeu des listes (`front/`), à la place de la maquette Streamlit |
| Notebook de rapport (`experiment_report.ipynb`) | **Non rempli** (cellules TODO) |
| Liste mixte total / partiel | **Non livrée** |

**Deux écarts connus à ne pas reproduire dans le deck :**

- Le fichier de priorisation chiffrait le churn avec **8 % de CA perdu par an** (30,6 M€ de marge) et une détection de 50 %. La part perdue mesurée est de **3,4 à 4,8 %** et la détection mesurée au test de **26,1 %** : voir [05_impact_and_pilot.md](05_impact_and_pilot.md). La mise à jour du fichier (hors dépôt) n'est pas vérifiée.
- Les notebooks `01` à `03` gardent la définition d'origine du jour d'achat (ligne de vente positive) ; à partir du `04`, c'est le CA net des retours du jour > 0,01 €. Les décisions de périmètre et de label s'appuient sur les notebooks `04` à `08`.

## 6. Règles d'équipe

### Git

- Branches courtes, petites pull requests vers `main`.
- **Une configuration déjà exécutée ne se modifie jamais** : on en crée une nouvelle.
- On versionne : code, configs, `results/runs.parquet`, `results/metrics.parquet`.
- On ne versionne **jamais** : données, `runs/`, `predictions.parquet`, `importances.parquet`, modèles (`*.pkl`), `data/processed/`.

### Confidentialité

Le brief est classé confidentiel et le dépôt a un remote GitHub. **Ne commitez rien de `docs/subject/`** (brief, interviews, financials) ni aucun fichier de données. Vérifiez `git status` avant chaque commit. **Attention :** `docs/subject/` n'est pas dans `.gitignore`, et d'anciens documents du cas restent dans l'historique Git ([07 § 7](07_risks_and_open_questions.md)). Ne collez pas les comptes rendus dans des outils d'IA non approuvés.

### Utiliser l'IA correctement

Vous restez maîtres du récit et des chiffres. Avant de livrer : recalculez, vérifiez la source, étiquetez la preuve (Confirmé / Source unique / Déduit / Manquant), et soyez capable d'expliquer chaque formule à un jury.

### Pièges fréquents

- Utiliser des dates non décalées dans le code (ou décaler deux fois). Le décalage vit **uniquement** dans `data.ingest`.
- Entraîner puis tester sur des photos qui se chevauchent.
- Choisir un seuil sur le test.
- Calculer des variables avec des données postérieures à T.
- Charger le CSV dans pandas.
- Citer un chiffre d'un notebook sans l'avoir relu dans sa sortie exécutée (les notebooks `01` à `03` gardent l'ancienne définition du jour d'achat).

## 7. Par où commencer, dans quel ordre lire

| Doc | Contenu | Quand |
|---|---|---|
| [02_modeling_decisions.md](02_modeling_decisions.md) | **Les décisions de modélisation, état actuel** (une page) | En premier |
| [03_evidence.md](03_evidence.md) | Les preuves chiffrées derrière chaque décision (lemmes 1 à 7) | Pour défendre un choix |
| [04_data_findings.md](04_data_findings.md) | Ce que disent les données, anomalies, saisonnalité | Avant de toucher aux données |
| [05_impact_and_pilot.md](05_impact_and_pilot.md) | Impact en M€, scénarios, plans A et B, pilote | Pour le récit et le deck |
| [06_engineering.md](06_engineering.md) | Architecture du code, commandes, contrats, ordre d'implémentation | Avant de coder |
| [07_risks_and_open_questions.md](07_risks_and_open_questions.md) | Vigilances, questions ouvertes, prochaines étapes | Chaque semaine |
| [08_decision_log.md](08_decision_log.md) | Historique daté des décisions et des corrections | Pour comprendre pourquoi |
| [09_presentation_notes.md](09_presentation_notes.md) | Storyline du deck, discours de justification, questions probables | Avant le 27/11 |
| [glossary.md](glossary.md) | Vocabulaire du projet | À tout moment |

**30 minutes (tout le monde)** : ce document, puis `docs/subject/Brief - ClientCo 2026 - TO USE.pptx` (la consigne officielle).

**2 heures (avant de coder)** : `02_modeling_decisions.md`, puis les notebooks `02` (définition du churn), `04` et `05` (périmètre et labels), `06` à `08` (calibration des labels), puis les docstrings de `src/churn/*.py` (elles spécifient les briques). Les notebooks sont numérotés dans l'ordre de lecture ; la carte est dans le [README](../README.md).

**Une question ?** Elle est probablement déjà traitée dans ces documents : cherchez-y d'abord, puis demandez à l'équipe.
