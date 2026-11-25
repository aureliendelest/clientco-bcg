# Ingénierie : code, configurations, contrats

> **Rôle** : Comment le code est organisé, comment lancer une expérience, quels contrats respecter, dans quel ordre implémenter.
> **Pour** : Quiconque écrit ou lance du code.
> **À jour au** : 25/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Architecture du code : sobre et reproductible

- Un **paquet Python** (`src/churn/`) piloté par **un fichier YAML par expérience**. Les notebooks servent au récit et aux graphiques, pas au calcul. Ni Airflow, Prefect, DVC ni MLflow (ajoutable plus tard).
- **Principe central : on sauvegarde les prédictions, pas seulement les métriques.** Les métriques se recalculent dans un seul module : changer une définition tard ne demande aucun réentraînement.
- Un **run** = un dossier `runs/<run_id>/` avec la config résolue, le commit Git, l'empreinte des données, la graine, les prédictions, l'importance des variables. Toute option que l'on veut comparer **doit être dans la config**.
- Le **notebook de rapport ne lit que `results/`**.
- **Mise en production simulée** : un script de scoring par lots écrit un CSV (liste courte, score, valeur espérée, trois raisons en français). Pas d'API. L'architecture cible du cas réel tient sur une slide.
- Explicabilité : SHAP pour les modèles boostés, coefficients pour la régression logistique, trois raisons par client en langage commercial (`reasons_fr.yaml`). *À confirmer, voir section 10.*

```mermaid
flowchart LR
  A["CSV brut<br/>jamais modifié"] --> B["Ingestion<br/>+2 ans, une fois"] --> C["Parquet décalé<br/>sales_shifted"] --> D["Photos<br/>DuckDB, par date T"]
  D -- "variables, labels" --> E["Lanceur<br/>run et sweep"]
  F["Configuration<br/>un YAML par essai"] --> E
  E -- "écrit un run" --> G["runs/<br/>1 dossier par run"]
  G --> H["Scoring par lots<br/>liste CSV courte"] --> I["Maquette<br/>optionnelle"]
  G -- "collect" --> J["results/<br/>tables Parquet"] --> K["Notebook<br/>figures et tableaux"]
```

## 2. Décisions d'ingénierie

| Sujet | Décision | Pourquoi |
| --- | --- | --- |
| Forme du code | Paquet `src/churn/` avec point d'entrée unique ; le notebook ne sert qu'au récit et aux graphiques | Les notebooks seuls ne sont ni reproductibles ni testables ; un framework complet serait de l'overkill |
| Orchestration | Lanceur en ligne de commande piloté par fichiers YAML : `run` (une expérience), `sweep` (une série), `collect` (fusion des runs en tables Parquet dans `results/`). Ni Airflow, Prefect, Dagster, DVC ni MLflow | Runs reproductibles et comparables ; MLflow reste ajoutable plus tard |
| Contenu d'un run | Configuration complète résolue, commit Git, empreinte des données, graine, prédictions par client et par date de photo, métriques, importance des variables ; modèle gardé en local | Toute option que l'on veut comparer doit être dans la configuration |
| Principe central | Sauvegarder les prédictions, pas seulement les métriques ; les métriques se recalculent dans un seul module | Changer une définition tard ne demande aucun réentraînement |
| Notebook de rapport | Lit uniquement `results/` et produit tous les graphiques et tableaux | Reprise simple à la fin, résultats réutilisables |
| Données | DuckDB + Parquet pour le calcul lourd ; pandas pour les tables de photos (\~120 000 clients par photo) | Déjà le choix du notebook d'exploration ; `duckdb` est à ajouter à `requirements.txt` |
| Dates | Script d'ingestion unique : +2 ans sur `date_order` et `date_invoice`, écrit `sales_shifted.parquet` ; le CSV brut n'est jamais modifié ; plus aucun décalage ensuite | Éviter un double décalage ; pas de collision, car il n'y a aucune ligne le 29/02/2024 |
| Mise en production simulée | Script de scoring par lots qui écrit un CSV : liste courte, score, valeur espérée, trois raisons principales. Pas d'API | Rien n'est déployé dans l'exercice ; l'architecture cible tient sur une slide |
| Tableau de bord | Maquette d'un seul écran de la liste hebdomadaire du commercial (Streamlit, lecture du CSV de scoring), limitée à une journée ; go ou no-go le 24/11 selon que le modèle est figé et qu'une personne est libre ; capture plutôt que démo en direct | Répond au risque d'adoption ; les résultats d'analyse restent dans les slides |

## 3. Le dépôt et comment travailler dessus

### Mise en route

Prérequis : Python 3.12, ~8 Go de RAM.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # versions exactes, testées
pip install -e ".[dev]"             # le paquet churn (ou : make setup)
python -m ipykernel install --user --name clientco-bcg
make test                           # doit afficher : 38 passed, 4 skipped
```

**Les données ne sont pas dans Git** (~4 Go) : demandez le CSV à Aurélien ou récupérez-le via le dossier du cas, et placez-le dans `data/dataset_sales_2026.csv`. Le notebook `01_data_quality_and_activity_overview` le convertit en `data/sales.parquet` au premier lancement (~1 min).

### Carte du code

```
configs/base.yaml          toutes les options et leurs valeurs par défaut
configs/experiments/*.yaml une expérience = ce qui change par rapport à base.yaml
configs/sweeps/*.yaml      une série d'expériences (grille de surcharges)
src/churn/
  cli.py        commandes : ingest | run | sweep | collect | score
  config.py     configuration typée, chargement YAML, identifiant de run
  data.py       ingestion CSV → Parquet décalé, accès DuckDB
  snapshots.py  variables, labels, segments, découpage temporel
  models.py     baseline, BG/NBD, logreg, HGB, XGBoost, calibration, tuning
  scoring.py    V, VE, taille de liste (K), liste mixte, raisons, liste CSV
  evaluation.py métriques (depuis les prédictions) et simulation d'impact
  pipeline.py   run, sweep, collect
  reasons_fr.yaml  phrases affichées aux commerciaux
tests/          anti-fuite et tests unitaires (data et snapshots actifs ; les autres briques : squelettes skippés)
notebooks/      01 à 08 : exploration, périmètre, labels (preuves) ; experiment_report.ipynb (lit results/ seulement)
docs/           documentation en français ; docs/subject/ = brief du cas (non versionné)
```

### Les commandes

```bash
python -m churn ingest                              # une seule fois : data/sales_shifted.parquet
python -m churn run configs/experiments/hgb.yaml    # une expérience
python -m churn sweep configs/sweeps/models.yaml    # une série
python -m churn collect                             # runs/ → results/*.parquet
python -m churn score <run_id> --date 2026-03-22    # liste d'action hebdomadaire (CSV)
make test
```

Tant qu'une brique est un stub, la commande répond `not implemented yet` (code retour 2) : c'est normal.

### Le contrat d'interface (à respecter, c'est ce qui permet de travailler en parallèle)

- **Table de photo** (`snapshots.py`) : `client_id, snapshot_date, <variables>, label, horizon_days, V, segment`. Variables « basic » : `recency_days, freq_180d, freq_90d, freq_prev_90d, rev_180d, rev_90d, rev_prev_90d, gap_median, gap_p90, recency_ratio, freq_trend, rev_trend` (les deux tendances sont des log-ratios : 0 = stable, < 0 = baisse).
- **Table de prédictions** (`evaluation.py`) : `run_id, client_id, snapshot_date, split, p_raw, p_cal, V, ev, label, gone, horizon_days, segment`.
- **Table de métriques** (`evaluation.evaluate_predictions`, une par run, fusionnée dans `results/metrics.parquet`) : `run_id, split, segment, metric, k, value, ci_low, ci_high`. Segment `all` : `capture_at_k`, `size_only_capture`, `oracle_capture` (une ligne par K de `evaluation.k_values`), `pr_auc`, `brier_raw`, `brier_cal` (`k` vide) ; un segment par valeur de `segment` : probabilités seulement. Définitions et choix dans `02_modeling_decisions.md` § 8 ; la liste des métriques demandées est `evaluation.metrics` (une métrique inconnue ou non implémentée est refusée).
- **Liste d'action** (`scoring.py`) : `rank, client_id, p_cal, V, ev, segment, score_kind, suggested_channel, reason_1, reason_2, reason_3`.

Les docstrings de chaque fonction décrivent le comportement attendu. Le code de référence pour construire les tables client × jour est dans les notebooks `01` et `04`.

### Ordre d'implémentation (les dépendances)

1. `data.ingest`, `data.connect`, `data.build_client_day_table`, `data.fingerprint` : **tout le reste en dépend.**
2. `snapshots` : dates, variables, labels A et B, segments, découpage (labels connus avant le test).
3. `models` (baseline d'abord) et `pipeline.run`.
4. `evaluation` (CA capté @K en premier) et `pipeline.collect`.
5. `scoring` (V, VE, taille de liste, liste mixte, raisons) et `score_batch`.
6. Variantes : label B (relatif), second score (partiel), segments par valeur, Optuna, simulation d'impact et tornade.
7. Si du temps reste : petit réseau de neurones.

À chaque brique implémentée, **retirer le `@pytest.mark.skip` du test correspondant** et le faire passer.

### Ordre des expériences

1. Baseline, logreg, HGB, XGBoost sur le label A (180 jours pour tous).
2. Label B (relatif) sur le meilleur modèle, second score (partiel) et variante par segment de valeur.
3. Optimisation du meilleur modèle.
4. **Gel du code le 25/11.**
5. Seulement s'il reste du temps : réseau de neurones.

## 4. Cadence : photos mensuelles ou hebdomadaires

Mesure du 18/11. Les photos hebdomadaires multiplient les lignes par 3,8 pour une qualité identique ; le gain en cadence ne se voit pas dans les résultats.

| Mesure | Mensuel | Hebdomadaire |
| --- | --- | --- |
| Photos d'entraînement | 7 | 27 |
| Lignes d'entraînement | 834 550 | 3 212 458 |
| Temps d'entraînement | 3 s | 12 s |
| AUC sur la photo de test | 0,860 | 0,860 |
| PR-AUC | 0,513 | 0,511 |
| F1 maximal (précision ; rappel) | 0,570 (0,47 ; 0,72) | 0,569 (0,46 ; 0,75) |
| CA churné capté, top 2 000 / 5 000 / 10 000 | 44,6 / 58,6 / 68,0 % | 45,1 / 58,7 / 68,3 % |

Conditions : 12 variables simples, HistGradientBoosting par défaut sans réglage, photo de test du 22/03/2024 (dates d'origine), classement par P × (CA des 180 derniers jours annualisé) sans décote ρ, label A (aucun achat dans les 180 jours). Le seuil du F1 est choisi sur le test, pour ce comparatif seulement. Mesure unique, à refaire avec les vraies variables.

Recommandation : photos mensuelles pour développer et régler (une optimisation de 50 essais en validation croisée à 5 plis, soit \~250 entraînements, prendrait \~12 min contre \~50 min), puis une vérification en hebdomadaire sur le modèle final.

## 5. Tests anti-fuite

Une suite de tests automatiques à lancer avant chaque série d'expériences (`tests/test_no_leakage.py`, `make test`) : variables de la photo T identiques sur données tronquées à T ; aucune fenêtre de label d'entraînement après la date de test ; AUC proche de 0,5 avec des labels mélangés ; variables de rythme calculées avec les seuls achats antérieurs à T ; calibration ajustée sans le test. Les tests des briques non implémentées sont skippés tant qu'elles sont des stubs.
