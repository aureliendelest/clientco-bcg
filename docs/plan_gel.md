# Plan de gel (25-26/11/2026)

> **Rôle** : brief d'exécution pour les trois flux E, F, S et pour l'intégrateur (I). Document de travail, conservé après le gel car le code le cite (résultats : `03_evidence.md`, § 11).
> **Décisions validées** : ordre 1, 2, 3, 4, 7, 9, 10, 5, 6 ; Option 2 ferme ; critères Q8 et Q9 ci-dessous ; aucun push.
> **Échéances (brief)** : code **26/11 12h**, slides **26/11 17h**, présentation 27/11.

## 1. Correspondance avec la liste de priorités

| # | Tâche | Flux | Phase |
|---|---|---|---|
| 1 | Oracles et CA capté sur les deux dénominateurs | E | 1 |
| 2 | Comparaison appariée par date | E | 1 |
| 3 | Nouvelles variables (`extended`) | F | 1 (code), 2 (sweep) |
| 4 | Impact et tornade en code, liste du churn total, raisons, nettoyage des stubs (Option 2) | S | 1 |
| 7 | Évaluation sous perte récente (`CA 90 j × 4`) | E | 3 |
| 9 | Listes mensuelles avec repos de 3 mois | E | 3 |
| 10 | Optuna, une fois, sur la configuration retenue | I | 3 |
| 5 | Date du 22/09/2025 signalée (une seule photo d'entraînement) | I | 4 |
| 6 | Gel, puis test du 22/03/2026, une seule fois | I | 4 |

**Hors plan (choix de l'équipe)** : grands comptes (top 1 % et 5 % de V), liste mixte et score partiel dans le pipeline (Option 3), survie, BG/NBD comme variable, perte pondérée par V. Ils sont annoncés comme limites ou feuille de route.

## 2. Critères d'acceptation (écrits avant tout résultat)

**Variables (Q8).** Le jeu `extended` est retenu contre `basic` si, **sur HGB et sur XGB**, à K = 700, sur les réguliers, label A, sur les **5 dates du 22/10/2025 au 22/02/2026** :
- gain moyen apparié d'au moins **+1,5 point** de CA capté ;
- au moins **4 dates sur 5** gagnantes ;
- moyenne des écarts à K = 2 000 supérieure ou égale à 0.

La date du 22/09/2025 est rapportée mais hors décompte. Si le critère échoue : **un seul repli**, le groupe « tendance relative au marché » seul : `basic` avec `freq_trend` et `rev_trend` remplacées par `mkt_freq_trend` et `mkt_rev_trend`, jugé sur le même critère (jeu `market_only`). S'il échoue aussi : `basic`. Aucune ablation groupe par groupe, aucun autre sauvetage.

**Modèle et label du test (Q9).** Label A, sauf si B passe le critère ci-dessus sur HGB **et** XGB. Famille : meilleur CA capté moyen à K = 700 sur les 5 dates avec le jeu retenu ; XGB si l'écart est inférieur à 1 point.

**Calendrier (Q6).** Coupure des variables **26/11 9h** : si le sweep n'est pas lu et tranché, gel de `basic`. Gel de la configuration **11h**. Code livré **12h**. Test l'après-midi, chiffres dans les slides avant 17h.

## 3. Règles communes

1. **Un fichier, un propriétaire** (tableau section 4). Un besoin sur un fichier d'autrui s'écrit dans la note de passage, pas dans le fichier.
2. **Interdit** : toute exécution, lecture de snapshot ou calcul avec la date `2026-03-22` (test). Le garde-fou de la phase 0 bloque `pipeline.run` sur le jeu de données réel avec cette date, sauf `CHURN_FINAL_TEST=1`.
3. Chaque agent n'exécute que **ses propres tests** (`pytest tests/<son fichier>`), jamais toute la suite ni `churn collect`. Seul I modifie `results/`.
4. **Aucun commit, aucun push.** I commite à chaque jonction.
5. **Un seul processus DuckDB lourd à la fois** : seul F construit des snapshots à partir du Parquet. E lit les snapshots en cache, S en construit un seul à la fois.
6. Non-régression : les valeurs des métriques existantes (204 runs) restent identiques à la référence sauvegardée en phase 0 (tolérance 1e-9).
7. Code et commentaires en anglais comme aujourd'hui ; textes affichés (raisons) en français.
8. Note de passage en fin de mission : `docs/plan_gel_handoff_<flux>.md` (ce qui change, interface, tests lancés, points ouverts, besoins sur d'autres fichiers).

## 4. Propriétés de fichiers

| Flux | Fichiers qu'il possède |
|---|---|
| **E** | `src/churn/evaluation.py`, `src/churn/comparison.py` (nouveau), `src/churn/simulation.py` (nouveau), `tests/test_evaluation.py` (nouveau) |
| **F** | `src/churn/data.py`, `src/churn/snapshots.py`, `tests/test_snapshots.py`, `tests/test_no_leakage.py`, `configs/sweeps/features.yaml` (nouveau) |
| **S** | `src/churn/scoring.py`, `src/churn/impact.py` (nouveau), `src/churn/reasons_fr.yaml`, `tests/test_scoring.py`, `tests/test_impact.py` (nouveaux) |
| **I** | `pipeline.py`, `config.py`, `cli.py`, `configs/base.yaml`, `Makefile`, `README.md`, `docs/`, `results/`, `runs/` |

## 5. Contrats d'interface

**C1. Univers (oracles).** `evaluation.load_universe(cfg) -> DataFrame` lit les snapshots **en cache** de la config (via `snapshots.load_or_build`) et renvoie, pour la date du run, **tous** les clients : `client_id, snapshot_date, V, gone, rhythm, rev_90d`. Le périmètre se déduit de `rhythm` (`in_perimeter`). `evaluate_predictions(pred, cfg, universe=None)` : sans univers, comportement actuel inchangé.

Nouvelles métriques, segment `all`, avec K, même table longue et mêmes intervalles de confiance :

| Nom | Définition |
|---|---|
| `capture_total_at_k` | V des partants du top K du modèle (classé par `ev`, réguliers) ÷ V de **tous** les partants de l'univers |
| `oracle_total_capture` | top K de tous les clients par `V × gone` ÷ même dénominateur |
| `size_only_total_capture` | K plus gros clients de tout l'univers (par V) ÷ même dénominateur |

Les métriques existantes (`capture_at_k`, `size_only_capture`, `oracle_capture`) gardent leur nom et leur valeur : dénominateur réguliers. Les trois nouvelles sont **calculées automatiquement quand un univers est fourni** et ne sont **pas** ajoutées à `evaluation.metrics` : `config_hash` couvre toute la configuration, donc modifier `base.yaml` changerait le `run_id` des 204 runs existants et les relancerait tous. I branche `load_universe` dans `pipeline.collect` en phase 2.

**C2. Comparaison appariée.** `comparison.paired_by_date(metrics, runs_a, runs_b, k) -> DataFrame` (une ligne par date : valeur A, valeur B, écart en points, gagnant) et `comparison.accept(table, min_gain=1.5, min_wins=4, excluded_dates=("2025-09-22",), k_check=2000) -> dict` (verdict par modèle et global). Seuils paramétrables, valeurs par défaut = critère section 2.

**C3. Variables.** `FEATURE_SETS["extended"]` = `basic` **moins** `freq_trend` et `rev_trend`, **plus** les 7 variables ci-dessous (17 colonnes). Les deux variables ajustées au marché **remplacent** les tendances brutes : avec les deux ensemble, le modèle retrouverait la constante du marché par date, c'est-à-dire un identifiant de date (7 valeurs d'entraînement), ce que la doc interdit déjà avec « mois ». Pas de `SNAPSHOT_VERSION` : `feature_set` est déjà dans la clé de cache, donc le cache `basic` reste valide (F le vérifie par comparaison du cache reconstruit à l'existant). Nouvelles variables, **historique strictement avant T** :

| Nom | Sens |
|---|---|
| `mkt_freq_trend` | remplace `freq_trend` : `freq_trend` du client moins la tendance de l'activité globale (log du nombre total de jours d'achat de tous les clients sur 90 j, moins le même nombre sur les 90 j précédents) |
| `mkt_rev_trend` | remplace `rev_trend` : idem sur le CA |
| `negative_lines_share_180d` | somme des lignes à `sales_net < 0` ÷ somme des lignes positives, 180 j. La quantité n'est jamais négative : avoirs ou remises possibles, donc **jamais** nommée « retours » dans les raisons (question 5 du tutorat sans réponse) |
| `product_div_trend` | `log1p(produits distincts sur 90 j) - log1p(produits distincts sur les 90 j précédents)` |
| `product_loss_share` | part du CA des 90 j précédents venant de produits non rachetés sur les 90 derniers jours ; **NaN si le client n'a aucun achat dans l'une des deux fenêtres** (sinon il vaut 1 et duplique la récence) |
| `channel_shift` | 1 moins la part du CA des 90 derniers jours dans le canal dominant des 90 jours précédents (NaN si pas d'achat précédent) |
| `branch_freq_trend` | log-ratio des jours d'achat (90 j contre 90 j précédents) de l'agence principale du client (agence au plus fort CA sur 180 j), calculé sur les transactions, **jamais sur les labels**, **hors achats du client lui-même** (leave-one-out) ; NaN si l'agence a moins de 30 jours d'achat dans une des deux fenêtres |

Faits sur les données (`sales_shifted.parquet`) : 5 canaux dont 3 pèsent (magasin 51 % des lignes, téléphone 40 % des lignes et 64 % du CA, en ligne 9 %), 32 % des clients n'ont qu'un canal ; 582 agences, 51 % des clients n'en ont qu'une ; 607 000 produits distincts, pas de catégorie ; lignes négatives : 3,8 % des lignes, −5,8 % du CA.

F peut simplifier une définition si le coût de calcul l'impose, en le notant dans sa note de passage. Budget : construction des 14 snapshots du sweep en moins de 20 min et sous 4 Go de mémoire DuckDB ; sinon F réduit le jeu et le dit.

**C4. Sortie métier.**
- `impact.impact_scenarios(detection, cfg, *, lost_share=None) -> DataFrame` : scénarios de rétention 10, 20 et 30 % × part perdue basse et haute, en M€ de marge par an, brut. Le coût d'opportunité n'est soustrait que s'il est fourni (`opportunity_cost_per_action` est `null` aujourd'hui) et il est signalé dans la table.
- `impact.tornado_table(cfg, ranges) -> DataFrame` : rétention, détection, part perdue ; impact bas et haut, autres paramètres au centre, trié par amplitude. Cas de test : la doc donne ≈ 15,7 M€ au centre (15 504 M€ × 4,1 % × 50 % × 20 % × 24,65 %).
- La détection de référence est `capture_total_at_k` à K = 700 (dénominateur « tout le churn »), choisie par I ; S reçoit la valeur en paramètre.
- `scoring.score_batch(run_id, snapshot_date) -> Path` : charge `runs/<run_id>/model.pkl`, construit le snapshot de la date **sans label**, garde le périmètre, classe par `ev`, garde **K = 700** (`list_size`), écrit `runs/<run_id>/action_list_<date>.csv`. Colonnes : `rank, client_id, p_cal, V, ev, segment, reason_1, reason_2, reason_3`. Pas de quota 25 % (liste du score total seul : limite annoncée dans le README et le deck). Ni `score_kind` ni `suggested_channel`.
- `top_reasons` : SHAP `TreeExplainer` (HGB et XGB, testé) ; coefficients standardisés pour la régression logistique ; aucune raison pour la baseline. Seules les variables qui **augmentent** le risque deviennent des raisons. Gabarits français dans `reasons_fr.yaml`.
- Nettoyage : `mixed_list` supprimée (hors périmètre), les stubs restants deviennent des erreurs explicites « option non prise en charge » ou disparaissent. Plus aucun `NotImplementedError` hérité dans le code livré.

## 6. Missions par flux

**E, phase 1.** `load_universe` et les trois métriques (C1), non-régression sur les 204 runs ; `comparison.py` (C2) ; suppression des stubs `impact_scenarios` et `tornado_table` d'`evaluation.py` (déplacés chez S) et traitement de `threshold_metrics` et `best_f1_threshold` (supprimés ou erreur explicite). Tests dans `tests/test_evaluation.py` avec des prédictions jouets (cas où l'univers contient des clients hors périmètre).

**E, phase 3** (sur la configuration retenue, prédictions déjà en cache) :
- **#7** : pour chaque date, classer par `p_cal × V` et par `p_cal × V_récent` (`V_récent = rev_90d × 4`), et évaluer chaque classement sous les deux pertes (`V` et `V_récent`) ; part de surestimation de `V` chez les partants en déclin (`freq_trend < 0`).
- **#9** : listes mensuelles successives de K = 700 sur les 6 dates, un client déjà listé dans les 3 mois précédents est exclu, CA capté de chaque liste et cumulé (un partant n'est compté qu'une fois).
- Résultats sous forme de fonctions de `simulation.py` renvoyant des tables, sans graphique.

**F, phase 1.** Tables produit, canal, retours et agence dans `data.py` ; variables de C3 dans `snapshots.py` ; `FEATURE_SETS["extended"]` et `FEATURE_SETS["market_only"]` (le repli : `basic` avec les deux tendances remplacées) ; tests : variable invariante quand on modifie des lignes à partir de T (modèle de `test_no_leakage.py`), cas jouets pour chaque variable, `basic` inchangé ; `configs/sweeps/features.yaml` : base `hgb.yaml`, grille `model.type` [hgb, xgb] × `label.kind` [fixed_horizon, relative_horizon] × `snapshots.feature_set` [extended] × les 6 dates du backtest (24 runs ; les runs `basic` existent déjà). Un second fichier `features_fallback.yaml` fait la même grille avec `market_only`, lancé seulement si le critère échoue. F **ne lance aucun run**.

**S, phase 1.** `impact.py` et ses tests ; `list_size`, `apply_min_proba`, `load_reasons`, `top_reasons`, `score_batch` ; `churn score <run_id> --date <date>` fonctionnel (le routage existe dans `cli.py`) ; raisons pour les 12 variables `basic` d'abord ; raisons des variables `extended` seulement après la décision de la phase 2. S teste `score_batch` sur un run du backtest, **jamais sur la date du test**.

## 7. Séquence

| Phase | Qui | Contenu | Durée indicative |
|---|---|---|---|
| **0** | I | Branche `gel-nov26` (l'état actuel est déjà commité sur `main`), commit du plan seul (sans `docs/subject/`, `.vscode/`, `data/`, notebooks non suivis) ; sauvegarde de référence des métriques ; garde-fou `CHURN_FINAL_TEST` dans `pipeline.run` et son test | 30 min |
| **1** | E, F, S en parallèle | Section 6 | 3 à 4 h |
| **2** | I | Intégration de C1 dans `collect` ; `collect` ; sweep `features.yaml` (24 runs, ~10 min) ; comparaison appariée ; application du critère section 2 ; décision `extended`, repli ou `basic` | 1 h |
| **3** | E, S, I | E : #7 et #9 ; S : raisons `extended` si retenues ; I : Optuna sur la configuration retenue (une fois, paramètres figés ensuite) | 2 h |
| **4** | I | #5 (date du 22/09 signalée dans les tableaux) ; gel à 11h ; test du 22/03/2026 une seule fois ; `collect` ; impact avec la détection du test ; liste CSV ; mise à jour du README et de la doc | 12h à 15h |

**Coupures** : 9h le 26/11 pour les variables (sinon `basic`) ; 11h gel ; 12h code livré. Si la phase 1 déborde de plus de 2 h, S passe après E et F dans la file de calcul, jamais avant.

## 8. Risques identifiés

| Risque | Parade |
|---|---|
| Requêtes produit sur 63 M de lignes trop lourdes en mémoire | F valide sur une date d'abord ; budget section 5 ; sinon jeu réduit |
| Comparaisons multiples sur 6 dates corrélées | Critère et repli unique écrits section 2, avant tout résultat |
| Un agent exécute la date du test par erreur | Garde-fou `CHURN_FINAL_TEST` ; règle 2 |
| Collision de fichiers | Section 4 ; besoins d'autrui dans la note de passage |
| Raisons SHAP non défendables | Relues par I avant livraison ; variables qui augmentent le risque seulement ; pas de raison sans gabarit validé |
| Débordement de la phase 1 | Coupure 9h, `basic` en repli : le code de S et E ne dépend pas du sweep |
