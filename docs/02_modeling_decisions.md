# Décisions de modélisation

> **Rôle** : **Source de vérité** des décisions de modélisation, état actuel uniquement. L'historique est dans `08_decision_log.md`, les preuves dans `03_evidence.md`.
> **Pour** : L'équipe ; à citer plutôt que recopier.
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Objectif et cadrage

- Maximiser la marge `M = r × m × Σ CA perdu` sur une liste de K clients contactés ; `m = 24,65 %`.
- La rétention `r` n'est pas mesurable : scénarios 10 / 20 / 30 %, mesure par pilote.
- Le score sert à classer des clients, pas à prédire le CA futur : `V` est le CA que le client générerait **s'il restait**, pas une prévision du CA réel (qui donnerait ≈ 0 pour ceux qui partent et compterait le départ deux fois).

## 2. Périmètre et définitions

| Élément | Décision |
|---|---|
| Population | Clients réguliers : ≥ 6 jours d'achat sur les 180 jours avant la photo, écart médian ≤ 30 jours (rapides ≤ 7 j, moyens 8 à 30 j). 51 % des clients, 95 % du CA, 52 % du CA du churn total |
| Hors modèle | Lents (≈ 0,1 % du CA) ; historique court (< 6 achats) : hors modèle (pas d'écart p90 fiable), une règle simple pourra les couvrir plus tard |
| Jour d'achat | CA net des retours du jour > 0,01 € |
| Valeur `V` | CA des 180 jours précédents × 2 |
| Données | +2 ans appliqués une fois (`sales_shifted.parquet`), CSV brut jamais modifié |

## 3. Churn total ou partiel

Définitions des types de perte, avec une photo T et trois fenêtres de 180 j (A avant T, B après T, C après B) :

| Terme | Définition | Perte retenue |
|---|---|---|
| Jour d'achat | CA net des retours du jour > 0,01 € | — |
| Base | Clients ayant au moins un jour d'achat dans A ; modèle : clients réguliers | — |
| CA de référence V | CA de A × 2 (annualisé) | — |
| Churn total | Aucun jour d'achat dans B | V en entier |
| Churn partiel brut | Au moins un achat dans B, mais CA(B) < 50 % de CA(A) (`drop_threshold`) | (CA(A) − CA(B)) × 2, soit en moyenne 71 % de V |
| … dont durable | Partiel et CA(C) < 50 % de CA(A) | idem ; **dimensionnement seulement** |
| Interruption | Ni total ni partiel, mais silence ≥ clip(3 × écart p90, 30, 180 j) clos par un achat dans B | Perte ponctuelle, ≈ 11 M€ ; **non retenue comme label** |
| Contre-poids | CA(B) > 150 % de CA(A) | CA gagné, mesure de volatilité |

Total, partiel et interruption s'excluent ; le durable est un sous-ensemble du partiel, utilisé pour le dimensionnement seulement (360 j de recul : une seule photo utilisable, aucune validation temporelle).

- **Total = label principal et chiffre de tête** : au test final, AUC 0,89 ; CA perdu capté 51,5 % à K = 700 et 66,2 % à K = 2 000 (`03_evidence.md`, § 11).
- **Partiel brut = second score**, à confirmer en pilote, hors chiffre de tête (CA en jeu plus grand, AUC 0,68).
- **Liste mixte : prévue, non livrée.** La liste du code (`churn score`) utilise le seul score total, et aucun run n'a entraîné le score partiel. Le quota d'environ 25 % des places au total (10 à 25 % acceptables) reste une proposition pour le pilote, issue du notebook `05` : deux scores `P(total) × V` et `P(partiel) × V`, listes qui se recouvrent peu (16 % à K = 2 000) ; le total capte alors 40 à 45 % de son CA en jeu contre 56 à 61 % avec toute la liste (K ≈ 1 000 à 1 400).
- **Écartés** : partiel durable (360 jours de recul, une seule photo), label élargi (confondu avec le partiel), interruption (≈ 11 M€, simple variable).

## 4. Labels et horizon

- **Vérité d'évaluation** : « parti » = aucun jour d'achat pendant les **180 jours** suivants, par convention (horizons courts exclus : à 60 jours, 37 % des rapides et 57 % des moyens reviennent ; pas de plateau ensuite).
- **Deux labels d'entraînement comparés** en plan glissant ; **le modèle final utilise A** (section 9).
  - **A** : 180 jours pour tous (même définition que la vérité). 7 photos d'entraînement.
  - **B** : relatif au rythme, `H = clip(3 × écart p90, 45, 90 jours)`. 10 photos. Pureté du label 60 %.
  - Les deux sont dans la configuration (`label.kind` : `fixed_horizon`, `relative_horizon`) ; A est la référence, B l'alternative (section 9).
- **Toute évaluation en CA se fait contre la vérité à 180 jours**, jamais contre le label court (qui surestime le CA perdu d'un facteur 1,7).
- **Décote du CA du churn total : non appliquée.** Environ 12 % de ce CA revient dans les 90 jours ; aucune marge calculée (notebooks, `impact.py`, tableau de bord) ne la déduit : les marges sont des majorants. L'appliquer les réduirait de 12 %. Une mesure directe de la perte réalisée sur 360 jours donne ≈ 2 % (`07_realized_revenue_loss.ipynb`).

## 5. Découpage et cadence

- **Photos mensuelles** : variables calculées avant la date, label après. Les 6 premiers mois ne servent qu'aux variables.
- **Entraînement** : photos du 22/03/2025 au 22/09/2025 (7 photos, label A) ou jusqu'au 22/12/2025 (10 photos, label B) ; `train_end` est dérivé du label dans la configuration. **Test** : 22/03/2026, lancé par le code le 26/11 après le gel ; la même période avait toutefois servi dans les notebooks `04` à `08` (protocole et réserves : sections 10 et `07_risks_and_open_questions.md`).
- **Règle** : le label le plus long d'une photo d'entraînement doit être connu avant la date de test.
- **Cadence** : entraînement mensuel, scoring hebdomadaire (hypothèse ; photos hebdomadaires = 3,8 fois plus de lignes pour la même qualité). Plan glissant pour la valider.
- **Reproductibilité** : lignes triées par client, un seul thread.

## 6. Modèle et score

- **Un seul modèle** avec le segment rapide / moyen en variable (pas de modèle par segment, sauf si le résultat déçoit).
- **Modèles** : baseline (récence ÷ écart p90), **BG/NBD** (repère statistique, non supervisé : `P(aucun achat sur l'horizon)` à partir de `freq_180d` et `recency_days`, sans le label ; `configs/sweeps/backtest_bgnbd.yaml`), régression logistique, HistGradientBoosting, XGBoost ; probabilités calibrées (isotonique). Optuna (30 à 50 essais) une fois le meilleur choisi. Pas de réseau de neurones.
- **Variables** : 12 (jeu `basic`), calculées sur 180 jours (fréquence, CA, tendances, écart médian et p90, récence et récence ÷ p90). Pas de variable « mois ». Les jeux `extended` et `market_only` ont été testés et rejetés (`03_evidence.md`, § 11).
- **Modèle final** : XGBoost, label A, hyperparamètres d'une étude Optuna de 50 essais (`configs/experiments/final.yaml`).
- **Score** : valeur espérée `P(départ) × V`, avec raisons principales ; classement par `p × V` (+15 points de CA capté face au classement par probabilité seule). **La décote ρ est retirée pour l'instant** (elle pourra être remise plus tard).

## 7. Liste et capacité

- **Hypothèse centrale** : 500 appels par semaine (100 commerciaux × 5 appels de 24 minutes), fourchette 200 à 1 800.
- **À l'échelle du dataset** (≈ 30 % du CA) : K ≈ 700 pour une liste mensuelle (600 à 760 appels par mois) ; K ≈ 4 000 pour six listes cumulées sans rappel.
- **Répartition de K entre les deux listes** (non livrée, section 3) : 25 % des places au total ne perd jamais plus de 0,1 M€ face à la meilleure répartition (testé à K = 1 000 et 4 000 ; non testé à 700). Rendements décroissants : doubler la liste cumulée (K 4 000 → 8 000) ajoute +33 % de marge ; la liste mensuelle passée de K 1 000 à 1 400 (+40 % de places) ajoute +15 %.

## 8. Évaluation et marge

- **Mesure principale** : **CA capté @K** = CA (V) des vrais partants dans les K premiers clients classés par valeur espérée, divisé par le CA de tous les vrais partants, au sens de la vérité à 180 jours (`gone`), quel que soit le label d'entraînement. K de tête : **700** ; 2 000 et 4 000 en robustesse (2 000 équivaut à 2,5 à 3 fois la capacité supposée).
- **Deux repères calculés avec chaque run**, au même K : **taille seule** (les K plus gros clients, sans modèle) et **oracle** (les vrais partants classés par V : le maximum atteignable avec K appels). On lit le modèle par son surplus sur la taille seule et par la part de l'oracle qu'il atteint ; le CA capté brut seul se défend mal, car la taille explique l'essentiel (lemme 4 de `03_evidence.md`, § 5).
- **Métriques sur les probabilités** : **PR-AUC** de `p_raw` contre `gone` (qualité du classement, comparable entre labels) et **Brier** de `p_raw` et de `p_cal` contre `label`, l'événement que le modèle prédit. Les Brier des labels A et B ne sont pas comparables entre eux (horizons différents) : les labels se comparent sur le CA capté. Elles servent à régler et comparer, jamais à décider ; par segment, ce sont les seules métriques calculées (un top K à l'intérieur d'un segment n'est pas la liste remise aux commerciaux).
- **Intervalles de confiance** : bootstrap sur les clients (200 tirages, les mêmes tirages pour toutes les métriques), 95 %. Un client tiré deux fois prend deux places dans le top K.
- **Hors du noyau, pour plus tard** : précision, rappel, F1 (au seuil choisi sur la validation, jamais sur le test), ROC-AUC, Brier skill score, ECE, precision@K et lift@K ; bootstrap **apparié** de la différence entre deux runs ; moyenne sur plusieurs photos de validation. La table de métriques est « longue » : une métrique de plus est une ligne de plus, sans rien casser.
- **Tests anti-fuite automatiques** avant chaque série d'expériences (`tests/test_no_leakage.py`).
- **Repères** : liste des K plus gros clients (sans modèle) et baseline récence ÷ p90 × V.
- **Tout le churn** : `capture_total_at_k` et ses deux repères divisent par le CA perdu de tous les clients, réguliers ou non (calculés avec `collect --universe`).
- **Comparaisons** : appariées date par date (`comparison.py`), une seule graine (0). La moyenne de plusieurs graines n'a pas été faite dans le code.
- **Marge sur le dataset** : `r × m × CA capté`, à r = 10 / 20 / 30 %, sans décote. **Impact entreprise** (`impact.py`) : `CA × part perdue × détection × r × m` ([05](05_impact_and_pilot.md)).
- **Pilote** : randomisé, deux listes séparées (total et partiel).

## 9. Comparaison des labels A et B

Plan glissant, réentraînement mensuel, 7 dates de test, vérité commune (`08_label_rolling_backtest.ipynb`, lemme 7 de `03_evidence.md`) : AUC identique (0,884) ; B capte +1,2 point de CA de plus à K = 700 et +0,8 point à K = 2 000, mieux que A sur 6 dates sur 7, pour +0,05 M€ de marge par an. Une partie de l'écart vient du nombre de photos (B en a plus à chaque date) et le bruit entre dates n'est pas dans l'IC : lecture prudente.

**Décision au gel : A pour le modèle final.** Sur le backtest du code (5 dates, K = 700), B moins A vaut +0,8 point avec HGB (4 dates gagnées) et −0,1 point avec XGBoost, le modèle retenu (2 dates) ; B ne passe pas le critère écrit dans `plan_gel.md` (+1,5 point). À K = 2 000, B reste devant (+0,4 et +1,3 point). B reste disponible (`label.kind`).

## 10. Protocole de validation et de réglage

En production, on apprend sur le passé pour prédire l'avenir : toute mesure de performance respecte l'ordre du temps. Deux boucles emboîtées, chacune pour ce qu'elle sait mesurer.

| Boucle | Découpage | Rôle |
|---|---|---|
| **Extérieure : plan glissant** | Par date : à chaque date `T_v`, entraînement sur les photos dont le label est connu avant `T_v`, évaluation sur `T_v` contre la vérité à 180 jours | Choisir : famille de modèle, label A ou B, jeu de variables. Seule source des chiffres de performance |
| **Intérieure : CV groupée par client** | 5 plis (`cv_folds`), toutes les photos d'un client dans le même pli | Régler et calibrer à l'intérieur de l'entraînement : prédictions hors-pli, calibrateur isotonique par segment |

**Déroulé**

1. **Plan glissant**, 6 dates (22/09/2025 au 22/02/2026 ; la première, à une seule photo d'entraînement, est hors moyenne). À chaque date : CV groupée sur les photos disponibles, calibrateur appris sur les prédictions hors-pli, réentraînement sur toutes ces photos, prédiction et évaluation de `T_v`. Hyperparamètres par défaut. Les clients de `T_v` ne servent jamais à entraîner ni à calibrer.
2. **Choix** de la famille, du label et des variables sur la moyenne des dates, comparaison appariée par date (une graine).
3. **Optuna, une seule fois**, sur le modèle retenu : CV groupée sur la fenêtre d'entraînement du test, test jamais utilisé. Hyperparamètres ensuite figés.
4. **Facultatif** : rejouer le plan glissant avec ces hyperparamètres pour mesurer le gain d'Optuna.
5. **Gel de la configuration**, puis **test du 22/03/2026, une seule fois**, avec la même procédure (CV, calibration, réentraînement).

**Règles**

- **Une seule fonction d'entraînement** (CV, calibration, réentraînement), appelée à chaque date du plan glissant et pour le modèle final : le backtest évalue exactement la procédure déployée.
- **Aucun chiffre de performance ne vient de la CV** (elle mélange les dates, donc optimiste) : seuls le plan glissant et le test sont présentés.
- **Pas de retour sur un choix après le test**, sinon le test devient un outil de réglage.
- **La date de test est exclue des dates de décision** du plan glissant.
- **Pas de CV à la place de l'évaluation d'une date** : retirer des clients de l'entraînement mesurerait une situation qui n'existe pas en production (le modèle a déjà vu l'historique des clients qu'il score). Le bootstrap sur les clients donne déjà le bruit d'échantillonnage.
- **Pas de validation temporelle à l'intérieur de l'entraînement** : le délai de label l'empêche. Avec le label A (180 jours), seule la photo de septembre pourrait être validée, avec une seule photo d'entraînement (mars). Le label B (≤ 90 jours) est moins touché.
- **Optuna hors de la boucle** : 30 à 50 essais × 5 plis × 7 dates représenterait plusieurs centaines à plus d'un millier d'entraînements (durée non mesurée). Optuna sert au modèle final, pas à comparer des approches.

Limites connues (test non vierge sur la période, réglages par défaut, dérive de calibration) : `07_risks_and_open_questions.md`.

## 11. Questions ouvertes

La liste complète est dans `07_risks_and_open_questions.md`. Les points qui touchent directement ces décisions :

- Rétention `r` et quota du partiel : seul un pilote les mesure.
- Une seule saison de test : le résultat est à confirmer sur une autre période.
- Fenêtre des variables : 180 jours partout dans le périmètre du modèle ; le lemme 1 donne aussi le découpage sur 365 jours à titre de comparaison.
- Calibration des probabilités d'un label court sur la vérité à 180 jours.
- Décote ρ : retirée du score pour l'instant ; à reconsidérer plus tard.
