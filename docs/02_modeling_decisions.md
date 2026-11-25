# Décisions de modélisation

> **Rôle** : **Source de vérité** des décisions de modélisation, état actuel uniquement. L'historique est dans `08_decision_log.md`, les preuves dans `03_evidence.md`.
> **Pour** : L'équipe ; à citer plutôt que recopier.
> **À jour au** : 25/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Objectif et cadrage

- Maximiser la marge `M = r × m × Σ CA perdu` sur une liste de K clients contactés ; `m = 24,65 %`.
- La rétention `r` n'est pas mesurable : scénarios 10 / 20 / 30 %, mesure par pilote.
- Le score sert à classer des clients, pas à prédire le CA futur : `V` est le CA que le client générerait **s'il restait**, pas une prévision du CA réel (qui donnerait ≈ 0 pour ceux qui partent et compterait le départ deux fois).

## 2. Périmètre et définitions

| Élément | Décision |
|---|---|
| Population | Clients réguliers : ≥ 6 jours d'achat sur les 180 jours avant la photo, écart médian ≤ 30 jours (rapides ≤ 7 j, moyens 8 à 30 j). 51 % des clients, 95 % du CA, 52 % du CA du churn total |
| Hors modèle | Lents (≈ 0,1 % du CA) ; historique court (< 6 achats) en baseline seule |
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

- **Total = label principal et chiffre de tête** : AUC 0,89, ≈ 65 % du CA perdu capté à K = 2 000.
- **Partiel brut = second score**, à confirmer en pilote, hors chiffre de tête (CA en jeu plus grand, AUC 0,68).
- **Liste mixte** : environ 25 % des places au total (10 à 25 % acceptables). Deux scores, `P(total) × V` et `P(partiel) × V` ; les deux listes se recouvrent peu (16 % à K = 2 000). Le total capte 40 à 45 % de son CA en jeu avec 25 % des places contre 56 à 61 % avec toute la liste : on annonce les deux chiffres.
- **Écartés** : partiel durable (360 jours de recul, une seule photo), label élargi (confondu avec le partiel), interruption (≈ 11 M€, simple variable).

## 4. Labels et horizon

- **Vérité d'évaluation** : « parti » = aucun jour d'achat pendant les **180 jours** suivants, par convention (horizons courts exclus : à 60 jours, 37 % des rapides et 57 % des moyens reviennent ; pas de plateau ensuite).
- **Deux labels d'entraînement, gardés et comparés** en plan glissant (`08_label_rolling_backtest.ipynb`).
  - **A** : 180 jours pour tous (même définition que la vérité). 7 photos d'entraînement.
  - **B** : relatif au rythme, `H = clip(3 × écart p90, 45, 90 jours)`. 10 photos. Pureté du label 60 %.
  - Les deux sont dans la configuration (`label.kind` : `fixed_horizon`, `relative_horizon`) ; A est la référence, B l'alternative (section 9).
- **Toute évaluation en CA se fait contre la vérité à 180 jours**, jamais contre le label court (qui surestime le CA perdu d'un facteur 1,7).
- **Décote du CA du churn total : 12 %** (part du CA qui revient dans les 90 jours), valeur gardée pour l'instant dans les marges. Une mesure directe de la perte réalisée sur 360 jours donne ≈ 2 % (`07_realized_revenue_loss.ipynb`) : à intégrer plus tard.

## 5. Découpage et cadence

- **Photos mensuelles** : variables calculées avant la date, label après. Les 6 premiers mois ne servent qu'aux variables.
- **Entraînement** : photos du 22/03/2025 au 22/09/2025 (7 photos, label A) ou jusqu'au 22/12/2025 (10 photos, label B) ; `train_end` est dérivé du label dans la configuration. **Test** : 22/03/2026, jamais touché pendant le développement.
- **Règle** : le label le plus long d'une photo d'entraînement doit être connu avant la date de test.
- **Cadence** : entraînement mensuel, scoring hebdomadaire (hypothèse ; photos hebdomadaires = 3,8 fois plus de lignes pour la même qualité). Plan glissant pour la valider.
- **Reproductibilité** : lignes triées par client, un seul thread.

## 6. Modèle et score

- **Un seul modèle** avec le segment rapide / moyen en variable (pas de modèle par segment, sauf si le résultat déçoit).
- **Modèles** : baseline (récence ÷ écart p90), régression logistique, HistGradientBoosting, XGBoost ; probabilités calibrées (isotonique). Optuna (30 à 50 essais) une fois le meilleur choisi. Pas de réseau de neurones.
- **Variables** : 12 à 13, calculées sur 180 jours (fréquence, CA, tendances, écart médian et p90, récence et récence ÷ p90). Pas de variable « mois ».
- **Score** : valeur espérée `P(départ) × V`, avec raisons principales ; classement par `p × V` (+15 points de CA capté face au classement par probabilité seule). **La décote ρ est retirée pour l'instant** (elle pourra être remise plus tard).

## 7. Liste et capacité

- **Hypothèse centrale** : 500 appels par semaine (100 commerciaux × 5 appels de 24 minutes), fourchette 200 à 1 800.
- **À l'échelle du dataset** (≈ 30 % du CA) : K ≈ 700 pour une liste mensuelle (600 à 760 appels par mois) ; K ≈ 4 000 pour six listes cumulées sans rappel.
- **Répartition de K entre les deux listes** : 25 % des places au total ne perd jamais plus de 0,1 M€ face à la meilleure répartition (≈ 175 places à K = 700, ≈ 1 000 à K = 4 000) ; 0 % n'est jamais la meilleure et 100 % est la pire. Rendements décroissants : doubler les appels n'ajoute que +15 % (liste mensuelle) à +33 % (liste cumulée) de marge.

## 8. Évaluation et marge

- **Mesure principale** : CA des vrais partants capté dans les K premiers, divisé par le CA de tous les vrais partants (au sens de la vérité à 180 jours), toujours comparée au repère « taille seule ».
- **Métriques sur les probabilités** (précision, rappel, F1, PR-AUC, ROC-AUC, Brier) : pour régler et comparer les modèles, jamais pour décider ; seuils choisis sur la validation, **jamais sur le test** ; par segment, avec IC par bootstrap sur les clients.
- **Tests anti-fuite automatiques** avant chaque série d'expériences (`tests/test_no_leakage.py`).
- **Repères** : liste des K plus gros clients (sans modèle) et baseline récence ÷ p90 × V.
- **Bootstrap** sur les clients (200 tirages), moyenne de plusieurs graines pour comparer des règles.
- **Marge** : `r × m × CA capté`, à r = 10 / 20 / 30 % ; réserves saison, persistance et rétention du partiel.
- **Pilote** : randomisé, deux listes séparées (total et partiel).

## 9. Comparaison des labels A et B

Plan glissant, réentraînement mensuel, 7 dates de test, vérité commune (`08_label_rolling_backtest.ipynb`, lemme 7 de `03_evidence.md`) : AUC identique (0,884) ; B capte +1,2 point de CA de plus à K = 700 et +0,8 point à K = 2 000, mieux que A sur 6 dates sur 7, pour +0,05 M€ de marge par an. Une partie de l'écart vient du nombre de photos (B en a plus à chaque date) et le bruit entre dates n'est pas dans l'IC : lecture prudente.

**Décision : A et B sont gardés.** A reste la référence (une seule définition de la perte, entraînement et vérité identiques) ; B est l'alternative qui capte un peu plus de CA. Les deux sont lancés et comparés dans les sweeps (`label.kind`).

## 10. Questions ouvertes

La liste complète est dans `07_risks_and_open_questions.md`. Les points qui touchent directement ces décisions :

- Rétention `r` et quota du partiel : seul un pilote les mesure.
- Une seule saison de test : le résultat est à confirmer sur une autre période.
- Fenêtre des variables : 180 jours partout dans le périmètre du modèle ; le lemme 1 donne aussi le découpage sur 365 jours à titre de comparaison.
- Calibration des probabilités d'un label court sur la vérité à 180 jours.
- Décote ρ : retirée du score pour l'instant ; à reconsidérer plus tard.
