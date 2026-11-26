# Journal des décisions

> **Rôle** : L'historique daté : ce qui a été décidé, repris ou corrigé, et pourquoi. L'état actuel est dans `02_modeling_decisions.md`.
> **Pour** : Quiconque veut comprendre comment on est arrivé aux décisions actuelles.
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## Gel du 25-26/11

| Sujet | Décision | Pourquoi |
|---|---|---|
| Tendances | `freq_trend` et `rev_trend` en log-ratios | Un rapport simple explosait quand la période précédente était proche de zéro |
| Modèles | BG/NBD ajouté comme repère statistique | Comparer à un modèle non supervisé |
| Variables | **`basic`** ; `extended` et `market_only` rejetés | Gain inférieur au critère écrit avant les résultats ([03](03_evidence.md), § 11) |
| Label | **A** pour le modèle final | B ne passe pas le critère (−0,1 point avec XGBoost à K = 700) |
| Modèle | **XGBoost**, réglé par une étude Optuna de 50 essais | 49,9 % contre 48,6 % pour HGB au backtest (K = 700) |
| Liste | Score total seul, 700 clients ; **pas de quota 25 %** | Périmètre réduit pour tenir l'échéance (`plan_gel.md`) |
| Métriques et impact | CA capté sur tout le churn ; impact et tornade en code | Détection utilisable pour l'impact entreprise |
| Tableau de bord | Page HTML locale au lieu de Streamlit | Rejeu des listes sans serveur |
| Test final | Lancé le 26/11 à 15 h 12 | 51,5 % à K = 700 |

## Décisions du 25/11

| Sujet | Décision | Pourquoi |
|---|---|---|
| Métriques d'évaluation | **Noyau réduit** : CA capté @K (K = 700 en tête, 2 000 et 4 000 en robustesse) avec les repères « taille seule » et « oracle », PR-AUC, Brier (brut et calibré, contre `label`), IC par bootstrap sur les clients. Précision, rappel, F1, ROC-AUC, Brier skill score, ECE, precision@K, lift@K, bootstrap apparié : repoussés | À la veille du gel du code, le noyau suffit à choisir et à défendre un modèle ; le chiffre de tête doit se lire contre la taille seule (près des trois quarts du CA capté du partiel) ; le Brier ne se compare pas entre A et B (événements différents). Format long : le reste s'ajoute sans refonte |
| Labels d'entraînement | **A et B gardés et comparés** : A = 180 j pour tous (référence), B = `clip(3 × p90, 45, 90 j)` ; les deux dans la configuration (`label.kind`) | B capte +1,2 pt de CA de plus à K = 700 en plan glissant, un petit écart ; A garde une seule définition de la perte |
| Périmètre des réguliers | Chiffres recalculés sur le rythme à **180 j** : 51 % des clients, 95 % du CA, 52 % du CA du churn total (et non 64 %, 97 %, 62 % sur 365 j) | Le périmètre utilisé par les modèles est défini sur 180 j |
| Capacité et K | **500 appels par semaine** conservés ; **K ≈ 700** par liste mensuelle (et non 1 000) ; K ≈ 4 000 en cumulé | 600 à 760 appels par mois donnent K ≈ 700 ; les tests à K = 1 000 restent valables |
| Décote ρ | **Retirée du score** pour l'instant ; score `P × V` | Pas encore mesurée ni justifiée ; pourra revenir |
| Décote du CA du churn total | **12 % laissée** dans les marges ; mesure directe à ≈ 2 % (`07_realized_revenue_loss`) à intégrer plus tard | Question jugée secondaire face à la rétention |
| Validation et réglage | **Deux boucles emboîtées** : plan glissant (temps) pour choisir ; à chaque date, CV groupée par client (5 plis) pour calibrer ; Optuna une seule fois sur le modèle retenu ; test une seule fois, exclu des dates de décision | Seule une évaluation ordonnée dans le temps imite la production ; la CV par client seule mélange les dates (optimiste) ; une validation temporelle dans l'entraînement est impossible avec le label A (délai de 180 jours) |
| Segment `frequency` | **Classe de rythme** : rapide (écart médian ≤ 7 j) / moyen (8 à 30 j) ; lent et historique court hors périmètre. Une photo garde tous les clients actifs, le périmètre se filtre avec `in_perimeter` | Cohérent avec « segment rapide / moyen en variable » (`02_modeling_decisions.md`) ; évite des tranches de `freq_180d` arbitraires qui recoupent la classe de rythme |
| Notebooks | Exploration découpée en `01` à `05`, ajout de `06` à `08`, suppression des notebooks `exploratory_…` ; noms en anglais | Clarté, une question par notebook |
| Code et configuration | `label.kind` : `fixed_horizon` / `relative_horizon` / `partial_drop` ; `train_end` dérivé du label ; capacité en `calls_per_week` ; suppression de ρ, des paliers et des horizons par classe | Aligner le squelette sur le plan |
| Documentation | Restructurée en 9 documents numérotés plus un glossaire (une source de vérité par type d'information) | Les décisions étaient recopiées à quatre endroits et divergeaient |

## Décisions du 24/11

| Sujet | Avant (18/11) | Maintenant (24/11) | Pourquoi (notebook) |
|---|---|---|---|
| Périmètre du modèle | Un seul modèle sur toute la base, segment de fréquence en variable | Modèle limité aux **clients réguliers** (≥ 6 jours d'achat sur 180 j, écart médian ≤ 30 j : rapides + moyens). Lents hors modèle ; historique court en baseline seule | Sur 180 j : rapide + moyen = 51 % des clients, 95 % du CA, 52 % du CA du churn total ; lent ≈ 0,1 % du CA ; l'historique court pèse 47 % du churn total (notebook `04`) |
| Horizon de la perte | 30 j rapides, 60 j moyens, 120 j lents, 180 j historique court | **Vérité d'évaluation : 180 j pour tous les réguliers, par convention.** **Deux labels d'entraînement gardés et comparés : A = 180 j pour tous (7 photos) ; B = H\_i = clip(3 × écart p90, 45, 90 j) (relatif au rythme, 10 photos)** ; l'alerte précoce vient aussi de variables relatives au rythme | À 60 j, 37 % des rapides et 57 % des moyens « perdus » reviennent dans les 90 j suivants ; à 180 j, encore 13 % et 27 % (12 % du CA du churn total) : pas de plateau, donc pas de seuil naturel (notebook `04`, recalculé) |
| Jour d'achat | Ligne de vente strictement positive | CA net des retours du jour **> 0,01 €**, CA = ce net | Un achat retourné le jour même n'est pas un achat ; le seuil évite un bruit de ±1e-13 (notebook `04`) |
| Label principal et chiffre de tête | Churn total ; chute durable en variante si le temps le permet | **Churn total** = label principal et chiffre de tête. Le **churn partiel brut** devient un second modèle, présenté comme potentiel à confirmer par le pilote, hors chiffre de tête | AUC 0,89 contre 0,68 ; marge sans dominance : tout dépend de la rétention (notebook `05`) |
| Label « chute durable » | Variante prévue | **Non retenu comme label de modèle** ; sert seulement au dimensionnement | Il exige 360 j de recul : une seule photo utilisable, aucune validation temporelle (notebook `04`) |
| Fusion des deux labels | — | **Pas de label élargi** : deux scores et une répartition de K | Le label élargi se confond avec le partiel et dilue le total (notebook `05`) |
| Capacité | Paliers : 20 % de commerciaux dédiés à 10 actions par semaine, 80 % à 5 (≈ 1 200 par semaine) | Hypothèse centrale **500 appels par semaine** (100 commerciaux × 5 appels, 2 h chacun), fourchette 200 à 1 800, scénario à 1 000 testé ; convertie à l'échelle du dataset : **K ≈ 700 (liste mensuelle, 600 à 760 appels) et K ≈ 4 000 (six listes cumulées)** ; les tests de répartition ont été faits à K = 1 000 et 4 000 (même conclusion) | Aucun chiffre dans les documents ; le dataset ≈ 30 % du CA de l'entreprise (notebook `05`) |
| Répartition de K | Non prévue | **≈ 25 % des places au churn total**, le reste au partiel (10 à 25 % selon les cas) | Meilleure ou quasi meilleure dans les 16 scénarios testés (notebook `05`) |
| Variable « mois » | « Indicateur du mois » prévu | **Pas de variable mois** | Un seul recoupement annuel : le modèle la mémoriserait (notebook `04`) |
| Évaluation | CA perdu capté dans les K premiers | + repère **« taille seule »** (liste des K plus gros clients, sans modèle), bootstrap sur les clients, marge corrigée des trois réserves (persistance, saison, rétention) | Près des trois quarts du CA capté par le modèle partiel viennent de la seule taille (notebook `05`) |
| Pilote | Randomisation par client | **Deux listes randomisées séparément** (total et partiel) | Seul moyen de mesurer la rétention de chacune |

## Du label de départ aux labels retenus

1. **Le problème (18/11).** Maximiser M(L) = r · m · Σ ℓᵢ sur une liste de K clients ; ℓᵢ est connu après coup, r ne se mesure pas. Le document de décisions partait d'un horizon par classe de rythme (30 j rapides, 60 j moyens, 120 j lents, 180 j historique court, « à calibrer »), avec en variante un horizon relatif k × p90 et une chute durable de CA.
2. **Quels clients (L1).** Les réguliers (rythme sur 180 j) : 51 % des clients, 95 % du CA, 52 % du churn total. L'historique court (47 % du churn total) n'a pas de variables de rythme ; le segment lent ne pèse rien.
3. **Quel horizon (L2).** Les horizons de départ sont trop courts (à 60 j, 37 % des rapides et 57 % des moyens « perdus » reviennent dans les 90 j) et il n'y a pas de plateau ensuite. La vérité est **180 j pour tous, par convention** : comparable en CA entre segments, avec un coût mesuré (≈ 12 % du CA du total revient dans les 3 mois).
4. **Quel type de perte (L3 à L6).** Le total est le label principal et le chiffre de tête (AUC 0,89 contre 0,68, validable dans le temps). Le partiel est un second score, à confirmer en pilote, avec ≈ 25 % des places. Le « durable » (360 j de recul, une seule photo) et le label élargi (confondu avec le partiel) sont écartés.
5. **Quel label d'entraînement (L7).** 180 j ne laisse que 7 photos. Le label relatif k × p90 (B : k = 3, h_min = 45, h_max = 90) en laisse 10 sans dégrader le CA capté (+1,2 pt à K = 700 en plan glissant). **A et B sont gardés** ; la vérité d'évaluation reste à 180 j.

## Corrections apportées

| Affirmation initiale | Correction après lecture du notebook |
| --- | --- |
| « 4 à 7,5 % du CA inactif » (clients inactifs 60 à 180 jours) | C'était un stock cumulé. En flux annuel : 3,4 à 4,8 % |
| « 8 % de CA perdu : à vérifier » | Mesuré sous cette valeur ; le scénario bas (5 %) est le plus réaliste |
| « La récence relative est une bonne variable » | Vrai chez les réguliers seulement |
| Dates d'origine du dataset | Fichier nommé 2026 mais commandes de 2022 à 2024 : dates fictives, décalage de +2 ans retenu |
| « Une ligne positive est un achat » | Un achat intégralement retourné le jour même n'est pas un achat : jour d'achat = CA net > 0,01 €. Base de 119 656 à 118 009 clients, CA de référence −6 % |
| « Horizon de 30 j pour les rapides, 60 j pour les moyens » | À 60 j, 37 % des rapides et 57 % des moyens « perdus » reviennent dans les 90 j suivants : horizons trop courts. 180 j est retenu par convention (13 % et 27 % reviennent encore, 12 % du CA du churn total), pas parce que les données imposent ce seuil |
| « Le label relatif k = 3 est nettement meilleur que 180 j » (premier passage de la calibration) | C'était du bruit d'entraînement : le même modèle variait de 1 à 3 points d'une exécution à l'autre (ordre des clients renvoyé par DuckDB, multithreading). Après correction : +1 à +3 points de CA capté, du même ordre que le bruit (0,3 à 2 pts), donc non démontré ; ce qui tient : pas de dégradation, 10 photos au lieu de 7, pureté de 60 % |
| « Le partiel est très saisonnier, le total peu » | Amplitudes voisines (±15 % et ±12 %) ; le défaut du partiel est le biais A contre B (rapport CA(B)/CA(A) de 0,88 à 1,07) |
| « L'avantage du partiel vient pour moitié de la taille des clients » | Près des trois quarts : 113 M€ sur 156 M€ captés viennent de la seule taille |
| « Le partiel est plus difficile à prédire, donc à écarter » | Vrai statistiquement (AUC 0,68 contre 0,89) mais son CA en jeu est 8 fois plus grand : il reste un second modèle crédible |
| **25/11** « 600 à 760 appels par mois, soit K ≈ 1 000 » | Incohérent : 600 à 760 appels donnent K ≈ 700. Les tests de répartition à K = 1 000 sont gardés comme sensibilité |
| **25/11** « Les réguliers sont 64 % des clients, 97 % du CA, 62 % du churn total » | Vrai avec le rythme sur 365 j ; avec le périmètre réellement utilisé (180 j) : 51 %, 95 %, 52 % |
| **25/11** « La décote de 12 % corrige le CA du churn total » | La mesure de la perte réalisée sur 360 j donne ≈ 2 % ; à intégrer plus tard |
| **26/11** « Les marges incluent la décote de 12 % » | Aucun calcul ne l'applique |
| **26/11** « Doubler les appels : +15 à +33 % » | +15 % correspond à une liste 40 % plus longue ; seul +33 % est un doublement |
| **26/11** « Impact de 13 à 18 M€ » | Détection supposée de 50 % ; avec 26,1 % mesurés, 8,2 M€ au central |
