# Preuves : churn total ou partiel, horizon, labels

> **Rôle** : Les preuves chiffrées derrière chaque décision, sous forme de lemmes, chacun relié à son notebook.
> **Pour** : Quiconque doit défendre un choix devant le jury ou l'équipe.
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

Sources : `notebooks/04_churn_scope_total_vs_partial` (dont le recalcul des horizons du 24/11), `05_label_test_and_list_allocation`, `06_label_horizon_calibration`, `07_realized_revenue_loss` et `08_label_rolling_backtest`. Montants en M€ annualisés, **périmètre du dataset** (≈ 4,6 à 4,8 Md€ par an, soit ≈ 30 % des 15,5 Md€ de l'entreprise), non extrapolés.

## 1. Énoncé

À chaque cycle, on contacte une liste L d'au plus K clients, de façon à maximiser

**M(L) = r · m · Σ ℓᵢ** (i ∈ L), où ℓᵢ est le CA annualisé réellement perdu par le client i, r la rétention et m = 24,65 % la marge brute.

ℓᵢ n'est connu qu'après coup et r n'est pas mesurable (aucun journal d'actions) : r = 20 %, en scénarios 10 / 20 / 30 %. Pour le total, ℓᵢ est approximé par CA(A) × 2 : c'est un majorant, car ≈ 12 % de ce CA revient dans les 3 mois suivants (lemme 2). Trois choix à justifier : quels clients (D1), quelle perte (D2), quel horizon (D3).

*Définitions.* À la date T : A = les 180 j avant, B = les 180 j après, C = les 180 j suivants. **Total** : aucun achat dans B. **Partiel** : achat dans B mais CA(B) < 50 % de CA(A). **Durable** : partiel et CA(C) < 50 % de CA(A). Perte du total : CA(A) × 2 ; du partiel : (CA(A) − CA(B)) × 2. Un jour d'achat a un CA net de retours > 0,01 €.

## 2. Lemme 1 : le CA est chez les clients réguliers (D1)

**Rythme sur 365 j** (exploration d'origine) :

| Segment (rythme sur 365 j) | Clients | CA | Churn total |
|---|---|---|---|
| Rapide (écart médian ≤ 7 j) | 35,8 % | 88,3 % | 57 M€ (46 %) |
| Moyen (8 à 30 j) | 28,1 % | 8,5 % | 20 M€ (16 %) |
| Lent (> 30 j) | 5,3 % | 0,7 % | 3 M€ (2 %) |
| Historique court (< 6 achats) | 30,8 % | 2,6 % | 45 M€ (36 %) |

**Rythme sur 180 j** (périmètre utilisé par les modèles, notebook `04`, même photo et même base de 118 009 clients) :

| Segment (rythme sur 180 j) | Clients | CA | Churn total |
|---|---|---|---|
| Rapide (écart médian ≤ 7 j) | 34,2 % | 88,3 % | 54 M€ (43 %) |
| Moyen (8 à 30 j) | 16,9 % | 6,9 % | 12 M€ (9 %) |
| Lent (> 30 j) | 0,3 % | 0,1 % | 0,1 M€ |
| Historique court (< 6 jours d'achat) | 48,6 % | 4,8 % | 59 M€ (47 %) |
| **Réguliers (rapides + moyens)** | **51,1 %** | **95,1 %** | **65 M€ (52 %)** |

*Preuve.* Lecture directe (notebook `04`, photo du 22/09/2023, 118 009 clients). Avec le rythme sur 365 j, les réguliers sont 64 % des clients, 97 % du CA et 62 % du churn total ; sur 180 j, la règle « au moins 6 jours d'achat » écarte davantage de petits clients : **51 % des clients, 95 % du CA, 52 % du churn total**. Le segment lent ne pèse rien ; l'historique court (47 % du churn total sur 180 j) n'a, sous 6 achats, ni écart p90 ni récence relative : au mieux une baseline. Le périmètre du modèle couvre donc environ la moitié du CA du churn total mais 95 % du CA à protéger. **Donc** périmètre = clients réguliers (≥ 6 jours d'achat sur 180 j, écart médian ≤ 30 j). ∎

## 3. Lemme 2 : 180 jours est une convention défendable, pas un seuil que les données imposent (D3)

*Correction du 24/11.* La première version mesurait la part des clients « sans achat pendant H jours » qui rachètent avant 180 j. Ce calcul était circulaire : la fenêtre de retour se terminait à 180 j, donc 180 j sortait à 0 % par construction, et la fenêtre rétrécissait quand H augmentait. Les chiffres ci-dessous utilisent une **fenêtre de retour fixe de 90 j**.

Part des clients « sans achat pendant H jours après T » qui rachètent dans les 90 jours suivants (clients réguliers, photos mensuelles poolées, de 2 à 8 selon H, car la fenêtre doit tenir dans les données) :

| H | 60 j | 90 j | 120 j | 180 j | 240 j |
|---|---|---|---|---|---|
| Rapides, % des clients | 37 % | 26 % | 20 % | 13 % | 10 % |
| Rapides, % du CA | 29 % | 18 % | 13 % | 8 % | 5 % |
| Moyens, % des clients | 57 % | 47 % | 38 % | 27 % | 21 % |
| Moyens, % du CA | 60 % | 47 % | 36 % | 24 % | 18 % |

Ce que deviennent les clients du churn total (aucun achat pendant 180 j, rapides + moyens) :

| Reviennent… | Rapides | Moyens | CA rapides + moyens |
|---|---|---|---|
| dans les 90 j suivants (4 photos, 22/09 au 22/12/2023) | 13 % | 27 % | 12 % (≈ 11 M€ sur 88 M€ par photo) |
| dans les 180 j suivants (1 photo, 22/09/2023) | 19 % | 37 % | 17 % (≈ 13 M€ sur 76 M€) |

Les clients qui reviennent le font tard : délai médian de 249 j après T, soit environ 70 j après la fin de la fenêtre (photo du 22/09/2023).

*Mesure.* Trois constats, dont un seul soutient 180 j.
1. **Les horizons courts sont exclus.** À 60 j, 37 % des rapides et 57 % des moyens déclarés perdus rachètent dans les 90 j suivants (à 30 j : 56 % et 69 %). Les valeurs de départ du document (30 j, 60 j) déclarent perdus des clients qui reviennent seuls.
2. **Il n'y a pas de plateau.** La part qui revient baisse lentement avec H (rapides 20 → 13 → 10 %, moyens 38 → 27 → 21 % de 120 à 240 j) et ne tombe jamais à un niveau négligeable. Aucun seuil naturel n'apparaît dans ces données, qui couvrent 2 ans : à 240 j, il ne reste que 2 photos.
3. **Les segments ne sont pas équivalents au même horizon.** À 180 j, les moyens reviennent deux fois plus que les rapides (27 % contre 13 % des clients ; 24 % contre 8 % du CA). La condition « ≤ 25 % de retours dans les 90 j suivants » est atteinte vers 90 j pour les rapides (26 %) et seulement entre 180 et 240 j pour les moyens (27 % puis 21 %) : 180 j est au plus près de l'horizon du segment le plus lent, avec un seuil de 25 % qui reste un choix.

En multiples de l'écart p90 du client, les deux segments se rapprochent au-delà de 4 fois (retour sous 90 j : 55 % contre 46 % entre 4 et 6 fois ; 35 % contre 37 % entre 6 et 10 fois) mais pas en dessous (81 % contre 65 % entre 2 et 3 fois). En jours depuis le dernier achat, l'écart entre segments est de 13 à 20 points de 60 à 180 j. Le label relatif ne règle donc pas la question à lui seul.

**Donc** : la perte est **définie à 180 j par convention** pour tous les réguliers (commune aux segments, donc comparable en CA ; plus court est faux, plus long est à peine testable avec 2 ans de données). Ce choix a un coût mesuré : le churn total est « aucun achat en 6 mois », pas « perte définitive ». Au moins **12 % de son CA revient dans les 3 mois suivants (17 % en 6 mois)**, 8 % pour les rapides et 24 % pour les moyens. Les M€ du churn total sont des majorants de cet ordre. L'avance d'alerte vient de variables relatives au rythme du client (récence ÷ écart p90, tendance), pas du label. Le label relatif a ensuite été testé (lemme 7) et retenu comme label d'entraînement ; la vérité d'évaluation reste à 180 j.

*Traçabilité (26/11).* La cellule du notebook `04` qui produit ces chiffres n'a pas de sortie enregistrée ; recalculée le 26/11, elle confirme les retours par horizon, la décote de 12 % et le délai de 249 jours. Les chiffres en multiples de l'écart p90 et la variation saisonnière de 34 à 55 % n'ont été retrouvés dans aucune cellule : ne pas les citer.

*Limites.* Les photos poolées se chevauchent (mêmes clients d'un mois à l'autre) : les pourcentages sont stables d'une photo à l'autre (à 180 j, 10 à 14 % du CA revient), mais les effectifs ne sont pas indépendants. Les retours dépendent de la saison (à 60 j, la part du CA qui revient va de 34 % pour la photo d'octobre à 55 % pour celle de mars). La décote de 12 à 17 % est mesurée sur toute la population régulière, pas sur la liste de K clients.

## 4. Lemme 3 : le plafond du partiel est haut, mais ce n'est pas du CA sauvable (D2)

| CA en jeu (notebook `04`) | M€ | % du CA |
|---|---|---|
| Churn total | 125 | 2,8 % |
| Partiel brut / durable | 509 / 357 | 11,5 % / 8,1 % |
| Contre-poids : hausses > 50 % | 826 | 18,7 % |

*Preuve.* (i) 357 ≈ 2,9 × 125 : l'attrait du partiel. (ii) Seuls 66 à 72 % de son CA dure (7 photos). (iii) Les hausses pèsent plus que les baisses : le CA de ces clients oscille d'un semestre à l'autre, donc le label est bruité. (iv) Le rapport CA(B)/CA(A) de la base varie de 0,88 à 1,07 selon la photo : le partiel compare deux saisons (amplitude voisine pour les deux labels : total 2,5 à 3,2 %, partiel 10,3 à 14,0 %). **Donc** 357 M€ est un majorant, pas un gisement. ∎

*Réserve (lemme 2).* Les 125 M€ du total sont eux aussi un majorant : ≈ 12 % du CA des clients réguliers déclarés perdus à 180 j revient dans les 3 mois suivants (8 % pour les rapides, 24 % pour les moyens). Cette décote ne change pas l'ordre de grandeur du rapport entre les deux labels (357 / 125 ≈ 2,9 contre ≈ 3,2 après décote du total seul) ; elle est mesurée sur les réguliers, pas sur les lents ni l'historique court.

## 5. Lemme 4 : le total est bien plus prévisible

Même modèle (HistGradientBoosting, 13 variables), même découpage temporel (180 j entre la fin de l'entraînement et le test) ; test au 22/03/2024 sur 61 286 clients réguliers.

| | Total | Partiel |
|---|---|---|
| Événements (taux de base) | 2 226 (3,6 %) | 14 420 (23,5 %) |
| ROC-AUC | 0,89 | 0,68 |
| Précision de la liste de 2 000 | 35 % | 28 % |
| CA perdu capté, liste de 2 000 [IC 95 %] | 65,7 % [61–71] | 30,9 % [28–33] |
| … par la liste des 2 000 plus gros clients | 14,5 % | 22,3 % |

*Preuve.* Le modèle total multiplie le taux de base par ≈ 10, le partiel par ≈ 1,2. Face à la liste « plus gros clients », sans modèle, le surplus du modèle est de **31 M€ (total) contre 43 M€ (partiel)** : près des trois quarts des 156 M€ captés par le modèle partiel (113 M€) viennent de la seule taille des clients. ∎

## 6. Lemme 5 : aucun label ne domine en marge, tout dépend de r

Marge (M€/an, K = 2 000 ; entre parenthèses, scénarios bas–haut sur persistance, saison et rétention) :

| Lecture | Total | Partiel | r_p / r_t d'équilibre |
|---|---|---|---|
| CA capté (aucune action aujourd'hui) | 2,0 (0,9–4,2) | 5,4 (2,0–10,1) | 37 % |
| Surplus vs plus gros clients (déjà suivis) | 1,5 (0,7–3,3) | 1,5 (0,6–2,8) | ≈ 100 % |

*Preuve.* M est linéaire en r : M_partiel ≥ M_total ⇔ r_p / r_t ≥ ρ*, avec ρ* = 37 % ou ≈ 100 % selon la lecture. Les interviews ne donnent aucun taux de rétention ; elles disent que les commerciaux privilégient leurs bons clients et laissent partir les autres, sans repérage systématique des baisses (Sales Manager, Head of Sales) : la pratique est entre les deux lectures. **Donc** ni r_p ≥ 0,37 r_t ni r_p < r_t ne se démontrent ici ; seul un pilote randomisé tranchera. ∎

*Réserve (lemme 2).* Les marges du total reposent sur un CA perdu majoré d'environ 12 % (retours après 180 j). À CA capté égal, le seuil ρ* de la première lecture passe d'environ 37 % à environ 33 % : l'ordre de grandeur et la conclusion ne changent pas. Ordre de grandeur seulement : la décote est mesurée sur toute la population régulière, pas sur la liste de K clients, et le partiel a sa propre part de retours (lemme 3).

## 7. Lemme 6 : les deux labels ne se fusionnent pas

| Liste de 2 000 | CA du total capté | CA du partiel capté | Total + partiel | Recouvrement avec « total » |
|---|---|---|---|---|
| Total seul | 66 % | 12 % | 102 M€ | 100 % |
| Partiel seul | 26 % | 31 % | 172 M€ | 16 % |
| Label élargi (perte ≥ 50 %, départ compris) | 45 % | 30 % | 180 M€ | 26 % |
| Mixte : 500 places total + 1 500 partiel | 51 % | 29 % | 179 M€ | — |

*Preuve.* Le partiel pèse 89 % du CA en jeu de l'union : le label élargi se confond avec lui (recouvrement 85 à 90 %) et dilue le total (45 % contre 66 %). Une liste mixte égale son CA capté en gardant 51 % du total. Écarts de quelques M€ : dans le bruit. ∎

## 8. Capacité (hypothèse de cadrage, à confirmer : question 3 du tutorat) et répartition de K

100 commerciaux (la moitié de 200) × 5 appels par semaine (2 h, soit 24 min par appel, préparation et saisie comprises) = 500 par semaine, soit **2 000 à 2 170 appels par mois**, plus prudent que la configuration de départ (≈ 1 200 par semaine). Deux conversions :

- **Échelle.** Ces appels couvrent l'entreprise ; le dataset en représente ≈ 30 % du CA (≤ 38 % des clients) : **≈ 600 à 760 appels par mois** pour ses clients, soit **K ≈ 700** pour une liste mensuelle (les tests de répartition ci-dessous ont été faits à K = 1 000, arrondi haut ; même conclusion).
- **Cadence.** Sur 180 j, six listes mensuelles sans rappel s'accumulent : **≈ 3 600 à 4 560 clients distincts**, soit **K ≈ 4 000**. (Le K = 2 000 des tableaux précédents équivaudrait à 5 300 à 6 700 appels par mois pour l'entreprise : 2,5 à 3 fois la capacité supposée.)

**Résultat** (notebook `05` ; bootstrap sur 200 tirages ; 4 scénarios : lecture « CA capté » ou « surplus », rétention du partiel égale à celle du total ou moitié moindre). Dans les 8 cas, la meilleure répartition réserve **10 à 25 % des places au total** ; 0 % n'est jamais la meilleure et 100 % est la pire. Le gain est faible à rétention égale (+3 %) mais atteint +10 à +40 % dans les scénarios prudents : le quota assure contre la rétention incertaine du partiel. **25 % ne perd jamais plus de 0,1 M€ face à la meilleure répartition : ≈ 250 places à K = 1 000, ≈ 1 000 à K = 4 000.** Coût pour le chiffre de tête : le total capte alors 40 % (K = 1 000) ou 63 % (K = 4 000) de son CA en jeu, contre 56 % et 74 % avec toute la liste. Une seule photo de test : le bootstrap mesure le bruit d'échantillonnage, pas la variation d'une saison à l'autre.

**Scénario à 1 000 appels par semaine** (K ≈ 1 400 mensuel, K ≈ 8 000 cumulé) : même conclusion (10 à 25 % des places au total). Rendements décroissants, en lecture « CA capté » à rétention égale : la liste cumulée doublée (K 4 000 → 8 000) donne +33 % de marge ; la liste mensuelle comparée de K 1 000 à 1 400, soit +40 % de places et non le double, donne +15 %. Presque aucun gain en lecture « surplus ».

## 9. Lemme 7 : un label d'entraînement relatif au rythme, évalué contre la vérité à 180 j

Source : `notebooks/06_label_horizon_calibration.ipynb` (calibration), `08_label_rolling_backtest.ipynb` (comparaison A / B en plan glissant). Réguliers (≥ 6 jours d'achat sur 180 j, écart médian ≤ 30 j), 12 variables sur 180 j, HistGradientBoosting sans calibration, 13 photos mensuelles du 22/03/2023 au 22/03/2024, test au 22/03/2024 (61 286 clients, 60,7 M€ de CA vraiment parti). Le calcul est reproductible : lignes triées par client et un seul thread (sans cela, le même modèle variait de 1 à 3 points d'une exécution à l'autre).

*Question.* Le lemme 2 fixe la vérité à 180 j, mais ce délai ne laisse que 7 photos d'entraînement (le label le plus long d'une photo d'entraînement doit être connu avant la date de test). Un label plus court, propre au rythme du client, laisse plus de photos : lequel, et que vaut-il en CA ?

*Règle testée.* **Vérité** : « parti » = aucun jour d'achat pendant 180 j après T. **Label d'entraînement** : H_i = clip(k × écart p90 du client, h_min, h_max) ; « perdu » = aucun achat avant H_i. Comme H_i ≤ 180 j, tout vrai partant est aussi « perdu » : la règle ajoute des faux perdus, des clients qui reviennent avant 180 j. 41 règles : k ∈ {1,5 ; 2 ; 3 ; 4 ; 5}, h_min ∈ {30, 45}, h_max ∈ {60, 90, 120, 180}, plus la référence « 180 j pour tous ».

**(a) Photos d'entraînement et CA borné** (pourcentage du CA des réguliers dont k × p90 dépasse h_max, 12 photos poolées) :

| h_max | Photos d'entraînement | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| 60 j | 11 | 4,5 % | 10,0 % | 14,6 % | 20,0 % |
| 90 j | 10 | 1,3 % | 4,5 % | 8,1 % | 11,8 % |
| 120 j | 9 | 0,3 % | 2,1 % | 4,5 % | 7,2 % |
| 180 j | 7 | 0,0 % | 0,3 % | 1,3 % | 2,9 % |

**(b) Pureté du label** = CA des vrais partants ÷ CA déclaré « perdu » (13 photos ; h_min = 45 j ; la référence à 180 j vaut 100 %) :

| k | h_max = 60 j | 90 j | 180 j |
|---|---|---|---|
| 1,5 | 47 % | 48 % | 49 % |
| 2 | 49 % | 52 % | 53 % |
| 3 | 52 % | 60 % | 63 % |
| 4 | 53 % | 64 % | 71 % |
| 5 | 54 % | 67 % | 77 % |

Par photo, les règles déclarent perdus 87 à 181 M€ pour 66,7 M€ de CA vraiment parti, soit 1,3 à 2,7 fois trop. h_min = 30 j est moins pur que 45 j de 3 à 10 points (l'écart diminue quand k augmente).

**(c) CA capté**, vérité à 180 j, classement par P × V, bootstrap sur les clients (200 tirages) :

| | Référence 180 j | k = 3, h_min = 45, h_max = 90 | Écart apparié |
|---|---|---|---|
| K = 700 : CA capté | 48,8 % [43–55] | 51,3 % [45–57] | +2,5 pts [0 ; +5] |
| K = 2 000 : CA capté | 64,8 % [60–70] | 65,5 % [61–70] | +0,65 pt [0 ; +2] |
| Moyenne de 5 graines, K = 700 | 49,2 % | 51,0 % | |
| Moyenne de 5 graines, K = 2 000 | 65,0 % | 65,5 % | |
| Marge à r = 20 %, K = 700 (M€/an) | 1,46 | 1,54 | +0,08 |
| Marge à r = 20 %, K = 2 000 (M€/an) | 1,94 | 1,96 | +0,02 |

K = 700 est le bas de la fourchette de capacité du dataset (600 à 760 appels par mois) ; K = 2 000 reprend les tableaux précédents.

*Preuve.* (i) **h_max = 180 j ne sert à rien** : il borne ≤ 0,3 % du CA pour k ≤ 3 mais ne laisse que 7 photos, contre 10 à h_max = 90 j. (ii) **La pureté monte avec k, h_min et h_max** et ne dépasse jamais 77 % : le CA « perdu » d'un label court n'est pas du CA récupérable, d'où la vérité à 180 j pour toute évaluation en CA (bornée elle aussi : un client qui revient au jour 200 compte comme parti). (iii) **Direction constante** : les six règles k ∈ {3, 4, 5}, h_min ∈ {30, 45}, h_max = 90 j sont au-dessus de la référence aux deux K (+2,1 à +3,1 pts à K = 700, +0,7 à +1,6 pt à K = 2 000) ; seule k = 4, h_min = 30 exclut zéro aux deux K (+3,1 [+1 ; +5] et +1,6 [+1 ; +3]). k = 1,5 est en dessous (jusqu'à −2,6 pts [−5 ; −1]). Les plafonds de 120 et 180 j sont mitigés. (iv) **Le bruit d'entraînement est du même ordre** : 5 graines font varier le CA capté d'un écart-type de 0,3 à 2 points, et l'IC d'une règle est de ± 5 points. (v) **En euros l'avance est petite** : à r = 20 %, la marge va de 1,38 à 1,56 M€ par an à K = 700 (référence 1,46) et de 1,86 à 1,99 M€ à K = 2 000 (référence 1,94), soit au mieux +0,09 M€ (3 à 6 %). (vi) **Le classement pèse plus que le label** : classer par P × V capte 48,8 % à K = 700 contre 33,6 % par probabilité seule (+15 points) ; le repère « récence ÷ p90 × V », sans modèle, 41,2 %. **Donc** : le label B (**k = 3, h_min = 45 j, h_max = 90 j**, règle fixée avant d'avoir vu k = 4 et 5, pour ne pas retenir le meilleur de 41 tirages ; k = 4 est équivalent aux erreurs de mesure près) est une alternative crédible au label A (180 j pour tous) : pas de dégradation, 10 photos au lieu de 7, pureté de 60 %. Le gain de CA capté (+1 à +3 pts) n'est pas démontré sur une seule photo de test ; la comparaison en plan glissant ci-dessous le précise. **Les deux labels sont gardés**, évalués contre la vérité à 180 j. ∎

*Comparaison A / B en plan glissant* (`08_label_rolling_backtest.ipynb`). Réentraînement mensuel : à chaque date de test, chaque chaîne est entraînée sur les photos dont le label est connu (A : `T ≤ T_test − 180 j` ; B : `T ≤ T_test − 90 j`) ; 7 dates de test du 22/09/2023 au 22/03/2024, vérité commune, 3 graines.

| Moyenne sur 7 dates | A | B | B − A |
|---|---|---|---|
| AUC | 0,884 | 0,884 | 0 |
| CA capté, K = 700 | 46,6 % | 47,8 % | +1,2 pt [+0,3 ; +2,0] |
| CA capté, K = 2 000 | 61,9 % | 62,7 % | +0,8 pt [+0,1 ; +1,5] |
| Marge à r = 20 %, K = 700 (M€/an) | 1,62 | 1,67 | +0,05 |

B fait mieux que A sur 6 dates sur 7. Une partie de l'écart tient au nombre de photos (B en a plus à chaque date : 4 contre 1 à la première, 10 contre 7 à la dernière) : sur les 4 dernières dates il tombe à +0,2 à +0,7 pt à K = 2 000. L'IC ignore le bruit entre dates (fenêtres qui se chevauchent, une seule année). B reste stable sur 10 dates, jusqu'au 22/06/2024 (AUC de 0,84 à 0,88 sur son propre label).

*Limites.* Une seule photo de test et une seule saison ; 41 règles comparées sur la même photo. La probabilité brute d'un label court n'est pas calibrée sur la vérité, ce qui joue dans P × V. Variables et segment rapide / moyen calculés sur 180 j ici, contre 365 j au lemme 1. 12 variables sans calibration, contre 13 variables calibrées au lemme 4 : les CA captés des deux lemmes ne se comparent pas au point près. La colonne « CA gaspillé » du notebook (CA annualisé des clients appelés qui restent) est dominée par les gros clients et ne sert pas à décider.

## 10. Ce que chaque lemme décide

État au gel : le lemme 6 (deux scores) n'est pas livré, et le modèle final utilise le label A (§ 11). La photo de test des lemmes 4 à 7 (22/03/2024) est la période du test final.

| Lemme | Décision (détail dans `02_modeling_decisions.md`) |
|---|---|
| 1 | Périmètre : les clients réguliers (51 % des clients, 95 % du CA, rythme sur 180 j) |
| 2 | Vérité à 180 jours, par convention ; horizons courts exclus ; décote de 12 % du CA du total mesurée, non appliquée dans les calculs |
| 3 | Le partiel n'est pas un gisement de CA sauvable : second score, hors chiffre de tête |
| 4 | Le total est le label principal et le chiffre de tête |
| 5 | Aucun label ne domine en marge : la rétention se mesure par pilote |
| 6 | Pas de label élargi ; deux scores avec ≈ 25 % des places au total |
| 7 | Deux labels d'entraînement gardés, A (180 j pour tous) et B (relatif au rythme) |

## 11. Résultats du code : backtest, gel et test final (25-26/11)

Source : `results/metrics.parquet` et `results/runs.parquet`. **Dates décalées (+2 ans).** Réguliers, vérité à 180 jours, classement par P × V.

**Backtest** (moyenne des 5 dates du 22/10/2025 au 22/02/2026, label A, CA capté) :

| Modèle | K = 700 | K = 2 000 |
|---|---|---|
| Baseline récence ÷ p90 (*) | 44,4 % | 56,0 % |
| Régression logistique (*) | 45,5 % | 60,2 % |
| BG/NBD (*) | 46,2 % | 58,9 % |
| HistGradientBoosting | 48,6 % | 64,2 % |
| XGBoost | 49,9 % | 64,2 % |
| XGBoost réglé (configuration finale) | 50,6 % | 65,4 % |

(*) Runs antérieurs au passage des tendances en log-ratios : comparaison indicative. Le gain du réglage est un peu optimiste (Optuna a vu des données plus récentes que les premières dates).

**Variables.** Critère écrit avant les résultats : +1,5 point à K = 700 sur HGB et XGBoost, 4 dates sur 5. `extended` : +0,6 (HGB, 4 dates) et +0,4 (XGBoost, 3 dates). `market_only` : +0,7 (3 dates) et −0,5 (0 date). **Les deux sont rejetés : `basic`.**

**Test final** (photo du 22/03/2026, run `final-14fde1d325`, un seul run du test dans `runs/`) : 61 302 réguliers, 2 226 partants (3,6 %), 60,7 M€ de CA parti. IC 95 % entre crochets.

| Mesure | K = 700 | K = 2 000 | K = 4 000 |
|---|---|---|---|
| CA capté, réguliers | **51,5 %** [46–57] | **66,2 %** [61–71] | 76,4 % [72–80] |
| Taille seule | 7,3 % | 14,5 % | 19,9 % |
| Oracle | 80,6 % | 99,3 % | 100 % |
| CA capté, **tout le churn** | **26,1 %** [23–30] | 33,6 % | 38,8 % |
| Taille seule, tout le churn | 3,7 % | 8,8 % | 13,6 % |
| Oracle, tout le churn | 50,2 % | 69,2 % | 81,6 % |

AUC 0,89, PR-AUC 0,39. Brier brut 0,0272 et calibré 0,0272 : la calibration n'améliore pas le Brier sur le test. Liste de 700 : 54 % de vrais partants, 31,3 M€ de CA capté, soit **1,54 M€ de marge à r = 20 %** (0,77 à 2,31 de r = 10 à 30 %), sans décote (1,36 M€ à r = 20 % avec 12 %).
