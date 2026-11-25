# Ce que disent les données

> **Rôle** : Constats sur les transactions : volumétrie, concentration, rythme d'achat, signaux, saisonnalité, anomalies.
> **Pour** : Quiconque travaille sur les données ou prépare le récit.
> **À jour au** : 25/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Constats clés

Le churn est mesuré à 3,4–4,8 % du CA par an, bien en dessous de l'hypothèse centrale de 8 % du chiffrage d'équipe, et il est prévisible chez les clients réguliers. Source : notebooks d'exploration `01` à `03` (photo du 22/03/2024 dans les dates d'origine, base = clients actifs sur 180 jours, churn = aucun achat dans les 180 jours suivants).

| Thème | Constat | Conséquence |
| --- | --- | --- |
| Volumétrie | 63,3 M de lignes, 170,6 k clients, 582 « agences », 2 ans de commandes | Sous-ensemble de l'entreprise (>450 k clients, >1 000 agences) |
| Concentration | 1 % des clients = 31 % du CA ; les 13 % de clients à 104+ jours d'achat = 76 % du CA ; les 26 % à 1–3 jours = moins de 1 % | Raisonner par segment et en valeur |
| Rythme d'achat | Écart médian 3 j, p95 = 36 j, p99 = 126 j ; l'écart typique varie d'un facteur \~20 selon le client | Aucun seuil universel de churn |
| Taux de churn | 18 % des clients actifs par semestre, mais \~3 % du CA. Plus petit décile de CA : 52,5 % de churn ; plus gros : 2,2 % (mais \~35 % du CA churné) | Évaluer le modèle en CA |
| CA perdu par an | 3,4 % (12 mois contre 12 mois) à 4,8 % (semestre annualisé) | Hypothèse centrale de 8 % à corriger |
| Érosion | Les reculs supérieurs à 20 % (–469 M) sont presque compensés par les hausses (+448 M) ; un recul supérieur à 50 % est durable pour \~62 % des clients concernés | Second label « chute durable de CA » |
| Signaux | Chez les réguliers, l'AUC de la récence atteint 0,90 ; les futurs partants décrochent \~5 mois avant ; une chute de fréquence de plus de 50 % sur 90 j = \~15 % de churn contre 1–2 % ailleurs | Le churn est prévisible, au moins chez les réguliers |
| Récence relative | AUC 0,88 chez les réguliers, 0,55 sur toute la base | Variables de rythme seulement avec assez d'historique (6 achats ou plus) |
| Canaux | Téléphone = 64 % du CA ; en ligne = 9 % ; visite commerciale = 0,05 % des lignes | Le téléphone est le canal naturel ; l'effet d'une visite est invisible |
| Clients multi-agences | 47 % achètent dans plusieurs agences | L'agence est un mauvais identifiant client |
| Effet agence | \~1 % d'agences atypiques après correction de la mixité de fréquence | Utile pour choisir des agences pilotes |
| Nouveaux clients | \~1/3 rachètent le mois suivant, 86–91 % refont un achat sous 6 mois | Phase d'intégration à part |

## 2. Saisonnalité

CA par jour de 71 à 75 (mai) à 118 à 123 (août), indice 100 = moyenne ; croissance de +4,7 % d'une année sur l'autre. Part du CA de la base : churn total 2,5 à 3,2 %, partiel 10,3 à 14,0 %, hausses 12,4 à 21,1 % ; rapport CA(B)/CA(A) de la base de 0,88 à 1,07. Les amplitudes du total et du partiel sont voisines (±12 % et ±15 %) ; le défaut du partiel est de comparer deux saisons.

## 3. Anomalies des données

| Anomalie | Détail | Traitement retenu |
| --- | --- | --- |
| Dates de commande | Du 22/09/2022 au 22/09/2024 alors que le fichier s'appelle 2026 | Dates fictives : +2 ans, appliqué une fois par le script d'ingestion |
| `date_invoice` | De 2021 à 2026 ; 29 % des lignes facturées après la commande, 4,1 % à plus de 30 jours ; 99 lignes avant la commande | Utiliser `date_order` comme date d'activité |
| Lignes négatives | 3,8 % des lignes ; retours = 577 M, 5,8 % du CA positif | Un avoir n'est pas un achat ; il compte comme signal, pas comme activité. Jour d'achat = CA net du jour > 0,01 € (un achat retourné le jour même n'est pas un achat), CA = ce net : base de 119 656 à 118 009 clients, CA de référence −6 % ; un retour fait un autre jour n'est pas déduit |
| Lignes à zéro | 3,8 % des lignes | Jour d'achat = CA net du jour > 0,01 € |
| Jour de semaine | Dimanche \~14 % du CA, samedi \~0 % : dates décalées | Aucune variable de jour de semaine |
| Calendrier | Aucune ligne le 29/02/2024 | Pas de collision au décalage ; à signaler |
| Creux d'activité | Trois creux qui divisent l'activité par \~2 : mai 2023, sept. 2023 (inexpliqué), mai 2024 | Un silence pendant un creux n'est pas forcément une perte ; question au tutorat |
| Quantité | Jamais inférieure à 3, même sur les lignes négatives | La quantité ne signale pas les retours ; le signe de `sales_net` oui |
| « Agences » | \~100 des 582 servent moins de 50 clients, presque tout par téléphone | Probablement des centres d'appels ou comptes clés ; question au tutorat |
| Démarrage | \~80 k clients « nouveaux » en sept.–oct. 2022 | Ce sont des clients existants vus pour la première fois (censure à gauche) ; période de chauffe de 3 mois |
| Canaux | Téléphone = 64 % du CA et en ligne = 9 %, contre « 60 % en agence » et « online quasi nul » dans les documents | À clarifier ; signaler l'écart |

## 4. Valeur espérée par décile de CA

Ordre de grandeur d'après les déciles du notebook (taux de churn × CA moyen sur 12 mois) :

| Décile de CA | Taux de churn | CA moyen | Valeur espérée moyenne |
| --- | --- | --- | --- |
| 10 (gros) | 2,2 % | 283 k | \~6 200 |
| 9 | 4,0 % | 57 k | \~2 300 |
| 8 | 5,6 % | 27 k | \~1 500 |
| 1 (petits) | 52,5 % | 234 | \~120 |

## 5. Où sont les chiffres du périmètre, de l'horizon et des labels

Ils sont dans `03_evidence.md` : segmentation par rythme (lemme 1), part des clients qui reviennent selon l'horizon (lemme 2), CA en jeu selon la définition de la perte (lemme 3), test des deux labels (lemmes 4 à 6), calibration et comparaison des labels A et B (lemme 7).
