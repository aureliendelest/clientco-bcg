# Risques, questions ouvertes, prochaines étapes

> **Rôle** : Ce qui peut mal tourner, ce qui reste à décider, ce qui reste à faire.
> **Pour** : L'équipe, à relire chaque semaine.
> **À jour au** : 25/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Vigilances pour plus tard

Les risques les plus coûteux sont la fuite temporelle, un signal qui arrive trop tard pour agir, et une rétention supposée plutôt que mesurée. Classés par criticité :

| Vigilance | Pourquoi | Parade prévue |
| --- | --- | --- |
| Fuite temporelle (délai de label) | Avec 2 ans, une photo exige un historique avant elle et 180 jours de suivi après. Les 7 photos poolées du notebook se chevauchent : les utiliser pour entraîner puis tester ferait « voir le futur » au modèle. Une AUC supérieure à 0,95 est presque toujours une fuite | Historique des variables limité à 180 jours ; entraînement sur les photos dont le suivi finit avant la date de test ; test automatique de non-chevauchement |
| Signal trop tardif | L'AUC de 0,90 de la récence est en partie mécanique : un client silencieux depuis 170 jours est déjà presque perdu | Juger le modèle sur sa capacité à alerter tôt (tendances de fréquence et de CA), pas sur le silence |
| Calibration | Multiplier une probabilité par V suppose que 0,05 signifie 5 % ; un boosting brut ne le garantit pas | Calibration isotonique ou de Platt sur la validation, par segment et par classe de rythme |
| Précision en baisse | Dans une liste de gros clients, le taux de base est \~2 % : la part de vrais partants sera faible même avec un bon modèle (44 % pour la liste « récence relative ») | Assumer l'échange précision contre euros ; seuil minimal de probabilité |
| Définition de V | Le CA passé surestime ce que perdent les clients en déclin avant de partir | Tester le CA des 180 derniers jours annualisé, le CA 12 mois, une prévision |
| Peu de partants chez les gros clients | \~240 événements par semestre dans le décile 10 : modèle instable, résultat dominé par quelques comptes | Intervalles de confiance par bootstrap ; variante par segment de valeur |
| Grands comptes déjà suivis | Les grands comptes et clients à interlocuteur dédié feront remonter dans le classement des comptes où l'appel apporte peu | Liste séparée pour leur responsable si le client les marque. Les interviews confirment un suivi relationnel des bons clients (Sales Manager, Head of Sales) sans repérage systématique des baisses ; la liste des plus gros clients capte 22 % du CA du partiel et 14,5 % du CA du total sans modèle : le repère « taille seule » en tient compte |
| Horizons incomparables | Avec un H par client, les taux de churn ne se comparent plus d'un segment à l'autre | Mesure commune en CA : CA réellement perdu capté dans les K premiers clients |
| Maturité du label | Un horizon long exclut les photos les plus récentes ; les clients lents ont le plus de censure | Horizons courts pour les clients rapides ; calibrer les classes de rythme sur les données |
| Saisonnalité | 2 ans seulement : un seul recoupement annuel ; trois creux d'activité (mai 2023, sept. 2023, mai 2024) tombent dans les fenêtres | Ne pas utiliser de jour de semaine. **Pas de variable « mois »** (un seul recoupement annuel : le modèle la mémoriserait) ; variables relatives au rythme du client, éventuellement normalisées par l'activité globale ; amplitude mesurée : CA par jour de 71 (mai) à 123 (août), churn total de 2,5 à 3,2 % du CA selon la photo ; à signaler dans le deck |
| Rétention non mesurée | Les 20 % de rétention du chiffrage sont une hypothèse sans mesure | Scénarios 10, 20, 30 % dans l'exercice ; pilote avec témoins dans le cas réel |
| Sous-ensemble du réseau | Un client « perdu » a peut-être acheté hors périmètre du dataset : churn surestimé ; l'extrapolation à 15,5 Md€ suppose un échantillon représentatif | À annoncer comme limite ; question au tutorat |
| Segments non identifiables | Contrats long terme, institutions publiques, appels d'offres : leur inactivité n'a pas le même sens | Question au tutorat ; sinon modèle par segment de valeur |
| Adoption | OffSupp : la plupart des équipes n'ont pas adopté immédiatement ; Sales Manager sceptique | Raisons lisibles par client, Head of Sales embarqué, indicateurs d'adoption définis avant le déploiement |
| Confidentialité | Le brief est classé confidentiel ; `docs/` n'est pas dans `.gitignore` alors que le dépôt a un remote GitHub dont la visibilité n'est pas connue | Ignorer `docs/`, `runs/`, prédictions et modèles ; ne jamais versionner de données ; ne pas coller les comptes rendus dans des outils non approuvés |

Rappel des règles du brief : ne livrer que ce que l'on sait expliquer et défendre, vérifier chaque chiffre, et étiqueter les preuves (Confirmé, Source unique, Déduit, Manquant).

Vigilances ajoutées le 18/11 :

- **Double comptage du churn.** Si V est prévu sur le CA réel, la probabilité de départ est appliquée deux fois.
- **Coût d'opportunité.** Dédier des commerciaux à la prévention retire du temps à leurs autres ventes ; le CA sauvé doit le dépasser.
- **Relation client.** Un commercial dédié ne connaît pas forcément le client, alors que beaucoup ont un interlocuteur attitré ; l'effet d'un appel peut être plus faible que celui du commercial habituel.

Vigilances ajoutées le 24/11 :

- **Label partiel bruité et saisonnier.** AUC de 0,68 (total : 0,89) ; environ 30 % de son CA ne dure pas ; il compare deux saisons (rapport CA(B)/CA(A) de 0,88 à 1,07 selon la photo). Parade : second modèle seulement, hors chiffre de tête, validé par le pilote.
- **Rétention du partiel inconnue.** Pour égaler le total, le partiel doit avoir une rétention d'au moins 37 % de celle du total (lecture « CA capté ») ou ≈ 100 % (lecture « surplus »). Aucune interview ne la quantifie. Parade : pilote randomisé des deux listes.
- **Effet de taille.** Près des trois quarts du CA capté par le modèle partiel (113 M€ sur 156 M€) viennent de la seule taille des clients. Parade : toujours comparer au repère « taille seule ».
- **Échelle de la capacité.** Les listes sont calculées sur le dataset (≈ 30 % du CA) et la capacité d'appels vaut pour l'entreprise : les mélanger surestime K d'un facteur ≈ 3. Parade : convertir (K ≈ 700 et 4 000 pour 500 appels par semaine).
- **Une seule photo de test.** Le bootstrap mesure le bruit d'échantillonnage, pas la variation d'une saison à l'autre ; aucune photo d'entraînement ne part de novembre-décembre, où les taux de churn sont plus élevés. Parade : le dire dans le deck, pas de variable « mois ».
- **Label durable non validable.** Une seule photo possible, donc pas de découpage temporel ; il ne sert qu'au dimensionnement.
- **Reproductibilité.** Un achat retourné laisse un net de ±1e-13 et DuckDB ne garantit pas l'ordre des lignes : sans seuil de 1 centime ni tri par client, deux exécutions donnaient 118 287 puis 118 271 clients. Parade : seuil et tri dans le code, à porter dans `src/churn/`.
- **Quota du total et chiffre de tête.** Avec 25 % des places, le total capte 40 à 45 % (K ≈ 1 000 à 1 400) de son CA en jeu au lieu de 56 à 61 % : le chiffre de tête est plus bas que celui d'une liste entièrement dédiée au total. Parade : annoncer les deux chiffres.
- **Capacité non documentée.** 500 appels par semaine est une hypothèse (aucun chiffre dans les documents), avec des rendements décroissants (doubler les appels : +15 à +33 % de marge). Parade : fourchette 200 à 1 800, question au tutorat.

## 2. Problèmes dans les documents

- **Devise.** Le fichier Financials est en EUR alors que le deck Key Facts parle de 15,5 Md$. Le dataset n'indique pas de devise.
- **Pertes de 50 M** (Supply Chain) : ni la devise ni la nature (CA ou marge) ne sont précisées.
- **+1 %** (CDO) : « income growth » est ambigu (CA ou résultat) et vient d'un plus petit acteur APAC. **+2 %** (OffSupp) : autre secteur, non transposable tel quel. **+1 %** du cross-sell : source la plus faible, un article lu par le Sales Manager.
- **Fonctions** : la Data Engineer est aussi citée Data Scientist ; le Supply Chain Director est cité Supply Chain Manager dans le brief ; les anciennetés de l'équipe data diffèrent entre la slide 20 et le compte rendu.
- **Absences.** Aucun taux de churn dans les interviews ; aucune interview du Marketing, de la Finance ni du CEO. Aucun taux de rétention ni nombre d'appels par semaine non plus : la capacité et la rétention restent des hypothèses.

## 3. Problèmes dans l'environnement de travail

- Le CSV de 4,2 Go ne se charge pas dans pandas : DuckDB et Parquet sont indispensables.
- ~~Le `requirements.txt` actuel ne liste ni `duckdb` ni de bibliothèque de modélisation~~ **Résolu** : `duckdb`, `scikit-learn`, `xgboost`, `optuna` et `shap` y figurent désormais.
- Le notebook exploratoire initial (supprimé le 25/11) prévoyait un décalage de +5 ans qui n'a pas été appliqué dans le fichier ; remplacé par la décision de +2 ans.

## 4. Questions ouvertes

Trois décisions attendent une réponse de l'équipe, et neuf questions sont à poser au tutorat du 20/11. Le seuil de probabilité, les grands comptes, V par défaut, K paramétré, la décote ρ (mise de côté le 25/11) et les paliers de capacité (repris le 24/11) avaient été validés. Les recommandations ci-dessous restent des propositions tant qu'elles ne sont pas confirmées.

### Décisions en attente

| Décision | Recommandation proposée |
| --- | --- |
| Explicabilité | SHAP (TreeExplainer) pour les modèles boostés, coefficients pour la régression logistique, importance par permutation sur le test ; trois raisons par client en langage commercial |
| Git et partage des runs | Ignorer `docs/`, `runs/`, prédictions et modèles ; versionner code, configurations et `results/runs.parquet` ; une configuration exécutée ne se modifie jamais ; branches courtes et petites pull requests vers `main` |
| Répartition des rôles | A données (chemin critique, 3 premiers jours), B modèles, C évaluation, impact et explicabilité, D récit, second use case et maquette ; contrat d'interface = schéma de la table de photos (`client_id`, date T, variables, labels), écrit dès le 18–19/11 et présenté comme structure algorithmique au CTM |

### Tour de questions à venir

- Métrique commune d'évaluation et taille du top-K : CA réellement perdu capté dans les K premiers clients, rapporté au CA total perdu. **Traité** : mesure retenue, avec le repère « taille seule » ; K ≈ 700 et 4 000 pour 500 appels par semaine (hypothèse).
- Découpage de validation, compatible avec l'horizon des labels. **Appliqué** : 7 photos d'entraînement (label A) ou 10 (label B), test au 22/03/2026.
- Tests anti-fuite et de cohérence des labels.
- Simulation d'impact : scénarios de rétention, par exemple trois scénarios ou une simulation de Monte Carlo.
- Architecture cible du cas réel (scoring hebdomadaire, suivi de dérive, journalisation du pilote).

### Questions au tutorat du 20/11

| # | Question | Pourquoi elle compte |
| --- | --- | --- |
| 1 | Comment le dataset a-t-il été échantillonné (agences et clients) ? Un client peut-il acheter hors périmètre ? | Le churn est peut-être surestimé |
| 2 | Peut-on marquer les clients sous contrat, institutions publiques et grands comptes centralisés ? | Leur inactivité n'a pas le même sens ; ciblage des gros clients |
| 3 | Combien d'appels par semaine un commercial peut-il passer ? | Fixe la taille K de la liste ; aucun chiffre dans les documents. Hypothèse de travail : 500 appels par semaine (100 commerciaux × 5), fourchette 200 à 1 800 |
| 4 | Existe-t-il un journal d'appels ou un CRM exploitable, et des rotations ou absences de commerciaux ? | Première estimation d'effet par inférence causale sur données d'observation, ou quasi-expériences |
| 5 | Quelle devise (Key Facts en dollars, Financials en euros) ? Que signifie une ligne négative, et pourquoi la quantité vaut au moins 3 ? | Interprétation des montants et des retours |
| 6 | Que sont les \~100 « agences » de moins de 50 clients, presque tout au téléphone ? | Centres d'appels ou comptes clés ? |
| 7 | Qu'explique le creux de mi-septembre 2023 ? Pourquoi aucune ligne le 29/02/2024 ? | Fenêtres d'étiquetage et jours manquants |
| 8 | Que représente `date_invoice` et pourquoi déborde-t-elle jusqu'en 2026 ? Les visites commerciales sont-elles saisies (0,05 % des lignes) ? | Choix de la date d'activité ; effet d'une visite invisible |
| 9 | Les 50 M de pertes : CA ou marge, quelle devise ? Le +1 % de la CDO : CA ou résultat ? Les onglets A à N de Financials sont-ils des concurrents ? | Fiabilité du chiffrage |
| 10 | Aujourd'hui, un commercial est-il alerté quand un client habituel baisse nettement ses achats ? Par qui, avec quel suivi ? | Départage la lecture « CA capté » de la lecture « surplus » |
| 11 | Les 1 000 à 4 000 plus gros clients sont-ils dans le périmètre des commerciaux ou gérés centralement ? | Les listes pourraient pointer des clients déjà suivis |
| 12 | Quelle part de son temps un commercial peut-il consacrer à des appels de prévention ? Existe-t-il une mesure de rétention (reprise de commande après un appel) ? | Capacité K et hypothèse de rétention, aujourd'hui sans mesure |
| 13 | Quelle est la part exacte du dataset dans le CA et les clients de l'entreprise (≈ 30 % du CA, ≤ 38 % des clients selon nos calculs) ? | Conversion de la capacité d'appels vers l'échelle du dataset |

## 5. Limites à annoncer dans le deck

Sous-ensemble de clients, pas de journal d'actions, contrats inconnus, deux ans d'historique seulement (saisonnalité mal couverte), rétention non mesurée.

## 6. Prochaines étapes

Le contrat d'interface (schéma de la table de photos) et le script d'ingestion débloquent tout le reste : sans eux, modèles et évaluation ne peuvent pas avancer en parallèle.

- [ ] Répondre aux trois décisions en attente (section Questions ouvertes)
- [ ] Écrire le contrat d'interface (schéma de la table de photos) avant le Nov 19, 2026
- [ ] Ajouter à `.gitignore` : `docs/`, `runs/`, prédictions et modèles ; compléter `requirements.txt` (DuckDB, scikit-learn, XGBoost, Optuna, SHAP)
- [ ] Écrire le script d'ingestion (+2 ans sur les deux dates) et produire `sales_shifted.parquet`
- [ ] Mettre à jour la feuille de chiffrage du fichier de priorisation avec la part de CA perdue mesurée (3,4 à 4,8 %)
- [ ] Préparer pour le CTM du Nov 20, 2026 la structure algorithmique et les questions au tutorat
- [ ] Décider du tableau de bord (go ou no-go) le Nov 24, 2026
- [ ] Geler le code le Nov 25, 2026 pour le rendu du Nov 26, 2026 à 12h
- [x] Notebook d'exploration découpé en notebooks `01` à `05` ; synthèse périmée supprimée, décisions dans `02_modeling_decisions.md`
- [ ] Recalculer le chiffre d'impact (détection mesurée sous contrainte de capacité, extrapolation à l'entreprise, nombre de cohortes par an à vérifier) et le reporter dans le fichier de chiffrage
- [ ] Porter dans `src/churn/` le label partiel brut, la définition du jour d'achat net > 0,01 €, le tri par client, le repère « taille seule » et le test de répartition de K
- [ ] Tester régression logistique et XGBoost sur les deux labels (A et B), puis un second test hors de la photo du 22/03/2026
- [ ] Poser les questions 10 à 13 (alerte actuelle sur les baisses, grands comptes, temps et rétention, part du dataset)
- [ ] Concevoir le pilote : deux listes randomisées, quota de 25 % pour le total, groupe témoin, indicateurs d'adoption

Prochain tour de questions de cadrage : tests anti-fuite, simulation d'impact et architecture cible du cas réel. La métrique commune et le découpage de validation sont traités ; la simulation d'impact doit encore recevoir l'extrapolation à l'entreprise.
