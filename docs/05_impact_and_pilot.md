# Impact, pilote et plans

> **Rôle** : Combien vaut le churn en M€, avec quelles hypothèses, et comment on le prouve (pilote randomisé, uplift).
> **Pour** : L'équipe qui construit le récit et le chiffrage du deck.
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Impact : une fourchette, pas un chiffre

`Impact = CA × part perdue (mesurée) × détection (mesurée au backtest) × rétention (hypothèse) × marge (24,65 %)`

- Rétention en trois scénarios : 10, 20, 30 %. **C'est l'hypothèse dominante** (tornade) et elle n'est pas mesurable avec les transactions seules : seul un pilote la mesurera.
- **Sur le dataset** (test final) : liste de K = 700, **1,54 M€ de marge par an** à r = 20 % (0,77 à 2,31 de r = 10 à 30 %).
- **À l'échelle de l'entreprise** (`impact.py`, détection mesurée de 26,1 %, part perdue 3,4 à 4,8 %) : **8,2 M€ par an au central** (part perdue 4,1 %, r = 20 %), 6,8 à 9,6 à r = 20 %, 3,4 à 14,4 de r = 10 à 30 %.
- Ces montants sont **bruts** : ni coût d'opportunité, ni décote de 12 % du CA qui revient.
- Coût d'opportunité des commerciaux dédiés à déduire.

**Ce que suppose l'extrapolation.** Le résultat du dataset (≈ 30 % du CA) vaut pour toute l'entreprise, avec une capacité d'appels à la même échelle (K ≈ 700 sur le dataset). La détection est prise sur tout le churn (`capture_total_at_k`), pas sur les seuls réguliers. La part perdue (3,4 à 4,8 %) vient des notebooks `01` à `03`, avec l'ancienne définition du jour d'achat.

**Ancien chiffrage, périmé.** 13 à 18 M€ (central 15,7) supposait une détection de 50 %, non mesurée sous contrainte de capacité. Ne plus le citer comme résultat.

## 2. Simulation d'impact et tornade

- **Simulation d'impact.** Trois scénarios (bas, central, haut) sur la rétention (10, 20, 30 %), avec la part de CA perdue mesurée et la détection issue du backtest ; graphique en tornade sur trois paramètres (part perdue, détection, rétention) ; impact net du coût d'opportunité ; Monte Carlo seulement en annexe.

**Tornade** (`impact.tornado_table`). Chaque ligne fait varier un seul paramètre, les autres au centre. Central : 8,2 M€ de marge par an (CA 15,5 Md€, marge 24,65 %, part perdue 4,1 %, détection 26,1 %, rétention 20 %). La détection varie dans l'intervalle à 95 % du test.

| Paramètre | Valeur basse → haute | Impact (M€ de marge par an) | Amplitude (M€) |
| --- | --- | --- | --- |
| Rétention | 10 % → 30 % | 4,1 → 12,3 | 8,2 |
| Part de CA perdue | 3,4 % → 4,8 % | 6,8 → 9,6 | 2,8 |
| Détection | 22,8 % → 29,7 % | 7,1 → 9,3 | 2,2 |

La rétention domine : c'est l'hypothèse à mesurer en priorité, par le pilote.

## 3. Deux plans : exercice et cas réel

Les deux listes (total et partiel) citées ci-dessous sont une cible pour le pilote : le code ne livre que la liste du score total.

Dans l'exercice rien n'est déployé : on prouve hors ligne ce qu'on peut avec les transactions, et on décrit le reste sur le papier. Dans le cas réel, un pilote randomisé apporte la preuve causale.

|  | Plan A : exercice | Plan B : cas réel |
| --- | --- | --- |
| Données | Transactions seulement | Transactions, journal d'actions, CRM, contrats, stocks |
| Preuve de valeur | Backtest hors ligne et simulation avec hypothèses | Pilote randomisé avec groupe témoin |
| Uplift | Non réalisable, présenté comme étape future | Après lecture du pilote, si justifié |
| Livrable | Code (26/11, 12h), 5 à 10 slides (26/11, 17h), oral (27/11) | Programme sur 12 mois |
| Discours | Voici ce que nous avons mesuré et ce que nous supposons | Voici ce que le pilote a prouvé |

### Plan A : dans les contraintes de l'exercice

- **Calendrier.** 17 au 19/11 : figer la définition du churn, préparer le schéma de l'algorithme. 20/11 : tutorat et Case Team Meeting. 21 au 23/11 : baseline, variables, modèles, backtest. 24 et 25/11 : évaluation en CA, simulation d'impact, explicabilité, gel du code. 26/11 : rendus. 27/11 : présentation.
- **Impact.** CA × part de CA perdue (mesurée) × détection (mesurée au backtest) × rétention (hypothèse en scénarios, par exemple 10, 20 et 30 %) × marge de 24,65 %. Une simulation sur ces scénarios donne une fourchette plutôt qu'un chiffre unique.
- **Ce qu'on dit.** Le churn est prévisible chez les clients réguliers, la valeur est dans le haut de la base, la rétention est une hypothèse à valider. Chiffre de tête : le churn total (part du CA perdu captée par la liste de K clients, convertie en marge à rétention 10 / 20 / 30 %). Le churn partiel est présenté comme un potentiel supplémentaire, hors chiffre de tête, à confirmer par le pilote.
- **Ce qu'on ne fait pas.** Pas de modèle d'uplift avec des traitements inventés : le jury le verrait. On annonce aussi les limites (sous-ensemble de clients, pas de journal d'actions, contrats inconnus).

Storyline de 10 slides au maximum : 1) synthèse et demande ; 2) programme global, les 5 use cases en feuille de route 2027, le churn en premier ; 3) l'enjeu mesuré ; 4) définition du churn et segmentation ; 5) modèle et performance face à la baseline, en CA ; 6) impact en M€, scénarios et sensibilité (le churn total en chiffre de tête, le partiel en potentiel à confirmer par le pilote) ; 7) mise en œuvre : playbook, pilote, indicateurs d'adoption ; 8) feuille de route 2027 et second use case ; 9) budget, ressources, risques ; 10) prochaines étapes. En annexe : qualité des données, contrôles anti-fuite, détails du modèle.

### Plan B : dans le cas réel

| Phase | Durée indicative | Contenu | Critère de passage |
| --- | --- | --- | --- |
| 0. Cadrage et données | 4 à 6 semaines | Définir le churn avec les métiers ; marquer contrats, institutions publiques, grands comptes ; récupérer CRM, journaux d'appels, stocks, délais | Définition validée par le Head of Sales et la Commercial Director |
| 1. Modèle et backtest | 6 à 8 semaines | Baseline, modèle par segment, validation temporelle, évaluation en CA | Le modèle bat la baseline en CA sous contrainte de capacité |
| 2. Pilote randomisé | 3 à 6 mois | Randomisation par client stratifiée, tous niveaux de risque, **deux listes randomisées séparément (churn total et churn partiel) pour mesurer la rétention de chacune**, journalisation de chaque action, \~25 % de témoins | Échantillon suffisant, suivi complet |
| 3. Lecture et décision | 4 à 6 semaines | Effet moyen, puis par décile de risque et par segment ; taux de rétention réel | ROI positif confirmé |
| 4. Uplift (optionnel) | 2 à 3 mois | T-learner, X-learner ou forêt causale, courbe de Qini sur données randomisées mises de côté | L'uplift bat le classement par risque × valeur |
| 5. Industrialisation | En continu | Scoring hebdomadaire, liste courte par commercial, suivi de dérive, groupe témoin permanent de faible taille | Adoption mesurée (listes consultées, appels passés, motifs saisis) |

Les durées sont des propositions de ma part, pas des données du client. Le plan A correspond aux phases 0 et 1 du plan B ; les phases 2 à 5 deviennent la feuille de route 2027 présentée.

### Articulation churn, pilote et uplift

Un client « persuadable » est un client dont l'action change le comportement : cet effet individuel n'est jamais observable, il ne peut venir que d'une expérience. Cibler les clients « moyennement récents » est une approximation de l'uplift fondée sur le risque (hypothèse d'un effet maximal au milieu) ; l'uplift mesure cet effet au lieu de le supposer.

La dépendance est asymétrique : le scoring de churn est un prérequis du pilote (il définit la population et la stratification), et l'expérience est un prérequis de la preuve (le scoring seul ne prouve pas que l'action sauve du CA).

- **Randomisation.** Dans la liste à risque, on tire au sort qui est appelé (traiter) et qui ne l'est pas (témoin) ; la comparaison des deux groupes donne l'effet de l'appel.
- **Journalisation.** On note qui a réellement été appelé, quand, par qui, par quel canal, avec quel résultat. Elle ne se rattrape pas après coup : un traitement non enregistré est perdu, et un groupe témoin reconstruit est biaisé.
- **Stratification.** On tire au sort à l'intérieur de groupes homogènes (segment de fréquence, tranche de valeur, niveau de risque), pour que les deux groupes aient la même composition.
- **Taille.** Ordre de grandeur à confirmer : avec 44 % de churn dans la liste (ancienne liste « récence relative » ; la liste de 700 du test final en compte 54 %, taille non recalculée) et une rétention de 20 % (\~9 points), l'effet moyen se détecte avec \~500 clients par groupe ; apprendre des effets différents par segment demande de l'ordre de 4 fois plus par sous-groupe.
- **Recouvrement.** Le pilote doit inclure tous les niveaux de risque, sinon on ne peut pas tester l'hypothèse de la fenêtre d'intervention.
- **Uplift.** Première analyse simple : l'effet moyen par décile de risque et par segment. Les modèles (T-learner, X-learner, forêt causale) ne servent que si l'effet varie au-delà du risque et si l'échantillon le permet.

Proposé pour la feuille de route 2027 (à valider) : trimestre 1, modèle de churn et mise en place de la randomisation ; trimestre 2, pilote et lecture ; trimestre 3, uplift si justifié ; trimestre 4, déploiement élargi.
