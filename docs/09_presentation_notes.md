# Notes de présentation

> **Rôle** : Storyline du deck, calendrier jusqu'au rendu, discours de justification et questions probables.
> **Pour** : L'équipe qui prépare la soutenance du 27/11.
> **À jour au** : 26/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## 1. Storyline cible du deck (10 planches max)

1) Synthèse et demande · 2) Programme global : les 5 use cases en feuille de route 2027, churn en premier · 3) L'enjeu mesuré · 4) Définition du churn et segmentation · 5) Modèle et performance face à la baseline, en CA · 6) Impact en M€, scénarios et sensibilité · 7) Mise en œuvre : playbook, pilote, indicateurs d'adoption · 8) Feuille de route 2027 et 2ᵉ use case · 9) Budget, ressources, risques · 10) Prochaines étapes. **Annexes** : qualité des données, contrôles anti-fuite, détails du modèle.

## 2. Calendrier jusqu'au rendu

| Date | Contenu |
| --- | --- |
| 17–19/11 | Définition du churn figée, schéma de l'algorithme (fait) |
| **20/11** | Tutorat BCG et Case Team Meeting (Zoom, créneau de 45 min par équipe) |
| 21–23/11 | Ingestion, baseline, variables, modèles, backtest |
| 24–25/11 | Évaluation en CA, simulation d'impact, explicabilité ; décision go/no-go maquette le 24 ; gel prévu le 25, **configuration gelée et test final le 26** |
| **26/11** | 12h : code. 17h : slides |
| **27/11** | Présentation au CEO |

## 3. Justifier la perte à 6 mois (discours d'une minute)

« Un client est "parti" quand il n'a rien acheté depuis six mois. C'est la définition qui sert à mesurer le CA réellement perdu.

Nous l'avons choisie parce que les alternatives ne tiennent pas. Avec un seuil de 60 jours, 37 % des clients rapides et 57 % des clients moyens déclarés perdus rachètent dans les trois mois suivants : on appellerait des clients qui seraient revenus seuls. Et passé ce seuil, il n'y a pas de palier naturel dans les données. Six mois est donc une convention, que nous assumons : elle est commune à tous les segments, donc comparable en euros.

Nous en connaissons le coût : environ 12 % du CA classé perdu revient dans les trois mois suivants. Nos marges ne le déduisent pas : ce sont des majorants.

Nous l'appliquons aux clients réguliers, qui font 95 % du chiffre d'affaires. Au test final, la liste de 700 clients prédite par le modèle capte 51 % du CA perdu des réguliers (26 % de tout le churn), contre 7 % pour les 700 plus gros clients ; la rétention reste à mesurer par un pilote. Nous avons aussi testé 41 définitions d'horizon adaptées au rythme de chaque client : à 2 000 clients, elles captent entre 62 et 66 %, aucune ne se détache nettement. Le modèle livré utilise la définition simple. »

## 4. Questions probables

- **« Pourquoi pas un seuil adapté à chaque client ? »** Nous l'avons testé : l'écart de CA capté est petit et dépend du modèle (+0,8 point avec HGB, −0,1 avec XGBoost, notre modèle). Une définition unique est plus simple à expliquer et à piloter.
- **« Et le client qui revient après six mois ? »** Il existe : 12 % du CA classé perdu revient dans les trois mois, 17 % dans les six. Nos marges ne le déduisent pas. Une mesure directe sur 360 jours situe la perte non réalisée plutôt vers 2 % (`07_realized_revenue_loss`), parce que ces clients reviennent tard et peu.
- **« Pourquoi pas douze mois ? »** Nos données couvrent deux ans : douze mois de recul ne laisseraient qu'une photo d'entraînement. Six mois est le plus long horizon que l'on puisse tester sérieusement.
- **« Et le churn partiel ? »** Son potentiel est plus grand (509 M€ contre 125 M€), mais il est moins prévisible (AUC 0,68 contre 0,89) et rien ne dit qu'on sauve ce CA-là. Il n'est pas dans la liste livrée : c'est un second score à tester en pilote.
- **« Votre résultat repose sur une seule période ? »** Oui : un test final en mars 2026, période déjà utilisée dans nos analyses exploratoires. Le backtest sur cinq autres dates donne un résultat proche (50,6 % en moyenne à 700), et le pilote randomisé mesurera aussi la rétention.

**À mettre sur la slide :** « Parti = aucun achat pendant 6 mois. Convention commune à tous les segments, testée contre 41 variantes, sans écart net. »
