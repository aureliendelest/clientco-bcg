# Glossaire

> **Rôle** : Le vocabulaire du projet, défini une fois.
> **Pour** : Tout le monde.
> **À jour au** : 25/11/2026. Dates : les notebooks et `03_evidence.md` utilisent les dates d'origine du dataset (22/09/2022 au 22/09/2024) ; le code, les configs et les décisions utilisent les dates décalées de +2 ans (la photo de test du 22/03/2024 devient 22/03/2026).

## Vocabulaire du projet

| Terme | Sens dans ce projet |
| --- | --- |
| **Photo (snapshot)** | Une table à une date T : une ligne par client actif sur les 180 derniers jours, avec ses variables (calculées **strictement avant T**), son label (parti ou non **après T**), la vérité à 180 jours (`gone`), son horizon, sa valeur V et son segment. |
| **Label (A, B)** | 1 = client perdu pour l'entraînement : A = pas d'achat pendant 180 jours ; B = pas d'achat dans son horizon `H = clip(3 × p90, 45, 90)`. |
| **Vérité** | « Parti » = aucun jour d'achat pendant 180 jours : sert à évaluer, quel que soit le label d'entraînement. |
| **Réguliers** | ≥ 6 jours d'achat sur 180 jours, écart médian ≤ 30 jours : le périmètre du modèle. |
| **H (horizon)** | Nombre de jours sans achat au bout duquel on déclare le client perdu. |
| **Jour d'achat** | Un couple (client, `date_order`) dont le CA net des retours du jour est > 0,01 €. |
| **Récence** | Jours depuis le dernier achat. **Récence relative** = récence ÷ écart p90 du client. |
| **Écart (gap)** | Nombre de jours entre deux jours d'achat consécutifs d'un client. |
| **V** | CA que le client générerait s'il restait. |
| **VE** | Valeur espérée = P × V. |
| **K** | Nombre de clients que les commerciaux peuvent traiter (capacité) : ≈ 700 par liste mensuelle à l'échelle du dataset. |
| **CA capté @K** | Part du CA réellement perdu qui se trouve dans les K premiers clients de la liste. |
| **Délai de label** | Temps nécessaire pour connaître le label d'une photo (180 jours pour A, 90 pour B) : les photos d'entraînement s'arrêtent ce délai avant la date de test, pour que le modèle ne « voie » pas le futur. |
| **Fuite temporelle** | Quand le modèle utilise une information postérieure à T. Une AUC > 0,95 en est presque toujours le signe. |
| **Calibration** | Faire en sorte qu'une probabilité prédite de x % corresponde à x % de départs observés. |
| **Run / sweep / collect** | Une expérience / une série d'expériences / la fusion des runs dans `results/`. |
