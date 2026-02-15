# 📊 Application Gestion des Risques - ISSEA

Application Streamlit interactive pour le devoir de Gestion des Risques sur les modèles d'arbres multinomiaux.

## 🎯 Objectif

Cette application permet d'analyser les mesures de risque (VaR, ES) dans le cadre :
- Des arbres multinomiaux (simulations)
- Des données réelles de marchés financiers
- De l'étude de la diversification de portefeuille

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

## 📁 Structure de l'Application

### Onglet 1 : Actif Unique
- **1.1 Étude Théorique** : Calculs de E[Sn], Var[Sn] pour différents horizons
- **1.2 Simulations** : 10,000 trajectoires Monte Carlo sur 252 jours
- **1.3 Mesures de Risque** : VaR, ES, probabilité de perte

### Onglet 2 : Portefeuille 2 Actifs
- **2.1 Modèle de Dépendance** : Construction de matrices de probabilités conjointes
- **2.2 Portefeuille Équipondéré** : Analyse avec différentes corrélations
- **2.3 Backtesting** : Test de Kupiec et analyse du clustering

### Onglet 3 : Données Réelles
- **3.1 Analyse Exploratoire** : Timeline des crises, choix de période
- **3.2 Rendements** : Calcul et analyse des rendements simples et log-rendements
- **3.3 Corrélations** : Matrices de corrélation et évolution temporelle
- **3.4 Mesures de Risque** : VaR historique et paramétrique, backtesting
- **3.5 Diversification** : Étude empirique sur portefeuilles variés

### Onglet 4 : Synthèse
- Questions de réflexion et analyses critiques
- Recommandations managériales

### Onglet 5 : À Propos
- Informations sur le projet
- Objectifs pédagogiques
- Conseils et consignes

## 📊 Fonctionnalités Principales

### Partie 1 : Modèle Multinomial
- Calculs théoriques automatiques
- Simulation Monte Carlo paramétrable
- Visualisations interactives (à compléter avec echarts)
- Comparaison théorique vs empirique

### Partie 2 : Corrélation
- Construction de matrices de corrélation
- Simulations conjointes
- Analyse de l'impact de la diversification
- Backtesting avec test de Kupiec

### Partie 3 : Données Réelles
- Upload de fichiers CSV
- Analyse de 7 actifs sur 3 secteurs
- Calcul de VaR historique et paramétrique
- Tests de normalité
- Étude de corrélation

## 🎨 Design

L'application utilise un design professionnel et distinctif avec :
- Typographie élégante (Playfair Display + Source Sans 3)
- Palette de couleurs sophistiquée
- Animations et transitions fluides
- Interface responsive
- Visualisations interactives

## 📝 Prochaines Étapes de Développement

Pour compléter l'application, il faut implémenter :

1. **Visualisations avec streamlit-echarts** :
   - Trajectoires de prix (échelle log)
   - Histogrammes de distributions
   - Graphiques d'évolution temporelle
   - Heatmaps de corrélation
   - Graphiques de backtesting

2. **Calculs des mesures de risque** :
   - Implémentation manuelle de VaR (historique, paramétrique)
   - Calcul de l'Expected Shortfall
   - Extrapolation √T pour différents horizons
   - Test de Kupiec

3. **Simulations** :
   - Simulation des trajectoires multinomiales
   - Génération de matrices de corrélation
   - Simulations conjointes pour 2 actifs
   - Backtesting sur 252 jours

4. **Traitement des données réelles** :
   - Lecture et parsing des CSV
   - Calcul des rendements
   - Tests statistiques (Shapiro-Wilk, Jarque-Bera)
   - Matrices de corrélation

## ⚠️ Notes Importantes

- **Pas de packages tout-faits** : VaR et ES doivent être implémentés manuellement
- **Code commenté** : Chaque fonction doit être documentée
- **Reproductibilité** : Utiliser des seeds aléatoires
- **Interprétation** : Les graphiques doivent être accompagnés d'analyses

## 🤝 Contribution

Ce projet est un devoir académique pour l'ISSEA.  
Groupe : [Votre Nom] et [Nom du Binôme]  
Année : 2025-2026

## 📧 Contact

**Enseignant** : NOUMEDEM Boris  
**Institution** : ISSEA - Option Finance et Actuariat  
**Date limite** : 08/02/2026 23h59

---

*Bonne chance pour votre projet ! 🍀*
