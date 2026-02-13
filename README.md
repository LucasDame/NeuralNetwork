# Neural Network from Scratch in C

Un réseau de neurones entièrement implémenté en C pur, sans dépendances externes. Ce projet éducatif vise à comprendre les fondamentaux du deep learning en construisant tout de zéro.

## 📋 Table des matières

- [Caractéristiques](#caractéristiques)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Exemple : XOR](#exemple--xor)
- [Test sur MNIST](#test-sur-mnist)
- [Mathématiques](#mathématiques)
- [Roadmap](#roadmap)
- [Contribuer](#contribuer)

## ✨ Caractéristiques

- **Pur C** : Aucune dépendance externe (pas de bibliothèques ML)
- **Architecture flexible** : Nombre de couches et neurones configurables
- **Fonctions d'activation** : ReLU et Sigmoid
- **Backpropagation** : Implémentation complète de la rétropropagation du gradient
- **Mini-batch training** : Support pour l'entraînement par batch
- **Optimisé** : Utilisation de matrices pour des calculs efficaces

## 📁 Structure du projet

```
.
├── src/
│   ├── matrix.c        # Opérations matricielles (multiplication, transposition, etc.)
│   ├── layer.c         # Définition et opérations sur une couche de neurones
│   ├── network.c       # Gestion du réseau multicouche
│   └── main.c          # Point d'entrée et exemples
├── Makefile            # Compilation du projet
└── README.md           # Ce fichier
```

## 🚀 Installation

### Prérequis

- GCC (ou tout compilateur C compatible)
- Make
- Linux/macOS/WSL (non testé sur Windows natif)

### Compilation

```bash
# Compiler le projet
make

# Exécuter le programme
./neural_network
```

### Nettoyage

```bash
make clean
```

## 💻 Utilisation

### Exemple basique (XOR)

Le fichier `main.c` contient actuellement un exemple d'entraînement sur le problème XOR :

```c
// Création du réseau : 2 entrées -> 3 neurones cachés -> 1 sortie
int layers[] = {2, 3, 1};
Network net = create_network(layers, 3);

// Données d'entraînement
// Input: (0,0) -> Output: 0
// Input: (0,1) -> Output: 1
// Input: (1,0) -> Output: 1
// Input: (1,1) -> Output: 0

// Entraînement
int epochs = 10000;
double learning_rate = 0.1;
train_network(&net, inputs, targets, learning_rate, batch_size, batch_counter);
```

### Résultats attendus

Après 10 000 époques, le réseau devrait produire :

```
Resultats apres entrainement :
Entree: 0 0 -> Sortie: 0.02145 (Cible: 0)
Entree: 0 1 -> Sortie: 0.97823 (Cible: 1)
Entree: 1 0 -> Sortie: 0.98012 (Cible: 1)
Entree: 1 1 -> Sortie: 0.03421 (Cible: 0)
```

## 🏗️ Architecture

### Composants principaux

#### 1. **Matrix** (`matrix.c`)
Gestion des opérations matricielles :
- `create_matrix()` : Allocation de matrices
- `multiply_matrices()` : Multiplication matricielle
- `transpose_matrix()` : Transposition
- `add_matrices()` : Addition
- `elementwise_multiply_matrix()` : Produit de Hadamard
- `substract_matrix()` : soustraction matricielle
- `scalar_multiply_matrix()` : Multiplication par un scalaire
- `reset_matrix()` : Réinitialisation à zéro
- `free_matrix()` : Libération de la mémoire
- `print_matrix()` : Affichage d'une matrice (pour le debug)
- `get_element()` : Accès à un élément spécifique
- `set_element()` : Modification d'un élément spécifique
- `copy_matrix()` : Copie d'une matrice
- `free_matrix()` : Libération de la mémoire d'une matrice

#### 2. **Layer** (`layer.c`)
Représentation d'une couche de neurones :
```c
typedef struct {
    Matrix weights;     // Poids de la couche
    Matrix biases;      // Biais de la couche
    
    Matrix z;           // Stockera (Input * Weights + Biases)
    Matrix activation;  // Stockera f(z)

    ActivationFunc func;    // Fonction d'activation
    ActivationFunc deriv;   // Dérivée de la fonction d'activation

    Matrix delta;            // Erreur locale (backprop)
    Matrix weight_gradients; // Gradients des poids
    Matrix bias_gradients;   // Gradients des biais
    
    Matrix t_weights;    // Transposée des poids
    Matrix t_biases;     // Transposée des biais
    Matrix error_temp;   // Buffer pour calculs intermédiaires
    Matrix z_prime;      // Dérivée de z
    Matrix t_input;      // Transposée de l'entrée
    Matrix buffer;       // Buffer pour calculs intermédiaires
} Layer;
```

**Opérations** :
- `forward_layer()` : Propagation avant (z = input × W + b, a = f(z))
- `compute_z_prime()` : Calcul de la dérivée de l'activation
- `apply_activation` : Application de la fonction d'activation
- `create_layer()` : Initialisation d'une couche
- `free_layer()` : Libération de la mémoire d'une couche
- `relu()` : Fonction d'activation ReLU 
- `relu_derivative()` : Dérivée de ReLU 
- `sigmoid()` : Fonction d'activation Sigmoid 
- `sigmoid_derivative()` : Dérivée de Sigmoid

#### 3. **Network** (`network.c`)
Gestion du réseau complet :
```c
typedef struct {
    Layer* layers;
    int num_layers;
} Network;
```

**Opérations** :
- `forward_network()` : Propagation avant complète
- `train_network()` : Entraînement (forward + backward + update)
- `create_network()` : Initialisation du réseau
- `free_network()` : Libération de la mémoire du réseau

### Fonctions d'activation

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = 1 si x > 0, sinon 0
```

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```

## 📐 Mathématiques

### Forward Pass

Pour une couche $l$ :

$$z^{[l]} = a^{[l-1]} W^{[l]} + b^{[l]}$$

$$a^{[l]} = f(z^{[l]})$$

Où :
- $a^{[l-1]}$ : activation de la couche précédente
- $W^{[l]}$ : matrice des poids
- $b^{[l]}$ : vecteur de biais
- $f$ : fonction d'activation

### Backward Pass

#### Dernière couche :
$$\delta^{[L]} = (a^{[L]} - y) \odot f'(z^{[L]})$$

#### Couches cachées :
$$\delta^{[l]} = (\delta^{[l+1]} W^{[l+1]T}) \odot f'(z^{[l]})$$

### Gradients :
$$\frac{\partial L}{\partial W^{[l]}} = (a^{[l-1]})^T \delta^{[l]}$$

$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$

### Mise à jour (Gradient Descent) :
$$W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$$

$$b^{[l]} := b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$$

Où $\alpha$ est le learning rate et $\odot$ désigne le produit de Hadamard (élément par élément).

## 🗺️ Roadmap

### ✅ Implémenté
- [x] Opérations matricielles de base
- [x] Couches fully-connected
- [x] Fonctions d'activation (ReLU, Sigmoid)
- [x] Forward propagation
- [x] Backpropagation
- [x] Mini-batch training
- [x] Exemple XOR fonctionnel

### 🚧 En cours / À venir
- [ ] Parser MNIST
- [ ] Fonction de perte (Cross-Entropy)
- [ ] Métriques (accuracy, loss)
- [ ] Sauvegarde/chargement de modèles
- [ ] Optimiseurs (Adam, RMSprop)
- [ ] Dropout pour la régularisation
- [ ] Batch Normalization
- [ ] Couches convolutionnelles (CNN)
- [ ] Interface CLI pour configurer le réseau
- [ ] Visualisation de l'apprentissage
- [ ] Support GPU (optionnel)

## 🐛 Bugs connus

- **Initialisation** : L'initialisation des poids pourrait être améliorée (Xavier/He initialization)
- **Overflow** : Pas de protection contre les valeurs numériques extrêmes

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Idées de contributions
- Optimiser les performances
- Ajouter des tests unitaires
- Améliorer la documentation

## 📚 Ressources

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

⭐ Si ce projet vous a aidé, n'hésitez pas à mettre une étoile !