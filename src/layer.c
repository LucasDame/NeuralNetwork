#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.c"

// Définition de PI 
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Génère un nombre aléatoire selon une loi normale (Moyenne 0, Variance 1)
// Algorithme de Box-Muller
double randn() {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    
    // Éviter log(0)
    if (u1 < 1e-100) u1 = 1e-100;
    
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

typedef double (*ActivationFunc)(double);

static inline double relu(double x) {
    return x > 0 ? x : 0;
}

static inline double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

static inline double softmax_placeholder(double x) {
    // Placeholder pour indiquer que cette couche utilisera softmax, gérée par un flag dans Layer
    return x; 
}

typedef struct {
    Matrix weights;
    Matrix biases;
    
    Matrix z;           // Stockera (Input * Weights + Biases)
    Matrix activation;  // Stockera f(z)

    ActivationFunc func;
    ActivationFunc deriv;

    Matrix delta;
    Matrix weight_gradients;
    Matrix bias_gradients;
    
    Matrix t_weights;
    Matrix t_biases;
    Matrix error_temp;
    Matrix z_prime;
    Matrix t_input;
    Matrix buffer;
    int use_softmax; // Flag pour indiquer si cette couche est une couche de sortie avec softmax
} Layer;

Layer create_layer(int input_size, int output_size, ActivationFunc activation, int use_softmax) {
    Layer layer;
    
    // 1. Allocation
    layer.weights = create_matrix(input_size, output_size, 1); 
    layer.biases = create_matrix(1, output_size, 1); 

    // 2. Initialisation Aléatoire (CRUCIAL pour l'apprentissage)
    // On met des petites valeurs aléatoires entre -1 et 1 pour les poids
    double scale = sqrt(2.0 / input_size);
    for(int i=0; i<input_size*output_size; i++) {
        layer.weights.data[i] = randn() * scale;
    }
    // On initialise les biais à 0
    for(int i=0; i<output_size; i++) {
        layer.biases.data[i] = (activation == relu) ? 0.01 : 0.0; // Un petit biais pour ReLU pour éviter les neurones morts
    }

    layer.func = activation;
    if (activation == relu) {
        layer.deriv = relu_derivative;
    } else if (activation == sigmoid) {
        layer.deriv = sigmoid_derivative;
    } else if (use_softmax) {
        layer.func = softmax_placeholder; // Le vrai softmax est appliqué dans apply_activation
        layer.deriv = NULL; // La dérivée du softmax est gérée différemment pendant la backpropagation
    } else {
        fprintf(stderr, "Activation non supportée !\n");
        exit(1);
    }

    // Allocation des caches (taille 1 pour batch size = 1)
    layer.z = create_matrix(1, output_size, 0); 
    layer.activation = create_matrix(1, output_size, 0); 

    layer.delta = create_matrix(1, output_size, 0); 
    layer.weight_gradients = create_matrix(input_size, output_size, 0); 
    layer.bias_gradients = create_matrix(1, output_size, 0); 

    layer.t_weights = create_matrix(output_size, input_size, 0);
    transpose_matrix(layer.weights, layer.t_weights); // Stocker la transposée des poids pour la backpropagation
    layer.t_biases = create_matrix(output_size, 1, 0);
    layer.error_temp = create_matrix(1, output_size, 0);
    layer.z_prime = create_matrix(1, output_size, 0);
    layer.t_input = create_matrix(input_size, 1, 0);
    layer.buffer = create_matrix(input_size, output_size, 0);
    layer.use_softmax = use_softmax;

    return layer;
}

void free_layer(Layer *layer) {
    free_matrix(&layer->weights);
    free_matrix(&layer->biases);
    free_matrix(&layer->z);
    free_matrix(&layer->activation);
    free_matrix(&layer->weight_gradients);
    free_matrix(&layer->bias_gradients);
    free_matrix(&layer->delta);
    free_matrix(&layer->t_weights);
    free_matrix(&layer->t_biases);
    free_matrix(&layer->error_temp);
    free_matrix(&layer->z_prime);
    free_matrix(&layer->t_input);
    free_matrix(&layer->buffer);
}

void apply_softmax(Layer *layer) {
    // 1. Trouver max (stabilité numérique)
    double max_val = max_matrix(layer->z);
    
    // 2. Calculer exp(z - max) et somme
    double sum = 0.0;
    for (int j = 0; j < layer->z.cols; j++) {
        double val = exp(get_element(layer->z, 0, j) - max_val);
        set_element(layer->activation, 0, j, val);
        sum += val;
    }
    
    // 3. Normaliser
    scalar_multiply_matrix(layer->activation, 1.0 / sum, layer->activation);
}

void apply_activation(Layer *layer) {
    if (layer->use_softmax) {
        apply_softmax(layer);
        return;
    }
    for (int i = 0; i < layer->z.rows; i++) {
        for (int j = 0; j < layer->z.cols; j++) {
            // CORRECTION : On ne rajoute plus le biais ici.
            // On prend la valeur Z qui contient DEJA le biais.
            double z_val = get_element(layer->z, i, j);
            set_element(layer->activation, i, j, layer->func(z_val));
        }
    }
}

void forward_layer(Layer *layer, Matrix input) {
    // 1. Z = Input * Poids
    multiply_matrices(input, layer->weights, layer->z); 

    // 2. Z = Z + Biais (On met à jour Z pour de bon)
    // C'est important que layer->z contienne le biais pour la backpropagation
    add_matrices(layer->z, layer->biases, layer->z);
    
    // 3. A = f(Z)
    apply_activation(layer);
}

void compute_z_prime(Layer *layer) {
    for (int i = 0; i < layer->z.rows; i++) {
        for (int j = 0; j < layer->z.cols; j++) {
            double z_val = get_element(layer->z, i, j);
            set_element(layer->z_prime, i, j, layer->deriv(z_val));
        }
    }
}

void print_layer(Layer layer) { 
    printf("Poids :\n"); 
    print_matrix(layer.weights); 
    printf("Biais :\n"); 
    print_matrix(layer.biases);
}