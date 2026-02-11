#include "layer.c"

typedef struct {
    Layer* layers;
    int num_layers;
} Network;

Network create_network(int* layer_sizes, int num_layers) {
    Network net;
    net.num_layers = num_layers - 1; // Le nombre de couches est le nombre de tailles - 1
    net.layers = malloc(net.num_layers * sizeof(Layer));
    if (net.layers == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les couches !\n");
        exit(1);
    }
    for (int i = 0; i < net.num_layers; i++) {
        net.layers[i] = create_layer(layer_sizes[i], layer_sizes[i + 1], relu);
    }
    return net;
}

void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
    net->layers = NULL; // Bonne pratique pour éviter les "double free"
}

void forward_network(Network net, Matrix input, Matrix output) {
    // Étape 1 : La première couche reçoit l'entrée externe du programme
    forward_layer(&net.layers[0], input);

    // Étape 2 : Les couches suivantes reçoivent l'activation de la précédente
    for (int i = 1; i < net.num_layers; i++) {
        // L'entrée de la couche [i] est la sortie (activation) de la couche [i-1]
        forward_layer(&net.layers[i], net.layers[i-1].activation);
    }

    // Étape 3 : Copier le résultat de la dernière couche vers la sortie utilisateur
    copy_matrix(net.layers[net.num_layers - 1].activation, output);
}

void train_network(Network *net, Matrix input, Matrix target, double learning_rate) {
    // 1. Forward pass
    forward_network(*net, input, target); // On peut réutiliser target comme "output" temporaire

    // 2. BACKWARD PASS
    // On parcourt les couches de la fin (output) vers le début (input)
    for (int i = net.num_layers - 1; i >= 0; i--) {
        Layer *layer = &net.layers[i];
        Layer *next_layer = (i < net.num_layers - 1) ? &net.layers[i + 1] : NULL;
        Layer *prev_layer = (i > 0) ? &net.layers[i - 1] : NULL;
        
        // --- A. Calcul du Delta (l'erreur du neurone) ---
        if (next_layer == NULL) {
            // CAS 1 : Couche de Sortie
            // Erreur = (Sortie - Cible) * f'(Z)
            // Note : Pour MSE, la dérivée de l'erreur est (A - Y)
            for (int r = 0; r < layer->delta.rows; r++) {
                for (int c = 0; c < layer->delta.cols; c++) {
                    double a = get_element(layer->activation, r, c);
                    double y = get_element(target, r, c);
                    double z = get_element(layer->z, r, c);
                    
                    // Calcul du delta : (A - Y) * f'(Z)
                    double delta_val = (a - y) * layer->deriv(z);
                    set_element(layer->delta, r, c, delta_val);
                }
            }
        } else {
            // CAS 2 : Couche Cachée
            // Erreur = (Poids_Next_Transposés * Delta_Next) * f'(Z)
            
            // Étape critique : On transpose les poids de la couche suivante
            // (Supposons que tu aies ajouté ce buffer dans la struct Layer)
            transpose_matrix(next_layer->weights, layer->weights_transposed_cache);
            
            // On propage l'erreur en arrière
            multiply_matrices(next_layer->delta, layer->weights_transposed_cache, layer->delta); // Erreur brute
            
            // On multiplie par la dérivée de l'activation : * f'(Z)
            for (int r = 0; r < layer->delta.rows; r++) {
                for (int c = 0; c < layer->delta.cols; c++) {
                    double z = get_element(layer->z, r, c);
                    double current_delta = get_element(layer->delta, r, c);
                    set_element(layer->delta, r, c, current_delta * layer->deriv(z));
                }
            }
        }

        // --- B. Calcul des Gradients ---
        // dW = Input_Transposé * Delta
        // Si c'est la première couche, l'input est l'entrée du réseau.
        // Sinon, l'input est l'activation de la couche précédente.
        Matrix layer_input = (i == 0) ? input : prev_layer->activation;
        
        // Astuce : Pour éviter d'allouer une matrice pour Input_Transposé,
        // on peut faire une multiplication manuelle ou ajouter un cache.
        // Pour l'instant, supposons que tu utilises une fonction multiply_transpose_A(Input, Delta, dW)
        // Ou que tu transposes l'input dans un cache temporaire.
        
        // Supposons un cache 'input_transposed' dans Layer pour l'exemple :
        transpose_matrix(layer_input, layer->input_transposed_cache);
        multiply_matrices(layer->input_transposed_cache, layer->delta, layer->weight_gradients);

        // db = Delta (pour un batch size de 1)
        copy_matrix(layer->delta, layer->bias_gradients);
    }

    // 3. MISE À JOUR DES POIDS (Gradient Descent)
    // W = W - learning_rate * dW
    for (int i = 0; i < net.num_layers; i++) {
        Layer *l = &net.layers[i];
        
        for (int j = 0; j < l->weights.rows * l->weights.cols; j++) {
            l->weights.data[j] -= learning_rate * l->weight_gradients.data[j];
        }
        
        for (int j = 0; j < l->biases.rows * l->biases.cols; j++) {
            l->biases.data[j] -= learning_rate * l->bias_gradients.data[j];
        }
    }
}