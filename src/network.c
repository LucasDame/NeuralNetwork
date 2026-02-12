#include "layer.c"

typedef struct {
    Layer* layers;
    int num_layers;
} Network;

Network create_network(int* layer_sizes, int num_layers, ActivationFunc* activations) {
    Network net;
    net.num_layers = num_layers - 1;
    net.layers = malloc(net.num_layers * sizeof(Layer));
    if (net.layers == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les couches !\n");
        exit(1);
    }
    for (int i = 0; i < net.num_layers; i++) {
        net.layers[i] = create_layer(layer_sizes[i], layer_sizes[i + 1], activations[i]);
    }
    return net;
}

void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
    net->layers = NULL; 
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

int train_network(Network *net, Matrix input, Matrix target, double learning_rate, int batch_size, int current_batch) {
    // 1. FORWARD PASS
    forward_network(*net, input, net->layers[net->num_layers - 1].activation);

    // 2. BACKWARD PASS
    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer *layer = &net->layers[i];

        // A. Calcul de f'(Z) -> Stocké dans layer->z_prime
        compute_z_prime(layer);

        // B. Calcul du Delta (Erreur locale)
        if (i == net->num_layers - 1) {
            // --- DERNIÈRE COUCHE ---
            // Delta = (Activation - Target) * f'(Z)
            // 1. Calcul (Activation - Target) -> stocké dans error_temp
            substract_matrices(layer->activation, target, layer->error_temp);
            
            // 2. Delta = error_temp * f'(Z) (Produit Hadamard)
            elementwise_multiply_matrix(layer->error_temp, layer->z_prime, layer->delta);
        } else {
            // --- COUCHES CACHÉES ---
            // Delta = (Delta_Next * W_Next_T) * f'(Z)
            // 1. Récupérer la couche suivante
            Layer *next_layer = &net->layers[i + 1];

            // 2. Propager l'erreur : Delta_Next * W_Next_T -> buffer
            multiply_matrices(next_layer->delta, next_layer->t_weights, layer->error_temp); // Assure-toi d'avoir t_weights dans Layer

            // 3. Delta = error_temp * f'(Z)
            elementwise_multiply_matrix(layer->error_temp, layer->z_prime, layer->delta);
        }

        // C. Accumulation des Gradients (Batch)
        // Gradient Poids = Input_Transposé * Delta
        
        // L'entrée de cette couche est soit l'input global, soit l'activation précédente
        Matrix layer_input = (i == 0) ? input : net->layers[i - 1].activation;
        
        // 1. Transposer l'entrée -> t_input
        transpose_matrix(layer_input, layer->t_input); // Assure-toi d'avoir t_input dans Layer
        
        // CORRECTION GRADIENTS :
        multiply_matrices(layer->t_input, layer->delta, layer->buffer); 
        
        // Accumuler dans weight_gradients
        add_matrices(layer->weight_gradients, layer->buffer, layer->weight_gradients);

        // Gradient Biais = Delta (accumulé)
        add_matrices(layer->bias_gradients, layer->delta, layer->bias_gradients);
    }

    double effective_lr = learning_rate / batch_size;
    // 3. MISE A JOUR (Uniquement à la fin du batch)
    if ((current_batch + 1) % batch_size == 0) {
        
        for (int i = 0; i < net->num_layers; i++) {
            Layer *l = &net->layers[i];

            // W = W - (lr * Gradients)
            scalar_multiply_matrix(l->weight_gradients, effective_lr, l->weight_gradients); // Scale les gradients par le learning rate
            substract_matrices(l->weights, l->weight_gradients, l->weights);
            reset_matrix(l->weight_gradients);
            transpose_matrix(l->weights, l->t_weights); // Met à jour la transposée des poids pour la prochaine itération

            // B = B - (lr * Gradients)
            scalar_multiply_matrix(l->bias_gradients, effective_lr, l->bias_gradients); // Scale les gradients par le learning rate
            substract_matrices(l->biases, l->bias_gradients, l->biases);
            reset_matrix(l->bias_gradients);
        }
        return 0; // Reset batch counter
    }

    return current_batch + 1;
}

void print_network(Network net){
for(int i=0; i<net.num_layers; i++){ 
    printf("Couche %d :\n", i); 
    print_layer(net.layers[i]); 
    printf("\n"); 
    }      
}