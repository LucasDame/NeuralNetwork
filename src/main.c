#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.c"
#include "mnist.c"

// Fonction utilitaire pour trouver l'index de la valeur max (argmax)
int argmax(Matrix m) {
    int max_index = 0;
    double max_val = get_element(m, 0, 0);
    for (int i = 1; i < m.cols; i++) {
        double val = get_element(m, 0, i);
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }
    return max_index;
}

int main() {
    srand(time(NULL));

    // 1. Chargement des données MNIST
    Matrix *train_inputs, *train_targets;
    int train_count;
    
    // Assurez-vous que les chemins sont corrects !
    load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 
               &train_inputs, &train_targets, &train_count);
    train_count = 10000; // Pour accélérer les tests, on limite à 10k images (au lieu de 60k)

    // 2. Création du réseau
    // 784 entrées (pixels) -> 128 cachés -> 10 sorties (chiffres 0-9)
    int layers[] = {784, 128, 10};
    ActivationFunc activations[] = {relu, softmax_placeholder}; // La dernière couche utilisera softmax, gérée par le flag dans Layer
    int use_softmax[] = {0, 1}; // Indique que la deuxième
    Network net = create_network(layers, 3, activations, use_softmax);

    // 3. Paramètres d'entraînement
    int epochs = 3; 
    double learning_rate = 0.01;    
    int batch_size = 32;
    int batch_counter = 0;

    printf("Début de l'entraînement sur %d images...\n", train_count);

    for (int e = 0; e < epochs; e++) {
        double total_error = 0;
        int correct_predictions = 0;

        // Mélanger les données serait mieux ici (Shuffle), mais on fait simple pour l'instant
        for (int i = 0; i < train_count; i++) {
            
            // Entraînement
            batch_counter = train_network(&net, train_inputs[i], train_targets[i], learning_rate, batch_size, batch_counter);

            // Calcul de précision (juste pour l'affichage, on refait un forward rapide)
            // Note: C'est couteux de refaire forward, mais utile pour voir la progression
            Matrix output = create_matrix(1, 10, 0);
            forward_network(net, train_inputs[i], output);
            
            int prediction = argmax(output);
            int target = argmax(train_targets[i]);
            
            if (prediction == target) {
                correct_predictions++;
            }
            free_matrix(&output);

            if (i % 1000 == 0 && i > 0) {
                printf("Epoch %d, Image %d/%d, Précision courante: %.2f%%\r", 
                       e+1, i, train_count, (double)correct_predictions/i * 100.0);
                fflush(stdout);
            }
        }
        printf("\nEpoch %d terminée. Précision finale: %.2f%%\n", e+1, (double)correct_predictions/train_count * 100.0);
    }

    // 4. Test rapide sur une image manuelle (optionnel)
    printf("\nTest sur la première image du set :\n");
    Matrix output = create_matrix(1, 10, 0);
    forward_network(net, train_inputs[0], output);
    for(int k=0; k<10; k++) {
        printf("Chiffre %d: Probabilité %.4f\n", k, get_element(output, 0, k));
    }
    printf("Vrai label: %d\n", argmax(train_targets[0]));

    // Nettoyage (Il faudrait libérer toutes les matrices du dataset aussi...)
    free_network(&net);
    free_matrix(&output);
    free_mnist_data(train_inputs, train_targets, train_count);
    
    return 0;
}