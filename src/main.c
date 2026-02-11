#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.c" // Inclut déjà layer.c et matrix.c

int main() {
    srand(time(NULL));

    // 1. Création du réseau : 2 entrées -> 3 cachés -> 1 sortie
    int layers[] = {2, 3, 1};
    Network net = create_network(layers, 3);

    // 2. Données d'entraînement (XOR)
    // Entrées : (0,0), (0,1), (1,0), (1,1)
    Matrix inputs[4];
    Matrix targets[4];

    for(int i=0; i<4; i++) {
        inputs[i] = create_matrix(1, 2, 0);
        targets[i] = create_matrix(1, 1, 0);
    }

    set_element(inputs[0], 0, 0, 0); set_element(inputs[0], 0, 1, 0); set_element(targets[0], 0, 0, 0);
    set_element(inputs[1], 0, 0, 0); set_element(inputs[1], 0, 1, 1); set_element(targets[1], 0, 0, 1);
    set_element(inputs[2], 0, 0, 1); set_element(inputs[2], 0, 1, 0); set_element(targets[2], 0, 0, 1);
    set_element(inputs[3], 0, 0, 1); set_element(inputs[3], 0, 1, 1); set_element(targets[3], 0, 0, 0);

    // 3. Boucle d'entraînement
    int epochs = 10000;
    double learning_rate = 0.1;
    int batch_size = 4; // Mise à jour après avoir vu les 4 exemples
    int batch_counter = 0;

    printf("Entrainement en cours...\n");
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < 4; j++) {
            // On entraîne !
            batch_counter = train_network(&net, inputs[j], targets[j], learning_rate, batch_size, batch_counter);
        }
    }

    // 4. Test et Affichage
    printf("\nResultats apres entrainement :\n");
    Matrix output = create_matrix(1, 1, 0);
    
    for (int i = 0; i < 4; i++) {
        forward_network(net, inputs[i], output);
        double out = get_element(output, 0, 0);
        printf("Entree: %.0f %.0f -> Sortie: %.5f (Cible: %.0f)\n", 
               get_element(inputs[i], 0, 0), get_element(inputs[i], 0, 1), 
               out, get_element(targets[i], 0, 0));
    }

    // 5. Nettoyage
    free_network(&net);
    // + free tes inputs/targets/output...
    
    return 0;
}