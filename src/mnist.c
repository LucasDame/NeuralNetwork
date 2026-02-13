#include <stdio.h>
#include <stdlib.h>
#include "matrix.c" 

// MNIST stocke les entiers en Big Endian (inversé par rapport aux ordis modernes)
// Cette fonction remet les octets dans le bon ordre.
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void load_mnist(const char *image_filename, const char *label_filename, Matrix **inputs, Matrix **targets, int *count) {
    FILE *image_file = fopen(image_filename, "rb");
    FILE *label_file = fopen(label_filename, "rb");

    if (!image_file || !label_file) {
        fprintf(stderr, "Erreur: Impossible d'ouvrir les fichiers MNIST.\n");
        exit(1);
    }

    // --- Lecture des en-têtes ---
    int magic_number, num_items, rows, cols;
    
    // Fichier images
    fread(&magic_number, sizeof(int), 1, image_file);
    fread(&num_items, sizeof(int), 1, image_file);
    fread(&rows, sizeof(int), 1, image_file);
    fread(&cols, sizeof(int), 1, image_file);
    
    num_items = reverse_int(num_items); // Doit être 60000 pour le train set
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    // Fichier labels
    int label_magic, label_count;
    fread(&label_magic, sizeof(int), 1, label_file);
    fread(&label_count, sizeof(int), 1, label_file);
    label_count = reverse_int(label_count);

    *count = num_items;
    *count = 10000; // Limiter à 10k pour les tests rapides
    printf("Chargement de %d images (%dx%d)...\n", *count, rows, cols);

    // --- Allocation des tableaux de matrices ---
    // Note : Pour un vrai gros projet, on chargerait par batch pour économiser la RAM.
    // Ici, 60k images * 784 doubles ~= 370 Mo de RAM, ça passe.
    *inputs = malloc(*count * sizeof(Matrix));
    *targets = malloc(*count * sizeof(Matrix));

    unsigned char *image_buffer = malloc(rows * cols);
    unsigned char label_byte;

    for (int i = 0; i < *count; i++) {
        // 1. Lire l'image brute (pixels 0-255)
        fread(image_buffer, sizeof(unsigned char), rows * cols, image_file);
        
        // Créer la matrice d'entrée (1 ligne, 784 colonnes)
        (*inputs)[i] = create_matrix(1, rows * cols, 0);
        for (int p = 0; p < rows * cols; p++) {
            // Normalisation : 0-255 -> 0.0-1.0
            set_element((*inputs)[i], 0, p, (double)image_buffer[p] / 255.0);
        }

        // 2. Lire le label (chiffre 0-9)
        fread(&label_byte, sizeof(unsigned char), 1, label_file);
        
        // Créer la cible (One-Hot Encoding)
        // Le chiffre 3 devient [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        (*targets)[i] = create_matrix(1, 10, 0); // Init à 0
        set_element((*targets)[i], 0, label_byte, 1.0);
    }

    free(image_buffer);
    fclose(image_file);
    fclose(label_file);
    printf("Chargement terminé.\n");
}

void free_mnist_data(Matrix *inputs, Matrix *targets, int count) {
    for (int i = 0; i < count; i++) {
        free_matrix(&inputs[i]);
        free_matrix(&targets[i]);
    }
    free(inputs);
    free(targets);
}