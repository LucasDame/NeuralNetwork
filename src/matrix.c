#ifndef MATRIX_c
#define MATRIX_c

#include <stdio.h>
#include <stdlib.h>


typedef struct {
    int rows;
    int cols;
    double* data;
} Matrix;

Matrix create_matrix(int rows, int cols, int initialisation) {
    Matrix mat = {rows, cols, NULL};
    if (initialisation == 0) {
        mat.data = calloc(rows * cols, sizeof(double));
    } else if (initialisation == 1) {
        mat.data = malloc(rows * cols * sizeof(double));
    }
    if (mat.data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire !\n");
        exit(1); 
    }
    return mat;
}

static inline double get_element(Matrix mat, int row, int col){
    return mat.data[row * mat.cols + col];
}

static inline void set_element(Matrix mat, int row, int col, double value){
    mat.data[row * mat.cols + col] = value;
}

static inline void free_matrix(Matrix *mat) {
    if (mat->data) {
        free(mat->data);
        mat->data = NULL; // Bonne pratique pour éviter les "double free"
    }
}

void copy_matrix(Matrix source, Matrix dest) {
    if (source.rows != dest.rows || source.cols != dest.cols) {
        fprintf(stderr, "Matrices de tailles différentes !\n");
        exit(1);
    }
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            set_element(dest, i, j, get_element(source, i, j));
        }
    }
}

void add_matrices(Matrix a, Matrix b, Matrix result) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "Matrices de tailles différentes !\n");
        exit(1);
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            double sum = get_element(a, i, j) + get_element(b, i, j);
            set_element(result, i, j, sum);
        }
    }
}

void substract_matrices(Matrix a, Matrix b, Matrix result) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "Matrices de tailles différentes !\n");
        exit(1);
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            double diff = get_element(a, i, j) - get_element(b, i, j);
            set_element(result, i, j, diff);
        }
    }
}

void multiply_matrices(Matrix a, Matrix b, Matrix result) {
    if (a.cols != b.rows) {
        fprintf(stderr, "Matrices incompatibles pour la multiplication !\n");
        fprintf(stderr, "A: %dx%d, B: %dx%d\n", a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a.cols; k++) {
                sum += get_element(a, i, k) * get_element(b, k, j);
            }
            set_element(result, i, j, sum);
        }
    }
}

void transpose_matrix(Matrix mat, Matrix result) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            set_element(result, j, i, get_element(mat, i, j));
        }
    }
}

void scalar_multiply_matrix(Matrix mat, double scalar, Matrix result) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            double val = get_element(mat, i, j) * scalar;
            set_element(result, i, j, val);
        }
    }
}

void elementwise_multiply_matrix(Matrix a, Matrix b, Matrix result) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "Matrices de tailles différentes !\n");
        exit(1);
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            double val = get_element(a, i, j) * get_element(b, i, j);
            set_element(result, i, j, val);
        }
    }
}

void reset_matrix(Matrix mat) {
    for(int i=0; i < mat.rows * mat.cols; i++) {
        mat.data[i] = 0.0;
    }
}

void print_matrix(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            printf("%.5f ", get_element(mat, i, j));
        }
        printf("\n");
    }
}

double max_matrix(Matrix mat) {
    double max_val = get_element(mat, 0, 0);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            double val = get_element(mat, i, j);
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    return max_val;
}

double min_matrix(Matrix mat) {
    double min_val = get_element(mat, 0, 0);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            double val = get_element(mat, i, j);
            if (val < min_val) {
                min_val = val;
            }
        }
    }
    return min_val;
}

#endif