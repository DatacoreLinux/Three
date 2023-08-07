#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <SDL2/SDL.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *e;

} Matrix;

#define MATRIX_ENTRY(m, row, col) m.e[row * m.cols + col]


float sigmoid(float x)
{
    return (1.0f / ( 1.0f + expf(-x) ) );
}

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

Matrix matrix_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.e = malloc(rows * cols * sizeof(float));

    return m;
}

void matrix_fill(Matrix m)
{
    float count = 2.0f;
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_ENTRY(m, i, j) = count;
            count++;
        }
    }

    return;
}

void matrix_rand(Matrix m)
{
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_ENTRY(m, i, j) = rand_float();
        }
    }

    return;
}

void matrix_print(Matrix m)
{
    
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            printf("%f, ", MATRIX_ENTRY(m, i, j) );
        }
        printf("\n");
    }

    return;
}

void matrix_sum(Matrix dst, Matrix src)
{
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t j = 0; j < dst.cols; j++) {
            MATRIX_ENTRY(dst, i, j) += MATRIX_ENTRY(src, i, j);
        }
    }

    return;
}

void matrix_dot(Matrix act, Matrix x, Matrix w)
{
    for(size_t i = 0; i <= act.cols; i++) {
        MATRIX_ENTRY(act, i, 0) = 0;
    }

    for(size_t i = 0; i < w.rows; i++) {
        for(size_t j = 0; j < w.cols; j++) {
            MATRIX_ENTRY(act, i, 0) += MATRIX_ENTRY(x, j, 0) * MATRIX_ENTRY(w, i, j);
        }
    }

    return;
}

void matrix_sig(Matrix m)
{
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_ENTRY(m, i, j) = sigmoid(MATRIX_ENTRY(m, i, j));
        }
    }

    return;
}
