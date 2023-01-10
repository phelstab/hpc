#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {

    // define 2-dimensional float array with malloc
    //float **array = (float **)malloc(10 * sizeof(float *));
    double ** arr = malloc(20*sizeof(double));
    for (int i = 0; i < 20; i++) {
        arr[i] = (double *)malloc(10 * sizeof(double));
    }

    // fill array with values
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) {
            arr[i][j] = i * j;
        }
    }

    // print array
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) {
            printf("Test: %f ", arr[i][j]);
        }
    }

    // free memory
    free(arr);

    return 0;
}



