#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "spline.h"

#define BLKSIZE         56
#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
//#define SPLINE_SIZE 1000
//#define LMAX_1F1 7

double *SPLINE = NULL;
double GRID_A;
double GRID_D;
double FLAPL_S;
int LMAX_1F1;
int SPLINE_SIZE;

int get_1f1_spline_size() {
    return SPLINE_SIZE;
}

inline double i2r_1f1(int i) {
    return GRID_A * (exp(GRID_D * i) - 1);
}

inline double r2i_1f1(double r) {
    return log(r / GRID_A + 1) / GRID_D;
}

/*void fill_1f1_x(double *x, double a, double d, int size) {
    int i;
    SPLINE_SIZE = size;

    for (i = 0; i < SPLINE_SIZE; i++) {
        x[i] = i2r_1f1(i);
        SPLINE[i] = (double) i;
    }
}*/

void free_spline_1f1() {
    if (SPLINE == NULL) {
        return;
    }
    free(SPLINE);
    SPLINE = NULL;
}

int check_1f1_initialization() {
    if (SPLINE == NULL) {
        return 0;
    } else {
        return 1;
    }
}

void initialize_spline_1f1(double *f, double a, double d,
                           int size, int lmax, double s) {
    if (SPLINE != NULL) {
        free_spline_1f1();
    }
    LMAX_1F1 = lmax;
    FLAPL_S = s;
    SPLINE_SIZE = size;
    GRID_D = d;
    GRID_A = a;
    SPLINE = (double *) calloc((LMAX_1F1 + 1) * 4 * SPLINE_SIZE, sizeof(double));
    double *tmp_spline = (double *) calloc(5 * SPLINE_SIZE, sizeof(double));
    double *my_spline;
    double *x = (double *) calloc(SPLINE_SIZE, sizeof(double));
    int i, l;
    for (i = 0; i < SPLINE_SIZE; i++) {
        x[i] = i;
    }
    for (l = 0; l <= LMAX_1F1; l++) {
        get_cubic_spline_coeff(x, f + l * SPLINE_SIZE,
                               tmp_spline, SPLINE_SIZE);
        my_spline = SPLINE + l * SPLINE_SIZE * 4;
        for (i = 0; i < SPLINE_SIZE; i++) {
            my_spline[4 * i + 0] = tmp_spline[1 * SPLINE_SIZE + i];
            my_spline[4 * i + 1] = tmp_spline[2 * SPLINE_SIZE + i];
            my_spline[4 * i + 2] = tmp_spline[3 * SPLINE_SIZE + i];
            my_spline[4 * i + 3] = tmp_spline[4 * SPLINE_SIZE + i];
        }
    }
    free(x);
    free(tmp_spline);
}

inline double eval_1f1(double r, double *my_spline) {
    double di = r2i_1f1(r);
    int i = (int) di;
    di -= i;
    //printf("i %d %f   %f %f %f %f\n", i, di, my_spline[i*4+0], my_spline[i*4+1], my_spline[i*4+2], my_spline[i*4+3]);
    return my_spline[i*4+0] + di * (my_spline[i*4+1]
           + di * (my_spline[i*4+2] + di * my_spline[i*4+3]));
}

void vec_eval_1f1(double *f, double *r, int n, int l) {
    double *my_spline = SPLINE + l * 4 * SPLINE_SIZE;
    int i;
    for (i = 0; i < n; i++) {
        f[i] = eval_1f1(r[i], my_spline);
    }
}

int GTOcontract_flapl0(double *ectr, double *coord, double *alpha, double *coeff,
                       int l, int nprim, int nctr, size_t ngrids, double fac)
{
    size_t i, j, k;
    double arr, eprim;
    double rr[BLKSIZE];
    double *gridx = coord;
    double *gridy = coord+BLKSIZE;
    double *gridz = coord+BLKSIZE*2;
    double sqrt_alpha;
    //fac = fac * 4 * atan(1.0) * tgamma(2 + l) / tgamma(1.5 + l);
    if (l > LMAX_1F1) {
        printf("l value too high! %d %d\n", l, LMAX_1F1);
        exit(-1);
    }
    double *my_spline = SPLINE + l * 4 * SPLINE_SIZE;
    double rmax = i2r_1f1(SPLINE_SIZE - 1);

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
    }

    for (i = 0; i < nctr*BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        sqrt_alpha = pow(alpha[j], FLAPL_S);
        //sqrt_alpha = sqrt(alpha[j]);
        for (i = 0; i < ngrids; i++) {
            arr = MIN(alpha[j] * rr[i], rmax);
            eprim = eval_1f1(arr, my_spline) * sqrt_alpha * fac;
            for (k = 0; k < nctr; k++) {
                ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
            }
        }
    }
    return 1;
}


