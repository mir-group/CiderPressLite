#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fblas.h"
#include "nr_cider_numint.h"

void dset0(double *p, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                p[i] = 0;
        }
}

int VXCao_empty_blocks_cider(char *empty, unsigned char *non0table, int *shls_slice,
                       int *ao_loc)
{
        if (non0table == NULL || shls_slice == NULL || ao_loc == NULL) {
                return 0;
        }

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        int bas_id;
        int box_id = 0;
        int bound = BOXSIZE;
        int has0 = 0;
        empty[box_id] = 1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                if (ao_loc[bas_id] == bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = 1;
                }
                empty[box_id] &= !non0table[bas_id];
                if (ao_loc[bas_id+1] > bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
        }
        return has0;
}

/* vv[n,m] = ao1[n,ngrids] * ao2[m,ngrids] */
static void dot_ao_coef(double *vv, double *ao, double *coef,
                        int nao, int ncoef, int ngrids, int bgrids,
                        unsigned char *non0table, int *shls_slice, int *ao_loc)
{
    int nbox = (nao+BOXSIZE-1) / BOXSIZE;
    char empty[nbox];
    int has0 = VXCao_empty_blocks_cider(empty, non0table, shls_slice, ao_loc);

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    if (has0) {
        int ib, leni;
        size_t b0i;
        for (ib = 0; ib < nbox; ib++) {
        if (!empty[ib]) {
            b0i = ib * BOXSIZE;
            leni = MIN(nao-b0i, BOXSIZE);
            dgemm_(&TRANS_T, &TRANS_N, &ncoef, &leni, &bgrids, &D1,
                   coef, &ngrids, ao+b0i*ngrids, &ngrids,
                   &D1, vv+b0i*ncoef, &ncoef);
        } }
    } else {
        dgemm_(&TRANS_T, &TRANS_N, &ncoef, &nao, &bgrids,
               &D1, coef, &ngrids, ao, &ngrids, &D1, vv, &ncoef);
    }
}


/* vv[nao,nao] = ao1[i,nao] * ao2[i,nao] */
void VXCdot_ao_coef(double *vv, double *ao, double *coef,
                    int nao, int ncoef, int ngrids, int nbas,
                    unsigned char *non0table, int *shls_slice, int *ao_loc)
{
    const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
    size_t Nao = nao;
    size_t Ncoef = ncoef;
    dset0(vv, Nao * Ncoef);

#pragma omp parallel
{
    int ip, ib;
    double *v_priv = calloc(Nao*Ncoef+2, sizeof(double));
#pragma omp for nowait schedule(static)
    for (ib = 0; ib < nblk; ib++) {
        ip = ib * BLKSIZE;
        dot_ao_coef(v_priv, ao+ip, coef+ip,
                    nao, ncoef, ngrids, MIN(ngrids-ip, BLKSIZE),
                    non0table+ib*nbas, shls_slice, ao_loc);
    }
#pragma omp critical
    {
        for (ip = 0; ip < Nao*Ncoef; ip++) {
            vv[ip] += v_priv[ip];
        }
    }
    free(v_priv);
}
}