#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nr_cider_numint.h"

// 1k grids per block
#define GRIDS_BLOCK     512

void VXCgen_grid_cider(double *out, double *coords, double *atm_coords,
                       double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *g = malloc(sizeof(double) * GRIDS_BLOCK);
        size_t ig0, n, ngs;
        double fac;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }

                        for (n = 0; n < ngs; n++) {
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                                buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(g);
        free(buf);
        free(grid_dist);
}
        free(atom_dist);
}


// 1k grids per block
#define GRIDS_BLOCK     512

void VXCgen_grid_cider_sm(double *out, double *coords, double *atm_coords,
                          double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *g = malloc(sizeof(double) * GRIDS_BLOCK);
        size_t ig0, n, ngs;
        double fac;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }
                        //for (n = 0; n < ngs; n++) {
                        //        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        //}
                        for (n = 0; n < ngs; n++) {
                                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                                buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(g);
        free(buf);
        free(grid_dist);
}
        free(atom_dist);
}

void VXCgen_grid_cider_exp(double *out, double *coords, double *atm_coords,
                           double *radii, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i;
        double dx, dy, dz;

#pragma omp parallel private(i, dx, dy, dz)
{
        double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *g = malloc(sizeof(double) * GRIDS_BLOCK);
        size_t ig0, n, ngs;
        double dr;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
            ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
            for (i = 0; i < natm; i++) {
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
            } }

            for (i = 0; i < natm; i++) {
                for (n = 0; n < ngs; n++) {
                    dr = grid_dist[i*GRIDS_BLOCK+n] / radii[i];
                    out[i*Ngrids+ig0+n] = exp(-2*dr) / (dr*dr*dr+1e-16);
                }
            }
        }
        free(g);
        free(buf);
        free(grid_dist);
}
}