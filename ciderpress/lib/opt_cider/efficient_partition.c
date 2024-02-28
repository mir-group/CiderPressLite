#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 1k grids per block
#define GRIDS_BLOCK     512
#define RCUT_LKO        5.0
#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

void VXCgen_grid_lko(double *out, double *coords, double *atm_coords,
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
            atom_dist[i*natm+j] = 1 / MIN(sqrt(dx*dx + dy*dy + dz*dz), RCUT_LKO);
        }
    }

#pragma omp parallel private(i, j, dx, dy, dz)
{
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *min_dist_i = malloc(sizeof(double) * natm);
    double *min_dist_n = malloc(sizeof(double) * GRIDS_BLOCK);
    double *g = malloc(sizeof(double) * GRIDS_BLOCK);
    size_t ig0, n, ngs;
    double fac;
    double max_min_dist;
#pragma omp for nowait schedule(dynamic,1)
    for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
        ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
        for (n = 0; n < ngs; n++) {
            min_dist_n[n] = 1e10;
        }
        for (i = 0; i < natm; i++) {
            min_dist_i[i] = 1e10;
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
                // for screening
                min_dist_i[i] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_i[i]);
                min_dist_n[n] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_n[n]);
            }
        }
        max_min_dist = 0.0;
        for (n = 0; n < ngs; n++) {
            max_min_dist = MAX(max_min_dist, min_dist_n[n]);
        }

        for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            // for screening
            if ((min_dist_i[i] > max_min_dist + RCUT_LKO) &&
                (min_dist_i[j] > max_min_dist + RCUT_LKO)) {
                continue;
            }
            fac = atom_dist[i*natm+j];
            for (n = 0; n < ngs; n++) {
                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                g[n] = MAX(-1, g[n]);
                g[n] = MIN(1, g[n]);
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

void VXCgen_grid_proj(double *out, double *coords, double *atm_coords,
                     double *radii_table, int natm, int ngrids)
{
    const size_t Ngrids = ngrids;
    int i, j;
    double dx, dy, dz;
    double *atom_dist = malloc(sizeof(double) * natm*natm);
    double *atom_diff = malloc(sizeof(double) * natm*natm*3);
    int ind;
    for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            ind = i*natm+j;
            dx = atm_coords[i*3+0] - atm_coords[j*3+0];
            dy = atm_coords[i*3+1] - atm_coords[j*3+1];
            dz = atm_coords[i*3+2] - atm_coords[j*3+2];
            atom_dist[ind] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
            atom_diff[3*ind+0] = -dx * atom_dist[ind];
            atom_diff[3*ind+1] = -dy * atom_dist[ind];
            atom_diff[3*ind+2] = -dz * atom_dist[ind];
            atom_dist[ind] = MAX(atom_dist[ind], 1.0 / RCUT_LKO);
        }
    }

#pragma omp parallel private(i, j, dx, dy, dz, ind)
{
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *grid_diff = malloc(sizeof(double) * 3*GRIDS_BLOCK);
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
            for (n = 0; n < ngs; n++) {
                grid_diff[0*ngs+n] = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                grid_diff[1*ngs+n] = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                grid_diff[2*ngs+n] = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
            }
            for (j = 0; j < i; j++) {
                ind = i*natm+j;
                fac = atom_dist[i*natm+j];
                for (n = 0; n < ngs; n++) {
                    //g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    //    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                    g[n] = (grid_diff[0*ngs+n] * atom_diff[3*ind+0]
                          + grid_diff[1*ngs+n] * atom_diff[3*ind+1]
                          + grid_diff[2*ngs+n] * atom_diff[3*ind+2]);
                    g[n] = g[n] * fac * 2 - 1;
                    g[n] = MAX(-1, g[n]);
                    g[n] = MIN(1, g[n]);
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
                    g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
                }
                for (n = 0; n < ngs; n++) {
                    buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                    buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
                }
            }
        }

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

/*
void VXCgen_grid(double *out, double *coords, double *atm_coords,
         double *radii_table, int natm, int ngrids)
{
        for (ii = 0; ii < natm; ii++) {
            i = ia_list[ii];
            if ( grid_dist[i*GRIDS_BLOCK+n] > r_nearest + RCUT ) {
                break;
            }

            for (jj = ii+1; jj < natm; jj++) {
                j = ia_list[jj];
                if (grid_dist[j*GRIDS_BLOCK+n] > grid_dist[i*GRIDS_BLOCK+n] + RCUT) {
                    break;
                }

                fac = MIN(RCUT, atom_dist[i*natm+j]);
                for (n = 0; n < ngs; n++) {
                    g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                    g[n] = MAX(-1, g[n]);
                    g[n] = MIN(1, g[n]);
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
                    g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
                }
                for (n = 0; n < ngs; n++) {
                    buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                    buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
                }
            }
        }
*/
