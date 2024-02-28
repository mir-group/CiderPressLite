#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fblas.h"
#include "nr_cider_numint.h"

void VXCfill_coefs_spline(double *p_ag, double *dp_ag, double *C_aip, double *cider_exp,
                          double *alphas, int ngrids, int nalpha)
{
    double *dq = (double *) malloc(ngrids * sizeof(double));
    int *iq = (int *) malloc(ngrids * sizeof(int));
    double alpha_min = alphas[0];
    double loglambd = log(alphas[1] / alphas[0]);
#pragma omp parallel
{
    double tmp;
    int g, a;
    double *C_p;
    int ind;
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        tmp = log(cider_exp[g] / alpha_min) / loglambd;
        iq[g] = (int) floor(tmp);
        iq[g] = fmin(nalpha-1, fmax(0, iq[g]));
        dq[g] = cider_exp[g] - alphas[iq[g]];
    }
#pragma omp for
    for (a = 0; a < nalpha; a++) {
        for (g = 0; g < ngrids; g++) {
            ind = a * ngrids + g;
            C_p = C_aip + (a*nalpha + iq[g]) * 4;
            p_ag[ind] = C_p[0] + dq[g] * (C_p[1] + dq[g] * (C_p[2] + dq[g] * C_p[3]));
            dp_ag[ind] = C_p[1] + dq[g] * (2 * C_p[2] + 3 * dq[g] * C_p[3]);
        }
    }
}
    free(dq);
    free(iq);
}

void VXCgenerate_atc_integrals(double *ovlp_mats, int *alpha_loc, int *shl_loc,
                               int *atm, int natm, int *bas, int nbas, double *env, int *atom_loc_ao,
                               double *alphas, double *gammas, double* gcoefs, int nalpha, int ngamma,
                               double *alpha_norms)
{
#pragma omp parallel
{
    double PI = 4 * atan(1.0);
    int aa, a, b, ish, jsh, ish0, ish1, gam0, gam1;
    int at, l, ind;
    double alpha, beta;
    double *expi = (double*) malloc(nalpha * sizeof(double));
    double *coefi = (double*) malloc(nalpha * sizeof(double));
    double expj, coefj;
    double *ovlp;
    int *ibas;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        gam0 = 0;
        gam1 = ngamma;
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            at = ibas[ATOM_OF];
            l = ibas[ANG_OF];
            beta = env[ibas[PTR_EXP]];
            for (b = 0; b < nalpha; b++) {
                alpha = alphas[a] + alphas[b];
                expi[b] = beta * alpha / (beta + alpha);
                coefi[b] = env[ibas[PTR_COEFF]] * pow(PI / alpha, 1.5) * pow(alpha / (beta + alpha), 1.5+l) * alpha_norms[a] * alpha_norms[b];
            }
            ind = 0;
            for (jsh = gam0; jsh < gam1; jsh++) {
                expj = gammas[jsh];
                coefj = gcoefs[l*ngamma+jsh];
                for (b = 0; b < nalpha; b++, ind++) {
                    ovlp[ind] = coefi[b] * coefj * gauss_integral(l, expi[b]+expj);
                }
            }
        }
    }
    free(expi);
    free(coefi);
}
}


void VXCsolve_atc_coefs(double *ovlp_mats, int *atm, int natm, int *bas, int nbas, double *env, int *atom_loc_ao,
                        double *cho_mats, int nalpha, int ngamma, char UPLO) {
#pragma omp parallel
{
    int aa, a, ish, ish0, ish1;
    int at, l;
    int x, y;
    double *ovlp, *chomat;
    int *ibas;
    int info;
    double *buf = (double*) malloc(ngamma*nalpha*sizeof(double));
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            l = ibas[ANG_OF];
            chomat = cho_mats + l*ngamma*ngamma;
            for (x = 0; x < ngamma; x++) {
                for (y = 0; y < nalpha; y++) {
                    buf[y*ngamma+x] = ovlp[x*nalpha+y];
                }
            }
            dpotrs_(&UPLO, &ngamma, &nalpha, chomat, &ngamma, buf, &ngamma, &info);
            for (x = 0; x < ngamma; x++) {
                for (y = 0; y < nalpha; y++) {
                    ovlp[x*nalpha+y] = buf[y*ngamma+x];
                }
            }
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
        }
    }
    free(buf);
}
}

void VXCmultiply_atc_integrals(
    double *q_au, double *p_ua,
    int *atom_loc_ao, int *ao_loc, int lmax,
    double *ovlp_mats, int nalpha, int ngamma,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    // atom_loc should be atom_loc within auxbasis env, not the large basis
    int nauxo = ngamma*(lmax+1)*(lmax+1);
    int aa, a, i, b, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1;
    double *q_u, *ovlp;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        q_u = q_au + aa*nauxo;
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        gam0 = 0;
        gam1 = ngamma;
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            j = ngamma*l*l;
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    #pragma omp simd
                    for (b = 0; b < nalpha; b++) {
                        // TODO this is not optimized due to 'complicated access pattern',
                        // will be faster if simiplified
                        q_u[j] += ovlp[gam*nalpha+b] * p_ua[i*nalpha+b];
                    }
                }
            }
        }
    }
}
}


void VXCmultiply_atc_integrals_bwd(
    double *q_au, double *p_ua,
    int *atom_loc_ao, int *ao_loc, int lmax,
    double *ovlp_mats, int nalpha, int ngamma,
    int *atm, int natm, int *bas, int nbas, double *env
) {
//#pragma omp parallel
{
    // atom_loc should be atom_loc within auxbasis env, not the large basis
    int nauxo = ngamma*(lmax+1)*(lmax+1);
    int aa, a, i, b, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1;
    double *q_u, *ovlp;
//#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        q_u = q_au + aa*nauxo;
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        gam0 = 0;
        gam1 = ngamma;
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            j = ngamma*l*l;
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    //#pragma omp simd
                    for (b = 0; b < nalpha; b++) {
                        //q_u[j] += ovlp[gam*nalpha+b] * p_ua[i*nalpha+b];
                        p_ua[i*nalpha+b] += ovlp[gam*nalpha+b] * q_u[j];
                    }
                }
            }
        }
    }
}
}

void VXCmultiply_becke_weights(double *pbecke, double *auxo, int *atm_list,
                               int natm, int ngrids, int nauxo) {
    
//#pragma omp parallel
{
    int atm, g, i;
    double *auxo_tmp;
    double *pbecke_tmp;
//#pragma omp for
    for (i = 0; i < nauxo; i++) {
        atm = atm_list[i];
        auxo_tmp = auxo + i*ngrids;
        pbecke_tmp = pbecke + atm*ngrids;
        for (g = 0; g < ngrids; g++) {
            auxo_tmp[g] *= pbecke_tmp[g];
        }
    }
}
}


/*
#define INITIALIZE_GRID_INTS \
    int natm = atco->natm; \
    int nauxo = atco->naux_conv; \
    int *conv_aux_loc = atco->conv_aux_loc; \
    int ntpr = nlm*nalpha; \
    double coefg, tmp, tmp0, tmp1, alpha, expnt; \
    int aa, a, i, b, at, l, gam, ind, nm, m, lm, r; \
    int gsh; \
    double *ovlp_q, *expar2_q, *theta_q, *f_u, *rad_l; \
    double *expar2 = (double*) malloc(ngamma * nalpha * sizeof(double)); \
    atc_conv_set atcc;

#define INITIALIZE_GRID_INT_RADS \
    for (r = 0; r < nrad; r++) { \
        rad_rl[lp1*r] = 1.0; \
        for (l = 1; l < lp1; l++) { \
            rad_rl[lp1*r+l] = rads[r] * rad_rl[lp1*r+l-1]; \
        } \
    }

#define INITIALIZE_GRID_INT_PREFACS \
    for (a = 0; a < nalpha; a++) { \
        for (l = 0; l < lp1; l++) { \
            for (gam = 0; gam < ngamma; gam++) { \
                for (b = 0; b < nalpha; b++) { \
                    ind = b+nalpha*(gam+ngamma*(l+lp1*a)); \
                    tmp = 1.0 / (gammas[gam] + alphas[a] + alphas[b]); \
                    ovlp_prefacs[ind] = pow(M_PI * tmp, 1.5); \
                    tmp *= alphas[a] + alphas[b]; \
                    ovlp_prefacs[ind] *= pow(tmp, l); \
                    ovlp_prefacs[ind] *= alpha_norms[a]; \
                    ovlp_prefacs[ind] *= alpha_norms[b]; \
                } \
            } \
        } \
    }

void compute_grid_ints_fwd(
    double *f_qu, double *theta_rlmq,
    atc_ovlp_set *atco, double *alphas, double *rads,
    int *ra_loc, double *alpha_norms, int nalpha, int nrad, int nlm
)
{
    int ngamma = atco->gamma_set_size;
    int lp1 = (int)sqrt(nlm+1e-8);
    double *rad_rl = (double*) malloc(nrad*lp1*sizeof(double));
    double *ovlp_prefacs = (double*) malloc(nalpha*nalpha*ngamma*lp1*sizeof(double));
    double *gammas = atco->gamma_set;
#pragma omp parallel
{
    INITIALIZE_GRID_INTS
#pragma omp for
    INITIALIZE_GRID_INT_RADS
#pragma omp for
    INITIALIZE_GRID_INT_PREFACS
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        atcc = atco->atc_convs[at];
        alpha = alphas[a];
        f_u = f_qu + a*nauxo;
        for (r = ra_loc[at]; r < ra_loc[at+1]; r++) {
            rad_l = rad_rl + r*lp1;
            for (gam = 0; gam < ngamma; gam++) {
                for (b = 0; b < nalpha; b++) {
                    tmp0 = alpha + alphas[b];
                    tmp1 = 1.0 / (gammas[gam] + tmp0);
                    expnt = gammas[gam] * tmp0 * tmp1;
                    expar2[gam*nalpha+b] = exp(-expnt * rad_l[2]);
                }
            }
            for (l = 0; l <= atcc.lmax; l++) {
                for (gsh = atcc.l_loc[l]; gsh < atcc.l_loc[l+1]; gsh++) {
                    //printf("GSH %d %d %d %d %d %d %d %d %lf\n", aa, a, l, gsh, atcc.l_loc[l], atcc.l_loc[l+1], at, atcc.ngsh, atcc.gcoefs[gsh]);
                    if (gsh < 0 || gsh >= atcc.ngsh) {
                        printf("ERROR\n");
                        exit(-1);
                    }
                    if (atcc.ia_loc + gsh >= atco->nshl_conv) {
                        printf("ERROR\n");
                        exit(-1);
                    }
                    if (atcc.lmax >= lp1) {
                        printf("ERROR %d %d\n", atcc.lmax, lp1);
                        exit(-1);
                    }
                    coefg = atcc.gcoefs[gsh];
                    coefg *= rad_l[l];
                    gam = atcc.gamma_ids[gsh];
                    i = conv_aux_loc[atcc.ia_loc + gsh];
                    lm = l*l;
                    nm = 2*l+1;
                    ovlp_q = ovlp_prefacs + nalpha*(gam+ngamma*(l+lp1*a));
                    expar2_q = expar2 + gam*nalpha;
                    for (m = 0; m < nm; m++, lm++, i++) {
                        theta_q = theta_rlmq + r*ntpr + lm*nalpha;
                        for (b = 0; b < nalpha; b++) {
                            f_u[i] += coefg * ovlp_q[b] * expar2_q[b]
                                      * theta_q[b];
                        }
                    }
                }
            }
        }
    }
}
    free(rad_rl);
    free(ovlp_prefacs);
}

void compute_grid_ints_bwd(
    double *f_qu, double *theta_rlmq,
    atc_ovlp_set *atco, double *alphas, double *rads,
    int *ar_loc, double *alpha_norms, int nalpha, int nrad, int nlm
)
{
    int ngamma = atco->gamma_set_size;
    int lp1 = sqrt(nlm+1e-8);
    double *rad_rl = (double*) malloc(nrad*lp1*sizeof(double));
    double *ovlp_prefacs = (double*) malloc(nalpha*nalpha*ngamma*lp1*sizeof(double));
    double *gammas = atco->gamma_set;
#pragma omp parallel
{
    INITIALIZE_GRID_INTS
#pragma omp for
    INITIALIZE_GRID_INT_RADS
#pragma omp for
    INITIALIZE_GRID_INT_PREFACS
#pragma omp for schedule(dynamic, 4)
    for (r = 0; r < nrad; r++) {
        rad_l = rad_rl + r*lp1;
        at = ar_loc[r];
        for (a = 0; a < nalpha; a++) {
            atcc = atco->atc_convs[at];
            alpha = alphas[a];
            f_u = f_qu + a*nauxo;
            for (gam = 0; gam < ngamma; gam++) {
                for (b = 0; b < nalpha; b++) {
                    tmp0 = alpha + alphas[b];
                    tmp1 = 1.0 / (gammas[gam] + tmp0);
                    expnt = gammas[gam] * tmp0 * tmp1;
                    expar2[gam*nalpha+b] = exp(-expnt * rad_l[2]);
                }
            }
            for (l = 0; l <= atcc.lmax; l++) {
                for (gsh = atcc.l_loc[l]; gsh < atcc.l_loc[l+1]; gsh++) {
                    coefg = atcc.gcoefs[gsh];
                    coefg *= rad_l[l];
                    gam = atcc.gamma_ids[gsh];
                    i = conv_aux_loc[atcc.ia_loc + gsh];
                    lm = l*l;
                    nm = 2*l+1;
                    ovlp_q = ovlp_prefacs + nalpha*(gam+ngamma*(l+lp1*a));
                    expar2_q = expar2 + gam*nalpha;
                    for (m = 0; m < nm; m++, lm++, i++) {
                        theta_q = theta_rlmq + r*ntpr + lm*nalpha;
                        for (b = 0; b < nalpha; b++) {
                            theta_q[b] += coefg * ovlp_q[b] * expar2_q[b]
                                          * f_u[i];
                        }
                    }
                }
            }
        }
    }
}
    free(rad_rl);
    free(ovlp_prefacs);
}
*/
