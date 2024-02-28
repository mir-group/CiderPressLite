#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fblas.h"
#include "nr_cider_numint.h"


/** 
 * Fills the coefficients p_ag for auxiliary basis dcomposition.
 * 
 * p_ag : coefficient array nalpha x ngrids
 * dp_ag : derivatives of coefficients wrt exponent alpha
 * cider_exp : exponent array
 * alphas : control points
 * ngrids : number of realspace grids
 * nalpha : number of control points
 */
void VXCfill_coefs(double *p_ag, double* dp_ag, double *cider_exp, double *alphas,
                   int ngrids, int nalpha)
{
    double pi32 = pow(4 * atan(1.0), 1.5);
#pragma omp parallel
{
    double tmp;
    int g, a;
#pragma omp for
    for (a = 0; a < nalpha; a++) {
        for (g = 0; g < ngrids; g++) {
            tmp = 1.0 / sqrt(cider_exp[g] + alphas[a]);
            p_ag[a*ngrids+g] = pi32 * tmp * tmp * tmp;
            dp_ag[a*ngrids+g] = -1.5 * p_ag[a*ngrids+g] * tmp * tmp;
        }
    }
}
}

/**
 * Same as above routine, but fills p_ga and dp_ga (i.e. transpose of above result)
 */
void VXCfill_coefs_t(double *p_ag, double* dp_ag, double *cider_exp, double *alphas,
                     int ngrids, int nalpha)
{
    double pi32 = pow(4 * atan(1.0), 1.5);
#pragma omp parallel
{
    double tmp;
    int g, a;
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        for (a = 0; a < nalpha; a++) {
            tmp = 1.0 / sqrt(cider_exp[g] + alphas[a]);
            p_ag[g*nalpha+a] = pi32 * tmp * tmp * tmp;
            dp_ag[g*nalpha+a] = -1.5 * p_ag[g*nalpha+a]* tmp * tmp;
        }
    }
}
}

/**
 * Same as above routine but for polynomial augmented features
 */
void VXCfill_poly_coefs_t(double *p_ag, double *p2_ag, double *p4_ag,
                          double* dp_ag, double *dp2_ag, double *dp4_ag,
                          double *cider_exp, double *alphas,
                          int ngrids, int nalpha)
{
    double coef00 = pow(4 * atan(1.0), 1.5);
    double coef11 = -2 * coef00;
    double coef21 = -4 * coef00;
    double coef22 = 4 * coef00;
#pragma omp parallel
{
    double tmpsum, tmp, dtmp, tmp0, tmp1;
    double poly2, poly4;
    int g, a;
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        for (a = 0; a < nalpha; a++) {
            tmpsum = 1.0 / (cider_exp[g] + alphas[a]);
            tmp = cider_exp[g] / tmpsum;
            poly2 = coef00 + coef11 * tmp;
            poly4 = coef00 + coef21 * tmp + coef22 * tmp * tmp;
            dtmp = -1 * alphas[a] * tmpsum * tmpsum;
            tmp0 = sqrt(tmp);
            tmp0 = tmp0 * tmp;
            tmp1 = -1.5 * tmp0 * tmp;
            p_ag[g*nalpha+a] = coef00 * tmp0;
            p2_ag[g*nalpha+a] = tmp0 * poly2;
            p4_ag[g*nalpha+a] = tmp0 * poly4;
            dp_ag[g*nalpha+a] = coef00 * tmp1;
            dp2_ag[g*nalpha+a] = tmp1 * poly2 + tmp0 * dtmp * coef11;
            dp4_ag[g*nalpha+a] = tmp1 * poly4 + tmp0 * dtmp * (coef21 + 2 * coef22 * tmp);
        }
    }
}
}

/**
 * Forward pass coefs for version k descriptors
 */
void VXCfill_coefs_kv1_t(double *p_ag, double* dp_ag, double *cider_exp, double *alphas,
                          int ngrids, int nalpha)
{
#pragma omp parallel
{
    double tmp;
    int g, a;
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        for (a = 0; a < nalpha; a++) {
            tmp = -1.5 / alphas[a];
            p_ag[g*nalpha+a] = exp(tmp * cider_exp[g]);
            dp_ag[g*nalpha+a] = tmp * p_ag[g*nalpha+a];
        }
    }
}
}

void VXCfill_coefs_spline_t(double *p_ga, double* dp_ga, double *exp_g, double *w_iap,
                            int ngrids, int nalpha, double amax, double lambd)
{
    int *i_g = (int*) malloc(ngrids * sizeof(int));
    double *diffi_gp = (double*) malloc(3 * ngrids * sizeof(double));
    double *derivi_g = (double*) malloc(ngrids * sizeof(double));
#pragma omp parallel
{
    double tmp, derivi;
    int g, a;
    double *diffi_p, *w_ap, *w_p, *p_a, *dp_a;
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        tmp = -1.0 / log(lambd);
        derivi_g[g] = tmp / exp_g[g];
        tmp *= log(exp_g[g] / amax);
        i_g[g] = (int) tmp;
        tmp -= i_g[g];
        diffi_p = diffi_gp + 3*g;
        diffi_p[0] = tmp;
        diffi_p[1] = tmp * tmp;
        diffi_p[2] = tmp * tmp * tmp;
    }
#pragma omp for
    for (g = 0; g < ngrids; g++) {
        p_a = p_ga + g * nalpha;
        dp_a = dp_ga + g * nalpha;
        w_ap = w_iap + i_g[g]*4*nalpha;
        diffi_p = diffi_gp + 3*g;
        derivi = derivi_g[g];
        for (a = 0; a < nalpha; a++) {
            w_p = w_ap + 4*a;
            p_a[a] = w_p[0] + diffi_p[0] * w_p[1] + diffi_p[1] * w_p[2]
                     + diffi_p[2] * w_p[3];
            dp_a[a] = w_p[1] + 2 * w_p[2] * diffi_p[0]
                      + 3 * w_p[3] * diffi_p[1];
            dp_a[a] *= derivi;
        }
    }
}
    free(i_g);
    free(diffi_gp);
    free(derivi_g);
}

/** int_0^infty dr r^(2*l+2) exp(-a*r^2) */
double gauss_integral(int l, double a) {
    return 0.5 * pow(a, -1.5-l) * tgamma(1.5+l);
}

/** int_0^infty dr r^(2+n) exp(-alpha*r^2) */
double gto_norm2(int n, double alpha) {
    return 0.5 * pow(alpha, -0.5 * (3 + n)) * tgamma(0.5 * (3 + n));
}

/** Helper function for free_atc_set */
void free_atc_conv_set(atc_conv_set atcc) {
    free(atcc.l_loc);
    free(atcc.l_loc2);
    free(atcc.gammas);
    free(atcc.gcoefs);
    free(atcc.gtrans);
}

/** Free an atc_ovlp_set object */
void free_atc_set(atc_ovlp_set *atco) {
    int ia;
    if (atco->atc_convs != NULL) {
        for (ia=0; ia<atco->natm; ia++) {
            free_atc_conv_set(atco->atc_convs[ia]);
        }
        free(atco->atc_convs);
    }
    if (atco->conv_aux_loc != NULL) {
        free(atco->conv_aux_loc);
    }
    if (atco->conv_aux_loc != NULL) {
        free(atco->atc_loc);
    }
    if (atco->bas != NULL) {
        free(atco->bas);
        free(atco->env);
    } 
    free(atco);
}

/** Getter functions for atc_ovlp_set */
int get_atco_nshl_conv(atc_ovlp_set *atco) {
    return atco->nshl_conv;
}

void get_atco_aux_loc(int *loc, atc_ovlp_set *atco) {
    int i;
    for(i=0; i<atco->nshl_conv+1; i++) {
        loc[i] = atco->conv_aux_loc[i];
    }
}

void get_atco_bas(int *bas, atc_ovlp_set *atco) {
    int i;
    for(i=0; i<BAS_SLOTS*atco->nshl_conv; i++) {
        bas[i] = atco->bas[i];
    }
}

void get_atco_env(double *env, atc_ovlp_set *atco) {
    int i;
    for(i=0; i<2*atco->nshl_conv; i++) {
        env[i] = atco->env[i];
    }
}

/**
 * Allocates and constructs an atc_ovlp_set, then assigns a pointer
 * to the object to *atco_p.
 * atco_p : Location to place pointer to created object
 * atc_loc : 
 * ag_loc :
 * gamma_loc :
 * all_gammas : List of gamma exponents (ETB basis for convolved feature contributions)
 *              for all atoms and angular moment values
 * TODO finish docs
 */
void make_atc_set(atc_ovlp_set **atco_p, int *atc_loc, 
                  int *ag_loc, int *gamma_loc, int *gamma_loc2,
                  double *all_gammas, double *all_gcoefs, int *lmaxs,
                  int natm, int nbas, char UPLO,
                  double *gamma_set, int *gamma_ids, int gamma_set_size)
{
    atc_ovlp_set *atco = malloc(sizeof(atc_ovlp_set));
    atco->atc_loc = malloc((nbas+1) * sizeof(int));
    atco->UPLO = UPLO;
    int info;
    int ish, gsh0, gsh1, g, g0, g1;
    for (ish = 0; ish <= nbas; ish++) {
        atco->atc_loc[ish] = atc_loc[ish];
    }
    atco->natc = atc_loc[nbas];
    atco->nshl_conv = gamma_loc[ag_loc[natm]];
    //printf("nshl %d\n", atco->nshl_conv);
    atco->naux_conv = 0;
    atco->conv_aux_loc = (int*) malloc((atco->nshl_conv+1) * sizeof(int));
    atco->conv_aux_loc[0] = 0;
    atco->atc_convs = (atc_conv_set*) malloc(natm * sizeof(atc_conv_set));
    atco->natm = natm;
    atco->env = (double*) malloc(atco->nshl_conv * 2 * sizeof(double));
    atco->bas = (int*) malloc(BAS_SLOTS * atco->nshl_conv * sizeof(int));
    atco->gamma_set = (double*) malloc(gamma_set_size * sizeof(double));
    for (g = 0; g < gamma_set_size; g++) {
        atco->gamma_set[g] = gamma_set[g];
    }
    atco->gamma_set_size = gamma_set_size;
    atc_conv_set *atcc;
    double *gammas;
    double *gcoefs;
    double *gtrans;
    double *gtrans_m;
    double *gtrans_p;
    int *l_loc, *l_loc2;
    int nglsh, ngsh, ngsh2, ia, l;
    double coef0, coef1, exp0, exp1;
    int shl = 0;
    // TODO do not parallelize unless shl is taken care of
    for (ia = 0; ia < natm; ia++) {
        atcc = atco->atc_convs + ia;
        atcc->lmax = lmaxs[ia];
        atcc->ia = ia;
        atcc->ia_loc = gamma_loc[ag_loc[ia]];
        //shl = atcc->ia_loc;
        gsh0 = gamma_loc[ag_loc[ia]];
        gsh1 = gamma_loc[ag_loc[ia+1]];
        ngsh = gsh1 - gsh0;
        ngsh2 = gamma_loc2[ag_loc[ia+1]] - gamma_loc2[ag_loc[ia]];
        atcc->l_loc = (int*) malloc((atcc->lmax+2) * sizeof(int));
        atcc->l_loc2 = (int*) malloc((atcc->lmax+2) * sizeof(int));
        l_loc = atcc->l_loc;
        l_loc2 = atcc->l_loc2;
        atcc->gammas = (double*) malloc(ngsh * sizeof(double));
        atcc->gcoefs = (double*) malloc(ngsh * sizeof(double));
        atcc->gtrans = (double*) malloc(ngsh2 * sizeof(double));
        atcc->gtrans_m = (double*) malloc(ngsh2 * sizeof(double));
        atcc->gtrans_p = (double*) malloc(ngsh2 * sizeof(double));
        atcc->gamma_ids = (int*) malloc(ngsh * sizeof(int));
        atcc->ngsh = ngsh;
        atcc->ngsh2 = ngsh2;
        gammas = atcc->gammas;
        gcoefs = atcc->gcoefs;
        gtrans = atcc->gtrans;
        gtrans_m = atcc->gtrans_m;
        gtrans_p = atcc->gtrans_p;
        for (l = 0; l < atcc->lmax+2; l++) {
            l_loc[l] = gamma_loc[ag_loc[ia]+l] - gsh0;
            l_loc2[l] = gamma_loc2[ag_loc[ia]+l] - gamma_loc2[ag_loc[ia]];
        }
        for (g = 0; g < ngsh; g++) {
            gammas[g] = all_gammas[g+gsh0];
            gcoefs[g] = all_gcoefs[g+gsh0];
            atcc->gamma_ids[g] = gamma_ids[g+gsh0];
        }
        for (l = 0; l < atcc->lmax+1; l++) {
            nglsh = l_loc[l+1] - l_loc[l];
            for (g0 = 0; g0 < nglsh; g0++) {
                coef0 = gcoefs[l_loc[l] + g0];
                exp0 = gammas[l_loc[l] + g0];
                // set bas and env
                //printf("shl %d\n", shl);
                atco->bas[shl*BAS_SLOTS+ATOM_OF] = ia;
                atco->bas[shl*BAS_SLOTS+ANG_OF] = l;
                atco->bas[shl*BAS_SLOTS+PTR_COEFF] = 2*shl;
                atco->bas[shl*BAS_SLOTS+PTR_EXP] = 2*shl+1;
                atco->bas[shl*BAS_SLOTS+7] = gamma_ids[l_loc[l] + g0];
                atco->env[2*shl] = coef0;
                atco->env[2*shl+1] = exp0;
                atco->conv_aux_loc[shl+1] = atco->conv_aux_loc[shl] + 2*l+1;
                shl++;
                for (g1 = 0; g1 < nglsh; g1++) {
                    coef1 = gcoefs[l_loc[l] + g1];
                    exp1 = gammas[l_loc[l] + g1];
                    gtrans[l_loc2[l] + g0*nglsh + g1] = 
                        coef0 * coef1 * gauss_integral(l, exp0+exp1);
                    gtrans_m[l_loc2[l] + g0*nglsh + g1] =
                        coef0 * coef1 * gauss_integral(l - 1, exp0+exp1);
                    gtrans_p[l_loc2[l] + g0*nglsh + g1] =
                        coef0 * coef1 * gauss_integral(l + 1, exp0+exp1);
                }
            }
            dpotrf_(&(atco->UPLO), &nglsh, gtrans+l_loc2[l], &nglsh, &info);
            dpotrf_(&(atco->UPLO), &nglsh, gtrans_m+l_loc2[l], &nglsh, &info);
            dpotrf_(&(atco->UPLO), &nglsh, gtrans_p+l_loc2[l], &nglsh, &info);
        }
    }
    atco->naux_conv = atco->conv_aux_loc[atco->nshl_conv];
    atco_p[0] = atco;
}

/**
 * For version 2 of generate_atc_integrals, need an loc array to find the coefficients and exponents
 * for the feature auxiliary basis for each l and atom. There also needs to be a loc array
 * to give the location, in the ovl_mats array, for a given shell (the increment for each shell
 * is the number of gammas at that shell's l value for that shell's atom).
*/
void VXCgenerate_atc_integrals2(double *ovlp_mats, int *alpha_loc, int *shl_loc,
                                int *atm, int natm, int *bas, int nbas, double *env,
                                int *atom_loc_ao, double *alphas, atc_ovlp_set *atco,
                                int nalpha, int ngamma_max, double *alpha_norms)
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
    atc_conv_set atcc;
    int natc = atco->natc;
    double *gammas, *gcoefs;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        atcc = atco->atc_convs[at];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET2(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            at = ibas[ATOM_OF];
            l = ibas[ANG_OF];
            beta = env[ibas[PTR_EXP]];
            gammas = atcc.gammas + atcc.l_loc[l];
            gcoefs = atcc.gcoefs + atcc.l_loc[l];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (b = 0; b < nalpha; b++) {
                alpha = alphas[a] + alphas[b];
                expi[b] = beta * alpha / (beta + alpha);
                coefi[b] = env[ibas[PTR_COEFF]] * pow(PI / alpha, 1.5) * pow(alpha / (beta + alpha), 1.5+l) * alpha_norms[a] * alpha_norms[b];
            }
            ind = 0;
            for (jsh = gam0; jsh < gam1; jsh++) {
                expj = gammas[jsh];
                coefj = gcoefs[jsh];
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


void VXCgenerate_atc_integrals4(double *ovlp_mats, int *alpha_loc, int *shl_loc,
                                int *atm, int natm, int *bas, int nbas, double *env,
                                int *atom_loc_ao, double *alphas, atc_ovlp_set *atco,
                                int nalpha, int ngamma_max, double *alpha_norms)
{
#pragma omp parallel
{
    double PI = 4 * atan(1.0);
    int aa, a, ish, jsh, ish0, ish1, gam0, gam1;
    int at, l;
    double alpha, beta, integral, intm, intp, tmp;
    double expi, coefi, coefi2;
    double expj, coefj;
    double *ovlp;
    int *ibas;
    atc_conv_set atcc;
    int natc = atco->natc;
    int naa = nalpha * natc;
    double *gammas, *gcoefs;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        atcc = atco->atc_convs[at];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET4(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            at = ibas[ATOM_OF];
            l = ibas[ANG_OF];
            beta = env[ibas[PTR_EXP]];
            gammas = atcc.gammas + atcc.l_loc[l];
            gcoefs = atcc.gcoefs + atcc.l_loc[l];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];

            alpha = alphas[a];
            expi = beta * alpha / (beta + alpha);
            coefi = env[ibas[PTR_COEFF]] * pow(PI / alpha, 1.5)
                    * pow(alpha / (beta + alpha), 1.5+l)
                    * alpha_norms[a];
            coefi2 = (l / alpha - (1.5 + l) / (alpha + beta));
            for (jsh = gam0; jsh < gam1; jsh++) {
                expj = gammas[jsh];
                coefj = gcoefs[jsh];
                integral = gauss_integral(l, expi + expj);
                intm = coefi * coefj * gauss_integral(l - 1, expi + expj);
                intp = coefi * coefj * gauss_integral(l + 1, expi + expj);
                tmp = coefi * coefj * integral;
                ovlp[jsh] = tmp;
                ovlp[jsh + naa] = intm / (2 * alpha);
                ovlp[jsh + 4 * naa] = 0.5 * intm;
                ovlp[jsh + 5 * naa] = tmp * alpha;
                ovlp[jsh + 2 * naa] = tmp * (1.5 + l) / (expi + expj) * beta * beta
                                       / (alpha + beta) / (alpha + beta);
                ovlp[jsh + 2 * naa] -= tmp * coefi2;
                ovlp[jsh + 3 * naa] = intp * (-1 * beta) / (beta + alpha);
                ovlp[jsh + 6 * naa] = ovlp[jsh + 2 * naa] * alpha;
                ovlp[jsh + 7 * naa] = intp * (-1 * beta * alpha) / (beta + alpha);
            }
        }
    }
}
}

void compute_gaussians(double *auxo_ig, double *ylm_a, double *R2_ag,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       int *shls_slice, int *ao_loc, int at0, int lm_max, int ngrids) {
#pragma omp parallel
{
    int m, g, nm, l, i0, ish;
    double coef, expi;
    double *auxo_mg, *auxo_g;
    double *R2g, *ylm, *ym;
    int *ibas;
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    double *expg = (double*) malloc(ngrids * sizeof(double));
#pragma omp for schedule(dynamic, 4)
    for (ish = ish0; ish < ish1; ish++) {
        ibas = bas+ish*BAS_SLOTS;
        l = ibas[ANG_OF];
        coef = env[ibas[PTR_COEFF]];
        expi = env[ibas[PTR_EXP]];
        i0 = ao_loc[ish] - ao_loc[ish0];
        nm = 2*l+1;
        auxo_mg = auxo_ig + i0 * ngrids;
        R2g = R2_ag + (ibas[ATOM_OF] - at0) * ngrids;
        ylm = ylm_a + (ibas[ATOM_OF] - at0) * ngrids * (lm_max);
        ym = ylm + l*l*ngrids;
        for (g = 0; g < ngrids; g++) {
            expg[g] = coef * exp(-expi * R2g[g]);// * pow(R2g[g], 0.5*l);
        }
        for (m = 0; m < nm; m++) {
            auxo_g = auxo_mg + m * ngrids;
            for (g = 0; g < ngrids; g++) {
                auxo_g[g] = expg[g] * ym[m*ngrids+g];
            }
        }
    }
    free(expg);
}
}

void VXCsolve_atc_coefs2(double *ovlp_mats, int *atm, int natm,
                         int *bas, int nbas, double *env, int *atom_loc_ao,
                         atc_ovlp_set *atco, int nalpha, int ngamma_max) {
#pragma omp parallel
{
    int aa, a, ish, ish0, ish1;
    int at, l;
    int x, y;
    double *ovlp, *chomat;
    int *ibas;
    int info;
    double *buf = (double*) malloc(ngamma_max*nalpha*sizeof(double));
    int natc = atco->natc;
    int ngamma;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET2(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            l = ibas[ANG_OF];
            atcc = atco->atc_convs[at];
            chomat = atcc.gtrans + atcc.l_loc2[l];
            ngamma = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (x = 0; x < ngamma; x++) {
                for (y = 0; y < nalpha; y++) {
                    buf[y*ngamma+x] = ovlp[x*nalpha+y];
                }
            }
            dpotrs_(&(atco->UPLO), &ngamma, &nalpha, chomat, &ngamma, buf, &ngamma, &info);
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


void VXCsolve_atc_coefs4(double *ovlp_mats, int *atm, int natm,
                         int *bas, int nbas, double *env, int *atom_loc_ao,
                         atc_ovlp_set *atco, int nalpha, int ngamma_max) {
#pragma omp parallel
{
    int aa, a, ish, ish0, ish1, jsh;
    int at, l;
    double *ovlp, *chomat;
    int *ibas;
    double *gammas;
    int info;
    double *buf = (double*) malloc(ngamma_max*nalpha*sizeof(double));
    int natc = atco->natc;
    int ngamma;
    atc_conv_set atcc;
    int naa = nalpha * natc;
    int one = 1;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        atcc = atco->atc_convs[at];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET4(a,ish);
            ibas = bas+ish*BAS_SLOTS;
            l = ibas[ANG_OF];
            chomat = atcc.gtrans + atcc.l_loc2[l];
            ngamma = atcc.l_loc[l+1] - atcc.l_loc[l];
            gammas = atcc.gammas + atcc.l_loc[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans_m + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans_p + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans_m + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            ovlp += naa;
            chomat = atcc.gtrans_p + atcc.l_loc2[l];
            dpotrs_(&(atco->UPLO), &ngamma, &one, chomat, &ngamma, ovlp, &ngamma, &info);
            if (info != 0) {
                printf("Cholesky error %d\n", info);
                exit(-1);
            }
            //ovlp -= naa;
            //for (jsh = 0; jsh < ngamma; jsh++) {
            //    ovlp[2 * naa + jsh] = ovlp[jsh] * (-2 * gammas[jsh]);
            //}
        }
    }
    free(buf);
}
}


void VXCfill_atomic_cho_factors2(int (*intor)(),
                                 double *aux_c, int *aux_shls_slices, int *cho_offsets,
                                 int comp, int hermi, int *ao_loc, void *opt,
                                 int *atm, int natm, int *bas, int nbas, double *env,
                                 char UPLO) {
#pragma omp parallel
{
    int shls_slice[4];
    int info=0;
    int size;
    int at;
    int m, i, j, li, lj, ni, nao_at;
    int ish, jsh;
    double integral, coefi, coefj, expi, expj;
    double *aux_c_tmp;
    int *ibas, *jbas;
    //char UTRI = 'N';
    //printf("CHO FACTORS\n");
#pragma omp for
    for (at = 0; at < natm; at++) {
        shls_slice[0] = aux_shls_slices[at];
        shls_slice[1] = aux_shls_slices[at+1];
        shls_slice[2] = shls_slice[0];
        shls_slice[3] = shls_slice[1];
        size = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]];
        aux_c_tmp = aux_c + cho_offsets[at];
        //GTOint2c(intor, aux_c_tmp, comp, hermi,
        //         shls_slice, ao_loc, opt,
        //         atm, natm, bas, nbas, env);
        nao_at = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]];
        for (ish=shls_slice[0]; ish<shls_slice[1]; ish++) {
            ibas = bas + ish*BAS_SLOTS;
            li = ibas[ANG_OF];
            coefi = env[ibas[PTR_COEFF]];
            expi = env[ibas[PTR_EXP]];
            ni = ao_loc[ish+1] - ao_loc[ish];
            for (jsh=shls_slice[2]; jsh<shls_slice[3]; jsh++) {
                jbas = bas + jsh*BAS_SLOTS;
                lj = jbas[ANG_OF];
                if (li != lj) {
                    continue;
                } else {
                    coefj = env[jbas[PTR_COEFF]];
                    expj = env[jbas[PTR_EXP]];
                    integral = coefi * coefj * gto_norm2(li+lj, expi+expj);
                    for (m=0; m<ni; m++) {
                        i = ao_loc[ish] - ao_loc[shls_slice[0]] + m;
                        j = ao_loc[jsh] - ao_loc[shls_slice[0]] + m;
                        aux_c_tmp[nao_at*i+j] = integral;
                    }
                }
            }
        }
        dpotrf_(&UPLO, &size, aux_c_tmp, &size, &info);
        //dtrtri_(&UPLO, &UTRI, &size, aux_c_tmp, &size, &info);
    }
}
}



void VXCfill_atomic_cho_solves(double *p_au, double *aux_c, int *aux_offsets, int *cho_offsets,
                               int natm, int nalpha, int nauxo, char UPLO) {
#pragma omp parallel
{
    int at, a, aa;
    int size;
    int info;
    int nrhs = 1;
    double *aux_c_tmp, *p_u_tmp;
    /*char TRANS1, TRANS2;
    if (UPLO == 'L') {
        TRANS1 = 'N';
        TRANS2 = 'T';
    } else {
        TRANS1 = 'T';
        TRANS2 = 'N';
    }
    int INCX = 1;
    char DIAG = 'N';*/
#pragma omp for
    for (aa = 0; aa < nalpha*natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        size = aux_offsets[at+1] - aux_offsets[at];
        aux_c_tmp = aux_c + cho_offsets[at];
        p_u_tmp = p_au + a * nauxo + aux_offsets[at];
        dpotrs_(&UPLO, &size, &nrhs, aux_c_tmp, &size, p_u_tmp, &size, &info);
        //dtrmv_(&UPLO, &TRANS1, &DIAG, &size, aux_c_tmp, &size, p_u_tmp, &INCX);
        //dtrmv_(&UPLO, &TRANS2, &DIAG, &size, aux_c_tmp, &size, p_u_tmp, &INCX);
    }
}
}

void VXCmultiply_atc_integrals2(
    double *q_au, double *p_ua, double *ovlp_mats,
    int *atom_loc_ao, int *ao_loc, atc_ovlp_set *atco,
    int nalpha, int ngamma_max,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    int natm = atco->natm;
    int nauxo = atco->naux_conv;
    int natc = atco->natc;
    int *conv_aux_loc = atco->conv_aux_loc;
    int aa, a, i, b, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1, jsh0;
    double *q_u, *ovlp;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        atcc = atco->atc_convs[at];
        q_u = q_au + a*nauxo;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET2(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            jsh0 = atcc.ia_loc + atcc.l_loc[l];
            j = conv_aux_loc[jsh0];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    #pragma omp simd
                    for (b = 0; b < nalpha; b++) {
                        q_u[j] += ovlp[gam*nalpha+b] * p_ua[i*nalpha+b];
                    }
                }
            }
        }
    }
}
}


void VXCmultiply_atc_integrals4(
    double *q_au, double *p_ua, double *ovlp_mats,
    int *atom_loc_ao, int *ao_loc, atc_ovlp_set *atco,
    int nalpha, int ngamma_max,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    int natm = atco->natm;
    int nauxo = atco->naux_conv;
    int natc = atco->natc;
    int *conv_aux_loc = atco->conv_aux_loc;
    int aa, a, i, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1, jsh0;
    double *q_u, *ovlp;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        atcc = atco->atc_convs[at];
        q_u = q_au + a*nauxo;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET4(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            jsh0 = atcc.ia_loc + atcc.l_loc[l];
            j = conv_aux_loc[jsh0];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    q_u[j] += ovlp[gam] * p_ua[i * nalpha + a];
                }
            }
        }
    }
}
}


void VXCmultiply_atc_integrals2_bwd(
    double *q_au, double *p_ua, double *ovlp_mats,
    int *atom_loc_ao, int *ao_loc, atc_ovlp_set *atco,
    int nalpha, int ngamma_max,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    int natm = atco->natm;
    int nauxo = atco->naux_conv;
    int natc = atco->natc;
    int *conv_aux_loc = atco->conv_aux_loc;
    int a, i, b, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1, jsh0;
    double *q_u, *ovlp;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    //for (aa = 0; aa < nalpha * natm; aa++) {
    for (at = 0; at < natm; at++) {
    for (a = 0; a < nalpha; a++) {
        //at = aa % natm;
        //a = aa / natm;
        atcc = atco->atc_convs[at];
        q_u = q_au + a*nauxo;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET2(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            jsh0 = atcc.ia_loc + atcc.l_loc[l];
            j = conv_aux_loc[jsh0];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    #pragma omp simd
                    for (b = 0; b < nalpha; b++) {
                        p_ua[i*nalpha+b] += ovlp[gam*nalpha+b] * q_u[j];
                    }
                }
            }
        }
    }
    }
}
}


void VXCmultiply_atc_integrals4_bwd(
    double *q_au, double *p_ua, double *ovlp_mats,
    int *atom_loc_ao, int *ao_loc, atc_ovlp_set *atco,
    int nalpha, int ngamma_max,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    int natm = atco->natm;
    int natc = atco->natc;
    int *conv_aux_loc = atco->conv_aux_loc;
    int aa, a, i, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1, jsh0;
    double *q_u, *ovlp;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        atcc = atco->atc_convs[at];
        q_u = q_au;// + a*nauxo;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET4(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            jsh0 = atcc.ia_loc + atcc.l_loc[l];
            j = conv_aux_loc[jsh0];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    p_ua[i * nalpha + a] += ovlp[gam] * q_u[j];
                }
            }
        }
    }
}
}

void VXCmultiply_atc_integrals4_bwdv2(
    double *q_au, double *p_ua, double *ovlp_mats,
    int *atom_loc_ao, int *ao_loc, atc_ovlp_set *atco,
    int nalpha, int ngamma_max,
    int *atm, int natm, int *bas, int nbas, double *env
) {
#pragma omp parallel
{
    int natm = atco->natm;
    int nauxo = atco->naux_conv;
    int natc = atco->natc;
    int *conv_aux_loc = atco->conv_aux_loc;
    int aa, a, i, at, l, gam0, gam1, i0, i1, gam, j;
    int ish, ish0, ish1, jsh0;
    double *q_u, *ovlp;
    atc_conv_set atcc;
#pragma omp for schedule(dynamic, 4)
    for (aa = 0; aa < nalpha * natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        atcc = atco->atc_convs[at];
        q_u = q_au + a*nauxo;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        for (ish = ish0; ish < ish1; ish++) {
            ovlp = OVLP_OFFSET4(a,ish);
            l = bas[ish*BAS_SLOTS+ANG_OF];
            i0 = ao_loc[ish];
            i1 = ao_loc[ish+1];
            jsh0 = atcc.ia_loc + atcc.l_loc[l];
            j = conv_aux_loc[jsh0];
            gam0 = 0;
            gam1 = atcc.l_loc[l+1] - atcc.l_loc[l];
            for (gam = gam0; gam < gam1; gam++) {
                for (i = i0; i < i1; i++, j++) {
                    p_ua[i * nalpha + a] += ovlp[gam] * q_u[j];
                }
            }
        }
    }
}
}

void fill_sph_harm_deriv_coeff(double *c_xl, double *d_vxl,
                               double *gaunt_vl, int nx, int lmax)
{
//#pragma omp parallel
{
    int l, m, lm, nlm, lmz;
    nlm = (lmax + 1) * (lmax + 1);
    int i;
    double *dx_l, *dy_l, *dz_l, *cx_l;
    double *gauntxm_l = gaunt_vl + 0 * nlm;
    double *gauntxp_l = gaunt_vl + 1 * nlm;
    double *gauntym_l = gaunt_vl + 2 * nlm;
    double *gauntyp_l = gaunt_vl + 3 * nlm;
    double *gauntz_l = gaunt_vl + 4 * nlm;
//#pragma omp for
    for (i = 0; i < nx; i++) {
        dx_l = d_vxl + (0 * nx + i) * nlm;
        dy_l = d_vxl + (1 * nx + i) * nlm;
        dz_l = d_vxl + (2 * nx + i) * nlm;
        cx_l = c_xl + i * nlm;
        for (l = 0; l < lmax; l++) {
            for (m = 0; m < 2 * l + 1; m++) {
                lm = l * l + m;
                lmz = lm + 2 * l + 2;
                dz_l[lm] += gauntz_l[lm] * cx_l[lmz];
                dx_l[lm] += gauntxm_l[lm] * cx_l[lmz - 1];
                dx_l[lm] += gauntxp_l[lm] * cx_l[lmz + 1];
                lmz += 2 * (l - m);
                dy_l[lm] += gauntym_l[lm] * cx_l[lmz - 1];
                dy_l[lm] += gauntyp_l[lm] * cx_l[lmz + 1];
            }
        }
    }
}
}

void fill_gaussian_deriv_coeff(
    double *f_u, double *d_vu, double *gaunt_vl, int nlm,
    int *atom_loc1_ao, int *ao_loc1,
    int *atm, int natm, int *bas1, int nbas1, double *env1,
    int *atom_loc2_ao, int *ao_loc2,
    int *bas2, int nbas2, double *env2
) {
#pragma omp parallel
{
    int at, l, i0, j0, lm, m, nm;
    int ish, jsh, ish0, ish1, jsh0, jsh1;
    int nao_d = ao_loc2[nbas2] - ao_loc2[0];
    double yx, yy, yz;
    double *dx_u = d_vu + 0 * nao_d;
    double *dy_u = d_vu + 1 * nao_d;
    double *dz_u = d_vu + 2 * nao_d;
    double *gauntxm_l = gaunt_vl + 0 * nlm;
    double *gauntxp_l = gaunt_vl + 1 * nlm;
    double *gauntym_l = gaunt_vl + 2 * nlm;
    double *gauntyp_l = gaunt_vl + 3 * nlm;
    double *gauntz_l = gaunt_vl + 4 * nlm;
#pragma omp for
    for (ish = 0; ish < nbas1; ish++) {
        l = bas1[ish * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            i0 = ao_loc1[ish];
            yx = f_u[i0];
            yy = f_u[i0 + 1];
            yz = f_u[i0 + 2];
            f_u[i0] = yy;
            f_u[i0 + 1] = yz;
            f_u[i0 + 2] = yx;
        }
    }
#pragma omp for schedule(dynamic, 4)
    for (at = 0; at < natm; at++) {
        //ish0 = atom_loc1_ao[at];
        ish1 = atom_loc1_ao[at+1];
        jsh0 = atom_loc2_ao[at];
        jsh1 = atom_loc2_ao[at+1];
        ish0 = ish1 - (jsh1 - jsh0);
        jsh = jsh0;
        for (ish = ish0; ish < ish1; ish++, jsh++) {
            l = bas2[jsh * BAS_SLOTS + ANG_OF];
            i0 = ao_loc1[ish];
            j0 = ao_loc2[jsh];
            nm = ao_loc2[jsh + 1] - j0;
            for (m = 0; m < nm; m++) {
                lm = l * l + m;
                dz_u[j0 + m] += gauntz_l[lm] * f_u[i0 + m + 1];
                dx_u[j0 + m] += gauntxm_l[lm] * f_u[i0 + m];
                dx_u[j0 + m] += gauntxp_l[lm] * f_u[i0 + m + 2];
                dy_u[j0 + m] += gauntym_l[lm] * f_u[i0 + 2 * l - m];
                dy_u[j0 + m] += gauntyp_l[lm] * f_u[i0 + 2 * l + 2 - m];
            }
        }
    }
#pragma omp for
    for (ish = 0; ish < nbas1; ish++) {
        l = bas1[ish * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            i0 = ao_loc1[ish];
            yy = f_u[i0];
            yz = f_u[i0 + 1];
            yx = f_u[i0 + 2];
            f_u[i0] = yx;
            f_u[i0 + 1] = yy;
            f_u[i0 + 2] = yz;
        }
    }
#pragma omp for
    for (jsh = 0; jsh < nbas2; jsh++) {
        l = bas2[jsh * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            j0 = ao_loc2[jsh];
            yy = dx_u[j0];
            yz = dx_u[j0 + 1];
            yx = dx_u[j0 + 2];
            dx_u[j0] = yx;
            dx_u[j0 + 1] = yy;
            dx_u[j0 + 2] = yz;

            yy = dy_u[j0];
            yz = dy_u[j0 + 1];
            yx = dy_u[j0 + 2];
            dy_u[j0] = yx;
            dy_u[j0 + 1] = yy;
            dy_u[j0 + 2] = yz;

            yy = dz_u[j0];
            yz = dz_u[j0 + 1];
            yx = dz_u[j0 + 2];
            dz_u[j0] = yx;
            dz_u[j0 + 1] = yy;
            dz_u[j0 + 2] = yz;
        }
    }
}
}

void fill_gaussian_deriv_coeff_bwd(
    double *f_u, double *d_vu, double *gaunt_vl, int nlm,
    int *atom_loc1_ao, int *ao_loc1,
    int *atm, int natm, int *bas1, int nbas1, double *env1,
    int *atom_loc2_ao, int *ao_loc2,
    int *bas2, int nbas2, double *env2
) {
#pragma omp parallel
{
    int at, l, i0, j0, lm, m, nm;
    int ish, jsh, ish0, ish1, jsh0, jsh1;
    int nao_d = ao_loc2[nbas2] - ao_loc2[0];
    double yx, yy, yz;
    double *dx_u = d_vu + 0 * nao_d;
    double *dy_u = d_vu + 1 * nao_d;
    double *dz_u = d_vu + 2 * nao_d;
    double *gauntxm_l = gaunt_vl + 0 * nlm;
    double *gauntxp_l = gaunt_vl + 1 * nlm;
    double *gauntym_l = gaunt_vl + 2 * nlm;
    double *gauntyp_l = gaunt_vl + 3 * nlm;
    double *gauntz_l = gaunt_vl + 4 * nlm;
#pragma omp for
    for (ish = 0; ish < nbas1; ish++) {
        l = bas1[ish * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            i0 = ao_loc1[ish];
            yx = f_u[i0];
            yy = f_u[i0 + 1];
            yz = f_u[i0 + 2];
            f_u[i0] = yy;
            f_u[i0 + 1] = yz;
            f_u[i0 + 2] = yx;
        }
    }
#pragma omp for
    for (jsh = 0; jsh < nbas2; jsh++) {
        l = bas2[jsh * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            j0 = ao_loc2[jsh];
            yx = dx_u[j0];
            yy = dx_u[j0 + 1];
            yz = dx_u[j0 + 2];
            dx_u[j0] = yy;
            dx_u[j0 + 1] = yz;
            dx_u[j0 + 2] = yx;

            j0 = ao_loc2[jsh];
            yx = dy_u[j0];
            yy = dy_u[j0 + 1];
            yz = dy_u[j0 + 2];
            dy_u[j0] = yy;
            dy_u[j0 + 1] = yz;
            dy_u[j0 + 2] = yx;

            j0 = ao_loc2[jsh];
            yx = dz_u[j0];
            yy = dz_u[j0 + 1];
            yz = dz_u[j0 + 2];
            dz_u[j0] = yy;
            dz_u[j0 + 1] = yz;
            dz_u[j0 + 2] = yx;
        }
    }
#pragma omp for schedule(dynamic, 4)
    for (at = 0; at < natm; at++) {
        //ish0 = atom_loc1_ao[at];
        ish1 = atom_loc1_ao[at+1];
        jsh0 = atom_loc2_ao[at];
        jsh1 = atom_loc2_ao[at+1];
        ish0 = ish1 - (jsh1 - jsh0);
        jsh = jsh0;
        for (ish = ish0; ish < ish1; ish++, jsh++) {
            l = bas2[jsh * BAS_SLOTS + ANG_OF];
            i0 = ao_loc1[ish];
            j0 = ao_loc2[jsh];
            nm = ao_loc2[jsh + 1] - j0;
            for (m = 0; m < nm; m++) {
                lm = l * l + m;
                f_u[i0 + m + 1] += gauntz_l[lm] * dz_u[j0 + m];
                f_u[i0 + m] += gauntxm_l[lm] * dx_u[j0 + m];
                f_u[i0 + m + 2] += gauntxp_l[lm] * dx_u[j0 + m];
                f_u[i0 + 2 * l - m] += gauntym_l[lm] * dy_u[j0 + m];
                f_u[i0 + 2 * l + 2 - m] += gauntyp_l[lm] * dy_u[j0 + m];
            }
        }
    }
#pragma omp for
    for (ish = 0; ish < nbas1; ish++) {
        l = bas1[ish * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            i0 = ao_loc1[ish];
            yy = f_u[i0];
            yz = f_u[i0 + 1];
            yx = f_u[i0 + 2];
            f_u[i0] = yx;
            f_u[i0 + 1] = yy;
            f_u[i0 + 2] = yz;
        }
    }
#pragma omp for
    for (jsh = 0; jsh < nbas2; jsh++) {
        l = bas2[jsh * BAS_SLOTS + ANG_OF];
        if (l == 1) {
            j0 = ao_loc2[jsh];
            yy = dx_u[j0];
            yz = dx_u[j0 + 1];
            yx = dx_u[j0 + 2];
            dx_u[j0] = yx;
            dx_u[j0 + 1] = yy;
            dx_u[j0 + 2] = yz;

            yy = dy_u[j0];
            yz = dy_u[j0 + 1];
            yx = dy_u[j0 + 2];
            dy_u[j0] = yx;
            dy_u[j0 + 1] = yy;
            dy_u[j0 + 2] = yz;

            yy = dz_u[j0];
            yz = dz_u[j0 + 1];
            yx = dz_u[j0 + 2];
            dz_u[j0] = yx;
            dz_u[j0 + 1] = yy;
            dz_u[j0 + 2] = yz;
        }
    }
}
}

void add_vh_grad_term(double *f, double *coords, double *atom_coord, int n,
                      int ig, int ix, int iy, int iz, int nf) {
#pragma omp parallel
{
    int g;
    double dx, dy, dz;
    double *f_q;
#pragma omp for
    for (g = 0; g < n; g++) {
        dx = coords[3 * g + 0] - atom_coord[0];
        dy = coords[3 * g + 1] - atom_coord[1];
        dz = coords[3 * g + 2] - atom_coord[2];
        f_q = f + nf * g;
        f_q[ix] += dx * f_q[ig];
        f_q[iy] += dy * f_q[ig];
        f_q[iz] += dz * f_q[ig];
        f_q[ig] = 0.0;
    }
}
}

void add_vh_grad_onsite(double *f, double *coords, int natm,
                        double *atom_coords, int *ar_loc,
                        int ig, int ix, int iy, int iz, int nf) {
    if (ar_loc == NULL) {
        exit(-1);
    }
#pragma omp parallel
{
    int g, a;
    double dx, dy, dz;
    double *f_q;
#pragma omp for
    for (a = 0; a < natm; a++) {
        for (g = ar_loc[a]; g < ar_loc[a+1]; g++) {
            dx = coords[3 * g + 0] - atom_coords[3 * a + 0];
            dy = coords[3 * g + 1] - atom_coords[3 * a + 1];
            dz = coords[3 * g + 2] - atom_coords[3 * a + 2];
            f_q = f + nf * g;
            f_q[ix] += dx * f_q[ig];
            f_q[iy] += dy * f_q[ig];
            f_q[iz] += dz * f_q[ig];
            f_q[ig] = 0.0;
        }
    }
}
}

void add_vh_grad_term_bwd(double *f, double *coords, double *atom_coord, int n,
                          int ig, int ix, int iy, int iz, int nf) {
#pragma omp parallel
{
    int g;
    double dx, dy, dz;
    double *f_q;
#pragma omp for
    for (g = 0; g < n; g++) {
        dx = coords[3 * g + 0] - atom_coord[0];
        dy = coords[3 * g + 1] - atom_coord[1];
        dz = coords[3 * g + 2] - atom_coord[2];
        f_q = f + nf * g;
        f_q[ig] = 0.0;
        f_q[ig] += dx * f_q[ix];
        f_q[ig] += dy * f_q[iy];
        f_q[ig] += dz * f_q[iz];
    }
}
}

void add_vh_grad_onsite_bwd(double *f, double *coords, int natm,
                            double *atom_coords, int *ar_loc,
                            int ig, int ix, int iy, int iz, int nf) {
#pragma omp parallel
{
    int g, a;
    double dx, dy, dz;
    double *f_q;
#pragma omp for
    for (a = 0; a < natm; a++) {
        for (g = ar_loc[a]; g < ar_loc[a+1]; g++) {
            dx = coords[3 * g + 0] - atom_coords[3 * a + 0];
            dy = coords[3 * g + 1] - atom_coords[3 * a + 1];
            dz = coords[3 * g + 2] - atom_coords[3 * a + 2];
            f_q = f + nf * g;
            f_q[ig] = 0.0;
            f_q[ig] += dx * f_q[ix];
            f_q[ig] += dy * f_q[iy];
            f_q[ig] += dz * f_q[iz];
        }
    }
}
}
