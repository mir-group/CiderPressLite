#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nr_cider_numint.h"
#include "spline.h"
#include "sph_harm.h"
#include "fblas.h"


void compute_spline_maps(
   double *w_rsp, double *Rg,
   int *bas, int nbas, double *env,
   int *shls_slice, int ngrids)
{
#pragma omp parallel
{
    int g, l, ish;
    double coef, expi;
    int *ibas;
    int ish0 = 0;
    int ish1 = nbas;
    int nsh = ish1 - ish0;
    double *auxo_g = (double*) malloc(ngrids * sizeof(double));
    double *spline = (double*) malloc(5 * ngrids * sizeof(double));
    double *w_rp;
#pragma omp for schedule(dynamic, 4)
    for (ish = ish0; ish < ish1; ish++) {
        ibas = bas+ish*BAS_SLOTS;
        l = ibas[ANG_OF];
        coef = env[ibas[PTR_COEFF]];
        expi = env[ibas[PTR_EXP]];
        for (g = 0; g < ngrids; g++) {
            //auxo_ig[(ish-ish0)*ngrids+g] = coef * pow(R2g[g], 0.5 * l) * exp(-expi * R2g[g]);
            auxo_g[g] = coef * pow(Rg[g], l) * exp(-expi * Rg[g] * Rg[g]);
        }
        auxo_g[ngrids-1] = 0;
        get_cubic_spline_coeff(Rg, auxo_g, spline, ngrids);
        w_rp = w_rsp + 4*(ish-ish0);
        for (g = 0; g < ngrids; g++) {
            w_rp[g*4*nsh+0] = spline[1*ngrids+g];
            w_rp[g*4*nsh+1] = spline[2*ngrids+g];
            w_rp[g*4*nsh+2] = spline[3*ngrids+g];
            w_rp[g*4*nsh+3] = spline[4*ngrids+g];
        }
    }
    free(auxo_g);
    free(spline);
}
}

void compute_alpha_splines(double *w_iap, int nalpha) {
#pragma omp parallel
{
    double *i_i = (double*) malloc(nalpha * sizeof(double));
    double *f_i = (double*) malloc(nalpha * sizeof(double));
    double *spline_pi = (double*) malloc(5 * nalpha * sizeof(double));
    int a, i;
    for (i = 0; i < nalpha; i++) {
        i_i[i] = i;
    }
#pragma omp for schedule(dynamic, 4)
    for (a = 0; a < nalpha; a++) {
        for (i = 0; i < nalpha; i++) {
            f_i[i] = 0.0;
        }
        f_i[a] = 1.0;
        get_cubic_spline_coeff(i_i, f_i, spline_pi, nalpha);
        for (i = 0; i < nalpha; i++) {
            w_iap[(i*nalpha+a)*4 + 0] = spline_pi[1*nalpha + i];
            w_iap[(i*nalpha+a)*4 + 1] = spline_pi[2*nalpha + i];
            w_iap[(i*nalpha+a)*4 + 2] = spline_pi[3*nalpha + i];
            w_iap[(i*nalpha+a)*4 + 3] = spline_pi[4*nalpha + i];
        }
    }
    free(i_i);
    free(f_i);
    free(spline_pi);
}
}

void check_splines(
    double *w_rsp, double *coords, double *atom_coord,
    double *auxo_gj, double *out_gi, int ni,
    int *bas, int nbas, int *shls_slice, int ngrids,
    int nrad, int nlm, double aparam, double dparam)
{
    // this is a test function for the spline basis
    // derivative unit tests
    int g, l, ish, p, lm, i, ir;
    double coef, expi;
    int *ibas;
    int ish0 = 0;
    int ish1 = nbas;
    int nsh = ish1 - ish0;
    double *w_sp;
    double diffr[3];
    double dr;
    for (g = 0; g < ngrids; g++) {
        diffr[0] = coords[3*g+0] - atom_coord[0];
        diffr[1] = coords[3*g+1] - atom_coord[1];
        diffr[2] = coords[3*g+2] - atom_coord[2];
        dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]);
        ir = (int) floor(log(dr / aparam + 1) / dparam);
        ir = MIN(ir, nrad-1);
        w_sp = w_rsp + 4*nsh*ir;
        i = 0;
        for (ish = ish0; ish < ish1; ish++) {
            ibas = bas+ish*BAS_SLOTS;
            l = ibas[ANG_OF];
            for (lm = l*l; lm < (l+1)*(l+1); lm++, i++) {
                //printf("%d %d %d %d\n", lm, i, ni, nlm);
                for (p = 0; p < 4; p++) {
                    out_gi[g*ni+i] += w_sp[4*ish+p] * auxo_gj[g*4*nlm+lm*4+p];
                }
            }
        }
    }
}

void solve_spline_maps(double *w_rsp, atc_ovlp_set *atco,
                       int nrad, int ngamma_max)
{
#pragma omp parallel
{
    int NSPLINE = 4;
    int r, at, l, x, y;
    double *w_sp, *chomat;
    int info;
    int ngamma;
    int natm = atco->natm;
    atc_conv_set atcc;
    double *buf = (double*) malloc(NSPLINE * ngamma_max * sizeof(double));
    int nfpr = atco->nshl_conv * NSPLINE;
#pragma omp for schedule(dynamic, 1)
    for (r = 0; r < nrad; r++) {
        for (at = 0; at < natm; at++) {
            atcc = atco->atc_convs[at];
            for (l = 0; l <= atcc.lmax; l++) {
                ngamma = atcc.l_loc[l+1] - atcc.l_loc[l];
                w_sp = w_rsp + r*nfpr + (atcc.ia_loc + atcc.l_loc[l])*NSPLINE;
                chomat = atcc.gtrans + atcc.l_loc2[l];
                for (x = 0; x < ngamma; x++) {
                    for (y = 0; y < NSPLINE; y++) {
                        buf[y*ngamma+x] = w_sp[x*NSPLINE+y];
                    }
                }
                dpotrs_(&(atco->UPLO), &ngamma, &NSPLINE,
                        chomat, &ngamma, buf, &ngamma, &info);
                for (x = 0; x < ngamma; x++) {
                    for (y = 0; y < NSPLINE; y++) {
                        w_sp[x*NSPLINE+y] = buf[y*ngamma+x];
                    }
                }
            }
        }
    }
}
}

void project_conv_onto_splines(
    double *f_qarlp, double *w_rsp, double *f_qi,
    int *shls_slice, int *atom_loc_ao,
    int *bas, int nbas, int *ao_loc,
    int nalpha, int natm, int nrad, int lm_max)
{
#pragma omp parallel
{
    // f_aqrlp must be zeros before passing, has size
    // (nalpha,natm,nrad,lm_max,4)
    int aa, at, a, ish, l, r, i, m;
    int nfpr = lm_max * 4;
    int nfpa = nrad * nfpr;
    int ish0 = 0;
    int ish1 = nbas;
    int nsh = ish1 - ish0;
    double *f_rlp, *w_sp, *w_p, *f_mp, *f_lp, *f_i;
    int nao = ao_loc[ish1] - ao_loc[ish0];
    int *ibas;
#pragma omp for
    for (aa = 0; aa < nalpha*natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        f_rlp = f_qarlp + aa * nfpa;
        f_i = f_qi + a*nao;
        for (r = 0; r < nrad; r++) {
            w_sp = w_rsp + r*4*nsh;
            f_lp = f_rlp + r*nfpr;
            for (ish = ish0; ish < ish1; ish++) {
                ibas = bas+ish*BAS_SLOTS;
                l = ibas[ANG_OF];
                f_mp = f_lp + l*l*4;
                w_p = w_sp + 4*ish;
                m = 0;
                for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                    f_mp[m++] += w_p[0] * f_i[i];
                    f_mp[m++] += w_p[1] * f_i[i];
                    f_mp[m++] += w_p[2] * f_i[i];
                    f_mp[m++] += w_p[3] * f_i[i];
                }
            }
        }
    }
}
}

void project_spline_onto_convs(
    double *f_qarlp, double *w_rsp, double *f_qi,
    int *shls_slice, int *atom_loc_ao,
    int *bas, int nbas, int *ao_loc,
    int nalpha, int natm, int nrad, int lm_max)
{
#pragma omp parallel
{
    // f_qi must be zeros before passing, has size
    // (nalpha,ngrid)
    int aa, at, a, ish, l, r, i, m;
    int nfpr = lm_max * 4;
    int nfpa = nrad * nfpr;
    int ish0 = 0;
    int ish1 = nbas;
    int nsh = ish1 - ish0;
    double *f_rlp, *w_sp, *w_p, *f_mp, *f_lp, *f_i;
    int nao = ao_loc[ish1] - ao_loc[ish0];
    int *ibas;
#pragma omp for
    for (aa = 0; aa < nalpha*natm; aa++) {
        at = aa % natm;
        a = aa / natm;
        ish0 = atom_loc_ao[at];
        ish1 = atom_loc_ao[at+1];
        f_rlp = f_qarlp + aa * nfpa;
        f_i = f_qi + a*nao;
        for (r = 0; r < nrad; r++) {
            w_sp = w_rsp + r*4*nsh;
            f_lp = f_rlp + r*nfpr;
            for (ish = ish0; ish < ish1; ish++) {
                ibas = bas+ish*BAS_SLOTS;
                l = ibas[ANG_OF];
                f_mp = f_lp + l*l*4;
                w_p = w_sp + 4*ish;
                m = 0;
                for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                    f_i[i] += w_p[0] * f_mp[m++];
                    f_i[i] += w_p[1] * f_mp[m++];
                    f_i[i] += w_p[2] * f_mp[m++];
                    f_i[i] += w_p[3] * f_mp[m++];
                }
            }
        }
    }
}
}

void compute_num_spline_contribs(
    int *num_ai, double *coords, double *atm_coords,
    double aparam, double dparam,
    int natm, int ngrids, int nrad, int *ar_loc
)
{
    int a, i, ir, g;
    double diffr[3];
    double dr;
    int *num_i;
    //printf("%lf %lf %d %d %d\n", aparam, dparam, natm, ngrids, nrad);
    for (i=0; i<natm*nrad; i++) {
        num_ai[i] = 0;
    }
    //#pragma omp parallel for private(a,i,ir,diffr,num_i,g,dr)
    for (a=0; a<natm; a++) {
        num_i = num_ai + a*nrad;
        for (g=0; g<ngrids; g++) {
            if (ar_loc == NULL || (g < ar_loc[a]) || (g >= ar_loc[a+1])) {
                diffr[0] = coords[3*g+0] - atm_coords[3*a+0];
                diffr[1] = coords[3*g+1] - atm_coords[3*a+1];
                diffr[2] = coords[3*g+2] - atm_coords[3*a+2];
                dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]);
                ir = (int) floor(log(dr / aparam + 1) / dparam);
                ir = MIN(ir, nrad-1);
                num_i[ir] += 1;
            }
        }
    }
}

#define ASSIGN_IND_ORDER \
            diffr[0] = coords[3*g+0] - atm_coord[0]; \
            diffr[1] = coords[3*g+1] - atm_coord[1]; \
            diffr[2] = coords[3*g+2] - atm_coord[2]; \
            dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]); \
            ir = (int) floor(log(dr / aparam + 1) / dparam); \
            ir = MIN(ir, nrad-1); \
            gp = loc_i[ir] + num_i_tmp[ir]; \
            ind_ord_fwd[gp] = g; \
            ind_ord_bwd[g] = gp; \
            coords_ord[3*gp+0] = coords[3*g+0]; \
            coords_ord[3*gp+1] = coords[3*g+1]; \
            coords_ord[3*gp+2] = coords[3*g+2]; \
            num_i_tmp[ir] += 1;

void compute_spline_ind_order(
    int *loc_i, double *coords, double *atm_coord,
    double *coords_ord, int *ind_ord_fwd, int *ind_ord_bwd,
    double aparam, double dparam,
    int ngrids, int nrad, int *ar_loc, int a
)
{
    int i, g, ir, gp;
    double dr;
    double diffr[3];
    int *num_i_tmp = malloc(nrad*sizeof(int));
    for (i=0; i<nrad; i++) {
        num_i_tmp[i] = 0;
    }
    if (ar_loc == NULL) {
        for (g = 0; g < ngrids; g++) {
            ASSIGN_IND_ORDER;
        }
    } else {
        for (g = 0; g < ar_loc[a]; g++) {
            ASSIGN_IND_ORDER;
        }
        for (g = ar_loc[a+1]; g < ngrids; g++) {
            ASSIGN_IND_ORDER;
        }
    }
    free(num_i_tmp);
}

void compute_spline_bas(
    double *auxo_agi, int *ind_ag,
    double *coords, double *atm_coords,
    int natm, int ngrids, int nrad, int nlm,
    double aparam, double dparam)
{
#pragma omp parallel
{
    int g, lm, ag, at;
    int ir;
    int i;
    double dr, dr2, dr3;
    double diffr[3];
    double *auxo_i;
    double *ylm = (double*)malloc(nlm*sizeof(double));
    sphbuf buf = setup_sph_harm_buffer(nlm);
#pragma omp for
    for (ag = 0; ag < natm*ngrids; ag++) {
        at = ag / ngrids;
        g = ag % ngrids;
        auxo_i = auxo_agi + ag*4*nlm;
        diffr[0] = coords[3*g+0] - atm_coords[3*at+0];
        diffr[1] = coords[3*g+1] - atm_coords[3*at+1];
        diffr[2] = coords[3*g+2] - atm_coords[3*at+2];
        dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]);
        diffr[0] /= dr;
        diffr[1] /= dr;
        diffr[2] /= dr;
        recursive_sph_harm(buf, diffr, ylm);
        ir = (int) floor(log(dr / aparam + 1) / dparam);
        ir = MIN(ir, nrad-1);
        dr -= aparam * (exp(dparam * (double)ir) - 1);
        ind_ag[ag] = ir;
        i = 0;
        dr2 = dr * dr;
        dr3 = dr2 * dr;
        for (lm = 0; lm < nlm; lm++) {
            auxo_i[i++] = ylm[lm];
            auxo_i[i++] = ylm[lm] * dr;
            auxo_i[i++] = ylm[lm] * dr2;
            auxo_i[i++] = ylm[lm] * dr3;
        }
    }
    free(ylm);
    free_sph_harm_buffer(buf);
}
}

void compute_spline_bas_deriv(
    double *auxo_vagi, int *ind_ag,
    double *coords, double *atm_coords,
    int natm, int ngrids, int nrad, int nlm,
    double aparam, double dparam)
{
#pragma omp parallel
{
    int bdisp = natm*ngrids*4*nlm;
    int ldisp;
    int g, lm, ag, at;
    int ir;
    int i;
    double dr, dr2, dr3;
    double diffr[3];
    double *auxox_i;
    double *auxoy_i;
    double *auxoz_i;
    double *ylm = (double*)malloc(nlm*sizeof(double));
    double *dylm = (double*)malloc(3*nlm*sizeof(double));
    double *dylmx = dylm + 0*nlm;
    double *dylmy = dylm + 1*nlm;
    double *dylmz = dylm + 2*nlm;
    sphbuf buf = setup_sph_harm_buffer(nlm);
#pragma omp for
    for (ag = 0; ag < natm*ngrids; ag++) {
        at = ag / ngrids;
        g = ag % ngrids;
        ldisp = ag*4*nlm;
        auxox_i = auxo_vagi + ldisp;
        auxoy_i = auxo_vagi + ldisp + bdisp;
        auxoz_i = auxo_vagi + ldisp + 2*bdisp;
        diffr[0] = coords[3*g+0] - atm_coords[3*at+0];
        diffr[1] = coords[3*g+1] - atm_coords[3*at+1];
        diffr[2] = coords[3*g+2] - atm_coords[3*at+2];
        dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]);
        diffr[0] /= dr;
        diffr[1] /= dr;
        diffr[2] /= dr;
        recursive_sph_harm_deriv(buf, diffr, ylm, dylmx);
        // TODO smarter way to avoid div by 0
        for (lm = 0; lm < nlm; lm++) {
            dylmx[lm] /= dr + 1e-10;
            dylmy[lm] /= dr + 1e-10;
            dylmz[lm] /= dr + 1e-10;
        }
        ir = (int) floor(log(dr / aparam + 1) / dparam);
        ir = MIN(ir, nrad-1);
        dr -= aparam * (exp(dparam * (double)ir) - 1);
        ind_ag[ag] = ir;
        i = 0;
        dr2 = dr * dr;
        dr3 = dr2 * dr;
        for (lm = 0; lm < nlm; lm++) {
            auxox_i[i++] = dylmx[lm];
            auxox_i[i++] = ylm[lm] * diffr[0] + dylmx[lm] * dr;
            auxox_i[i++] = 2 * ylm[lm] * dr * diffr[0] + dylmx[lm] * dr2;
            auxox_i[i++] = 3 * ylm[lm] * dr2 * diffr[0] + dylmx[lm] * dr3;
        }
        i = 0;
        for (lm = 0; lm < nlm; lm++) {
            auxoy_i[i++] = dylmy[lm];
            auxoy_i[i++] = ylm[lm] * diffr[1] + dylmy[lm] * dr;
            auxoy_i[i++] = 2 * ylm[lm] * dr * diffr[1] + dylmy[lm] * dr2;
            auxoy_i[i++] = 3 * ylm[lm] * dr2 * diffr[1] + dylmy[lm] * dr3;
        }
        i = 0;
        for (lm = 0; lm < nlm; lm++) {
            auxoz_i[i++] = dylmz[lm];
            auxoz_i[i++] = ylm[lm] * diffr[2] + dylmz[lm] * dr;
            auxoz_i[i++] = 2 * ylm[lm] * dr * diffr[2] + dylmz[lm] * dr2;
            auxoz_i[i++] = 3 * ylm[lm] * dr2 * diffr[2] + dylmz[lm] * dr3;
        }
    }
    free(ylm);
    free(dylm);
    free_sph_harm_buffer(buf);
}
}

void contract_grad_terms(double *excsum, double *f_g, int natm,
                         int a, int v, int ngrids, int *ga_loc) {
    double *tmp = (double*) calloc(natm, sizeof(double));
    int ib;
#pragma omp parallel
{
    int ia;
    int g;
#pragma omp for
    for (ia = 0; ia < natm; ia++) {
        //printf("%d %d %d %d\n", ia, natm, ga_loc[ia], ga_loc[ia+1]);
        for (g = ga_loc[ia]; g < ga_loc[ia+1]; g++) {
            tmp[ia] += f_g[g];
        }
    }
}
    for (ib = 0; ib < natm; ib++) {
        excsum[ib*3+v] += tmp[ib];
        excsum[a*3+v] -= tmp[ib];
    }
}

void compute_mol_convs_single(
    double *f_gq, double *f_rqlp,
    double *auxo_gi, int *loc_i,
    int *ind_ord_fwd,
    int nalpha, int nrad,
    int ngrids, int nlm, int maxg
)
{
#pragma omp parallel
{
    int nfpr = nalpha * nlm * 4;
    int nfpa = nlm * 4;
    int q, ir, g;
    double *f_qlp;
    double BETA = 0;
    double ALPHA = 1;
    char NTRANS = 'N';
    char TRANS = 'T';
    double *f_gq_buf = malloc(nalpha*maxg*sizeof(double));
    double *f_gq_tmp, *auxo_glp;
    int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
    for (ir = 0; ir < nrad-1; ir++) {
        f_qlp = f_rqlp + ir * nfpr;
        auxo_glp = auxo_gi + loc_i[ir] * nfpa;
        ng = loc_i[ir+1] - loc_i[ir];
        //printf("ERROR %d %d %d %d %d\n", ir, ng, maxg, nrad, nalpha);
        dgemm_(&TRANS, &NTRANS, &nalpha, &ng, &nfpa,
               &ALPHA, f_qlp, &nfpa,
               auxo_glp, &nfpa, &BETA,
               f_gq_buf, &nalpha);
        gq_ind = 0;
        for (g=loc_i[ir]; g<loc_i[ir+1]; g++) {
            gp = ind_ord_fwd[g];
            f_gq_tmp = f_gq + gp*nalpha;
            for (q=0; q<nalpha; q++, gq_ind++) {
                f_gq_tmp[q] += f_gq_buf[gq_ind];
            }
        }
    }
    free(f_gq_buf);
}
}

void compute_pot_convs_single(
    double *f_gq, double *f_rqlp,
    double *auxo_gi, int *loc_i,
    int *ind_ord_fwd,
    int nalpha, int nrad,
    int ngrids, int nlm, int maxg
)
{
#pragma omp parallel
{
    int nfpr = nalpha * nlm * 4;
    int nfpa = nlm * 4;
    int q, ir, g;
    double *f_qlp;
    double BETA = 0.0;
    double ALPHA = 1.0;
    char NTRANS = 'N';
    char TRANS = 'T';
    double *f_gq_buf = malloc(nalpha*maxg*sizeof(double));
    double *f_gq_tmp, *auxo_glp;
    int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
    for (ir = 0; ir < nrad-1; ir++) {
        f_qlp = f_rqlp + ir * nfpr;
        auxo_glp = auxo_gi + loc_i[ir] * nfpa;
        ng = loc_i[ir+1] - loc_i[ir];
        gq_ind = 0;
        for (g=loc_i[ir]; g<loc_i[ir+1]; g++) {
            gp = ind_ord_fwd[g];
            f_gq_tmp = f_gq + gp*nalpha;
            for (q=0; q<nalpha; q++, gq_ind++) {
                f_gq_buf[gq_ind] = f_gq_tmp[q];
            }
        }
        dgemm_(&NTRANS, &TRANS, &nfpa, &nalpha, &ng,
               &ALPHA, auxo_glp, &nfpa,
               f_gq_buf, &nalpha, &BETA,
               f_qlp, &nfpa);
    }
    free(f_gq_buf);
}
}

void compute_mol_convs(
    double *f_qg, double *f_qarlp,
    double *auxo_agi, int *ind_ag,
    int nalpha, int natm, int nrad, int ngrids, int nlm)
{
#pragma omp parallel
{
    int nfpr = nlm * 4;
    int nfpa = nrad * nfpr;
    int q, at, aa, ir, ag, g, qg_ind;
    double *auxo_i, *f_lp;
    int inc = 1;
#pragma omp for
    for (q=0; q<nalpha; q++) {
        for (at=0; at<natm; at++) {
            aa = q*natm+at;
            for (g=0; g<ngrids; g++) {
                ag = at*ngrids+g;
                auxo_i = auxo_agi + ag*nfpr;
                ir = ind_ag[ag];
                f_lp = f_qarlp + aa * nfpa + ir * nfpr;
                qg_ind = q*ngrids+g;
                //for (lmp=0; lmp<nfpr; lmp++) {
                //    f_qg[qg_ind] += f_lp[lmp] * auxo_i[lmp];
                //}
                f_qg[qg_ind] += ddot_(&nfpr, f_lp, &inc, auxo_i, &inc);
            }
        }
    }
}
}

void compute_pot_convs(
    double *f_qg, double *f_qarlp,
    double *auxo_agi, int *ind_ag,
    int nalpha, int natm, int nrad, int ngrids, int nlm)
{
#pragma omp parallel
{
    int nfpr = nlm * 4;
    int nfpa = nrad * nfpr;
    int q, at, aa, lmp, ir, ag, g;
    double *auxo_i, *f_lp;
    double f_curr;
#pragma omp for
    for (q=0; q<nalpha; q++) {
        for (at=0; at<natm; at++) {
            aa = q*natm+at;
            for (g=0; g<ngrids; g++) {
                ag = at*ngrids+g;
                auxo_i = auxo_agi + ag*nfpr;
                ir = ind_ag[ag];
                f_lp = f_qarlp + aa * nfpa + ir * nfpr;
                f_curr = f_qg[q*ngrids+g];
                #pragma omp simd
                for (lmp=0; lmp<nfpr; lmp++) {
                    f_lp[lmp] += f_curr * auxo_i[lmp];
                }
            }
        }
    }
}
}

void reduce_angc_to_ylm(
    double *theta_rlmq,  double *y_glm, double *theta_gq,
    int *rad_loc, int *ylm_loc,
    int nalpha, int nrad, int ngrids, int nlm
)
{
#pragma omp parallel
{
    double ALPHA=1.0;
    double BETA=0.0;
    char NTRANS = 'N';
    char TRANS = 'T';
    int r, nw;
    double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
    for (r = 0; r < nrad; r++) {
        y_wlm = y_glm + ylm_loc[r]*nlm;
        nw = rad_loc[r+1] - rad_loc[r];
        theta_lmq = theta_rlmq + r*nlm*nalpha;
        theta_wq = theta_gq + rad_loc[r]*nalpha;
        dgemm_(&NTRANS, &TRANS, &nalpha, &nlm, &nw,
               &ALPHA, theta_wq, &nalpha,
               y_wlm, &nlm, &BETA,
               theta_lmq, &nalpha);
    }
}
}

void reduce_ylm_to_angc(
    double *theta_rlmq,  double *y_glm, double *theta_gq,
    int *rad_loc, int *ylm_loc,
    int nalpha, int nrad, int ngrids, int nlm
)
{
#pragma omp parallel
{
    double ALPHA=1.0;
    double BETA=0.0;
    char NTRANS = 'N';
    char TRANS = 'T';
    int r, nw;
    double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
    for (r = 0; r < nrad; r++) {
        y_wlm = y_glm + ylm_loc[r]*nlm;
        nw = rad_loc[r+1] - rad_loc[r];
        theta_lmq = theta_rlmq + r*nlm*nalpha;
        theta_wq = theta_gq + rad_loc[r]*nalpha;
        dgemm_(&NTRANS, &NTRANS, &nalpha, &nw, &nlm,
               &ALPHA, theta_lmq, &nalpha,
               y_wlm, &nlm, &BETA,
               theta_wq, &nalpha);
    }
}
}

void contract_rad_to_orb(
    double *theta_rlmq, double *p_uq,
    int *ra_loc, double *rads, int nrad, int nlm,
    int *atm, int natm, int *bas, int nbas, double *env,
    int *ao_loc, int *atom_loc_ao, double *alphas, int nalpha
)
{
#pragma omp parallel
{
    int a, ish, i0, L0, nm, l, at;
    double *p_mq, *theta_mq;
    int *ibas;
    double coef, beta, val;
    int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
    for (ish = 0; ish < nbas; ish++) {
        ibas = bas + ish * BAS_SLOTS;
        at = ibas[ATOM_OF];
        l = ibas[ANG_OF];
        coef = env[ibas[PTR_COEFF]];
        beta = env[ibas[PTR_EXP]];
        i0 = ao_loc[ish];
        nm = 2*l+1;
        L0 = l*l;
        for (r = ra_loc[at]; r < ra_loc[at+1]; r++) {
            val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
            theta_mq = theta_rlmq + nalpha * (r*nlm + L0);
            p_mq = p_uq + i0*nalpha;
            mq = 0;
            for (m = 0; m < nm; m++) {
                for (q = 0; q < nalpha; q++, mq++) {
                    p_mq[mq] += val * theta_mq[mq];
                }
            }
        }
    }
}
}

void contract_orb_to_rad(
    double *theta_rlmq, double *p_uq,
    int *ar_loc, double *rads, int nrad, int nlm,
    int *atm, int natm, int *bas, int nbas, double *env,
    int *ao_loc, int *atom_loc_ao, double *alphas, int nalpha
)
{
#pragma omp parallel
{
    int a, ish, i0, L0, nm, l, at;
    double *p_mq, *theta_mq;
    int *ibas;
    double coef, beta, val;
    int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
    for (r = 0; r < nrad; r++) {
        at = ar_loc[r];
        for (ish = atom_loc_ao[at]; ish < atom_loc_ao[at+1]; ish++) {
            ibas = bas + ish * BAS_SLOTS;
            l = ibas[ANG_OF];
            coef = env[ibas[PTR_COEFF]];
            beta = env[ibas[PTR_EXP]];
            i0 = ao_loc[ish];
            nm = 2*l+1;
            L0 = l*l;
            val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
            theta_mq = theta_rlmq + nalpha * (r*nlm + L0);
            p_mq = p_uq + i0*nalpha;
            mq = 0;
            for (m = 0; m < nm; m++) {
                for (q = 0; q < nalpha; q++, mq++) {
                    theta_mq[mq] += val * p_mq[mq];
                }
            }
        }
    }
}
}
