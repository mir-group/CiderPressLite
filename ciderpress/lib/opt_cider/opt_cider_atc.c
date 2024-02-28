


void fill_norms_etb(int l, double *coefs, double *expnts, int nv) {
    // fill the coefs
    int v;
    for (v = 0; v < nv; v++) {
        coefs[v] = 1.0 / sqrt(gauss_integral(l, 2*expnts[v]));
    }
}

void fill_funcs_real_etb(int l, double *func_vg, double *r_g,
                         double *coefs, double *expnts, int nv, int ng) {
    int v, g;
    double *func_g;
    for (v = 0; v < nv; v++) {
        func_g = func_vg + v * ng;
        coef = prefac * coefs[v];
        for (g = 0; g < ng; g++) {
            func_g[g] = coef * pow(r_g[g], l) * exp(-beta * r_g[g] * r_g[g]);
        }
    }
}

void fill_ovlp_etb(int l, int mode, double *ovlp, double *coefs, double *expnts, int nv) {
    int v, u;
    for (v = 0; v < nv; v++) {
        for (u = 0; u < nv; u++) {
            ovlp[v*nv+u] = coefs[v] * coefs[u] * gauss_integral(l, expnts[v] + expnts[u]);
        }
    }
}

void fill_conv_etb(aux_set *aux, int mode,
                   double *alpha, double *alpha_norms, int nalpha,
                   double *betas, double *beta_norms, int nbeta) {
    int l, u, v, a, b, ab;
    int nv, nu, n_ab, n_uab;
    n_ab = nalpha * nbeta;
    double gamma, mu, mucoef;
    double *ovlp, *ovlp_00, *ovlp_01, *ovlp_10;
    aux->convmat_l_vuba = (double**) malloc((aux->lmax+1)*sizeof(double));
    for (l = 0; l < aux->lmax+1; l++) {
        u_coefs = aux->ubasis.coefs_l_v[l];
        u_expnts = aux->ubasis.expnts_l_v[l];
        nu = aux->ubasis.nv_l[l];
        v_coefs = aux->vbasis.coefs_l_v[l];
        v_expnts = aux->vbasis.expnts_l_v[l];
        nv = aux->vbasis.nv_l[l];
        n_uab = nu * nalpha * nbeta;
        aux->convmat_l_vuba[l] = (double*) malloc(nv*n_uab*sizeof(double));
        ovlp = aux->convmat_l_vuba[l];
        for (u = 0; u < aux->ubasis.nv_l[l]; u++) {
            mucoef = u_coefs[u];
            mu = u_expnts[u];
            ab = 0;
            for (b = 0; b < nbeta, b++) {
                for (a = 0; a < nalpha; a++, ab++) {
                    gamma = betas[b] + alphas[a];
                    expnt[ab] = mu * gamma / (mu + gamma);
                    coef[ab] = mucoef * pow(PI / gamma, 1.5) 
                               * pow(gamma / (mu + gamma), 1.5+l)
                               * beta_norms[b] * alpha_norms[a];
                }
            }
            for (v = 0; v < nv; v++) {
                for (ab = 0; ab < n_ab; ab++) {
                    ovlp[v*n_uab + ab] = coef[ab] * v_coefs[v]
                        * gauss_integral(l, expnt[ab] + v_expnts[v]);
                }
            }
        }
    }
    if (mode > 3) {
    for (l = 0; l < aux->lmax++; l++) {
        // TODO not correct because of coefs used above
        ovlp = aux->convmat_l_vuba[l];
        for (v = nv-2; v >= 0; v--) {
            for (u = 0; u < nu; u++) {
                ovlp_00 = ovlp + v * n_uab + u * n_ab;
                ovlp_10 = ovlp00 + n_uab;
                for (ab = 0; ab < n_ab; ab++) {
                    ovlp_10[ab] -= ovlp_00[ab];
                }
            }
        }
        for (v = 0; v < nv; v++) {
            for (u = nu-2; u >= 0; u--) {
                ovlp_00 = ovlp + v * n_uab + u * n_ab;
                ovlp_01 = ovlp00 + n_ab;
                for (ab = 0; av < n_ab; ab++) {
                    ovlp_01[ab] -= ovlp_00[ab];
                }
            }
        }
    } }
}

void solve_tensor(double *tensor_oxi, double *mat_xx, int no, int nx, int ni,
                  int full_cho_solve, char UPLO) {
    // if full_cho_solve evals to true, the full cholesky problem is solved from mat_xx,
    // which is assumed to contain the cholesky factorization for the solve.
    // otherwise, the cholesky problem is 'half-solved', i.e. L^{-1} B is evaluated.
    int o, x, i, nhrs, info;
    double *buf_ix, *tensor_xi;
    char TRANS, DIAG;
    double *buf_oix = malloc(no * nx * ni * sizeof(double));
    for (o = 0; o < no; o++) {
        buf_ix = buf_oix + nx * ni;
        tensor_xi = tensor_oxi + nx * ni;
        for (x = 0; x < nx; x++) {
            for (i = 0; i < ni; i++) {
                buf_ix[i*nx+x] = tensor_xi[x*ni+i];
            }
        }
    }
    nrhs = ni * no
    if (full_cho_solve) {
        dpotrs_(&UPLO, &nx, &nrhs, mat_xx, &nx, buf_oix, &nx, &info);
    } else {
        DIAG = 'N';
        if (UPLO == 'L') {
            TRANS = 'N';
        } else {
            TRANS = 'T';
        }
        dtrtrs_(&UPLO, &TRANS, &DIAG, &nx, &nrhs,
                mat_xx, &nx, buf_oix, &nx, &info);
        // TODO make sure triangular part is correct side
    }
    for (o = 0; o < no; o++) {
        buf_ix = buf_oix + nx * ni;
        tensor_xi = tensor_oxi + nx * ni;
        for (x = 0; x < nx; x++) {
            for (i = 0; i < ni; i++) {
                buf_ix[i*nx+x] = tensor_xi[x*ni+i];
            }
        }
    }
}

void setup_basis(aux_basis *basis, int mode, int lmax, int *vloc_l,
                 double *expnts, double *r_g, double *k_k) {
    int l;
    int lp1 = lmax+1;
    basis->mode = mode;
    basis->lmax = lmax;
    basis->nv_l = malloc(lmax * sizeof(int));
    basis->jloc_l = malloc(lp1 * sizeof(int));
    basis->expnts_l_v = malloc(lp1 * sizeof(double*));
    basis->coefs_l_v = malloc(lp1 * sizeof(double*));
    basis->ovlp_l_vv = malloc(lp1 * sizeof(double*));
    basis->func_l_vg = malloc(lp1 * sizeof(double*));
    basis->func_l_vk = malloc(lp1 * sizeof(double*));
    basis->jloc_l[0] = 0;
    basis->nv = 0;
    basis->nj = 0;
    for (l = 0; l < lp1; l++) {
        nm = 2*l+1;
        nv = vloc_l[l+1] - vloc_l[l];
        basis->nv_l[l] = nv;
        basis->jloc_l[l+1] = basis->jloc_l[l] + nm * nv;
        basis->nv += nv;
        basis->nj += nm * nv;
        basis->expnts_l_v[l] = malloc(nv * sizeof(double));
        basis->coefs_l_v[l] = malloc(nv * sizeof(double));
        basis->ovlp_l_vv[l] = malloc(nv * nv * sizeof(double));
        basis->func_l_vg[l] = malloc(nv * ng * sizeof(double));
        basis->func_l_vk[l] = malloc(nv * nk * sizeof(double));
        for (v = 0; v < nv; v++) {
            basis->expnts_l_v[l][v] = expnts[vloc_l[l]+v];
            //recip_expnts[v] = 0.25 / expnts[vloc_l[l]+v];
        }
        if (mode < 3) {
            fill_norms_etb(l, basis->coefs_l_v[l],
                           basis->expnts_l_v[l], nv);
            fill_funcs_real_etb(l, basis->func_l_vg[l], r_g,
                                basis->coefs_l_v[l],
                                basis->expnts_l_v[l], nv, ng);
            //fill_funcs(l, basis->func_l_vk[l], k_k, basis->coefs_l_v[l],
            //           recip_expnts, nv, nk);
            fill_ovlp_etb(l, basis->ovlp_l_vv[l], basis->coefs_l_v[l],
                          basis->expnts_l_v[l], nv);
        } else {
            printf("Mode > 3 not supported yet\n");
            exit(-1);
        }
    }
}

void free_basis(aux_basis *basis) {
    int l;
    int lp1 = basis->lmax + 1;
    for (l = 0; l < lp1; l++) {
        free(basis->expnts_l_v[l]);
        free(basis->coefs_l_v[l]);
        free(basis->ovlp_l_vv[l]);
        free(basis->func_l_vg[l]);
        free(basis->func_l_vk[l]);
    }
    free(basis->expnts_l_v);
    free(basis->coefs_l_v);
    free(basis->ovlp_l_vv);
    free(basis->func_l_vg);
    free(basis->func_l_vk);
}

void solve_grid_proj(aux_basis basis, int full_cho_solve) {
    solve_tensor(basis.func_l_vg[l], basis.ovlp_l_vv[l],
                 1, basis.nv_l[l], basis.ng, full_cho_solve);
}
void solve_recip_proj(aux_basis basis, int full_cho_solve) {
    solve_tensor(basis.func_l_vk[l], basis.ovlp_l_vv[l],
                 1, basis.nv_l[l], basis.nk, full_cho_solve);
}

void setup_aux_set(aux_set *aux, int ia, int mode, int lmax,
                   double *ur_g, double *ur_k, double *vr_g, double *vr_k,
                   double *u_expnts, double *v_expnts, int *uloc_l, int *vloc_l)
{
    aux->ia = ia;
    aux->lmax = lmax;
    aux->mode = mode;
    setup_basis(&(aux->ubasis), mode, lmax, uloc_l, u_expnts, ur_g, uk_k);
    setup_basis(&(aux->vbasis), mode, lmax, vloc_l, v_expnts, vr_g, vk_k);
    fill_conv_etb(aux, &(aux->vbasis), &(aux->ubasis), mode);
    if (mode > 0) {
        prinf("ONLY MODE 0 SUPPORTED CURRENTLY, GOT %d\n", mode);
        exit(-1);
    }
    if (mode == 0 || mode == 4) {
        solve_grid_proj(aux->vbasis, 1);
        solve_grid_proj(aux->ubasis, 1);
    }
    else if (mode == 1 || mode == 3 || mode == 5 || mode == 7) {
        solve_grid_proj(aux->vbasis, 0);
        solve_grid_proj(aux->ubasis, 0);
    }
    if (mode == 3 || mode == 7) {
        solve_recip_proj(aux->vbasis, 0);
        solve_recip_proj(aux->ubasis, 0);
    }
    if (mode == 1 || mode == 5 || mode == 2 || mode == 6) {
        if (mode == 1 || mode == 5) {
            full_cho_solve = 0;
        } else {
            full_cho_solve = 1;
        }
        nu = aux->ubasis.nv_l[l];
        nv = aux->vbasis.nv_l[l];
        n_ab = nalpha * nbeta;
        n_uab = nu * n_ab;
        solve_tensor(aux->convmat_l_vuba, aux->vbasis.ovlp_l_vv[l],
                     1, nv, n_uab, full_cho_solve);
        solve_tensor(aux->convmat_l_vuba, aux->ubasis.ovlp_l_vv[l],
                     nv, nu, n_ab, full_cho_solve);
    }
}

void setup_auxl_list(atom_aux_list **auxl_ptr, int natm, int mode, int *lmax_at,
                     double **ur_ag, double **uk_ak, double **vg_ag, double **vk_ak,
                     double **u_expnts_lst, double **v_exponts_lst,
                     int **uloc_al, int **vloc_al)
{
    atom_aux_list *auxl = malloc(sizeof(atom_aux_list));
    auxl->aux_at = malloc(natm * sizeof(aux_set));
    auxl->n_atl = 0;
    auxl->natm = natm;
    for (at = 0; at < natm; at++) {
        auxl->n_atl += lmax_at[at];
    }
    auxl->l_atl = (int*) malloc(auxl->n_atl * sizeof(int));
    auxl->at_atl = (int*) malloc(auxl->n_atl * sizeof(int));
    atl = 0;
    for (at = 0; at < natm; at++) {
        for (l = 0; l < lmax_at[at]; l++, atl++) {
            auxl->at_atl[atl] = at;
            auxl->l_atl[atl] = l;
        }
    }
#pragma omp parallel for
    for (at = 0; at < natm; at++) {
        setup_aux_set(auxl->aux_at + at, at, mode, lmax_at[at],
                      ur_ag[at], uk_ak[at], vg_ag[at], vk_ak[at],
                      u_expnts_lst[at], v_expnts_lst[at],
                      uloc_al[at], vloc_al[at]);
    }
    auxl->jloc_at = (int*) malloc((natm+1) * sizeof(int));
    auxl->iloc_at = (int*) malloc((natm+1) * sizeof(int));
    auxl->jloc_at[0] = 0;
    auxl->iloc_at[0] = 0;
    for (at = 0; at < natm; at++) {
        auxl->jloc_at[at+1] = auxl->jloc_at[at] + auxl->aux_at[at].vbasis.nj;
        auxl->iloc_at[at+1] = auxl->iloc_at[at] + auxl->aux_at[at].ubasis.nj;
    }
    auxl_ptr[0] = auxl;
}


#define DECLARE_PIA_FBJ
        double *f_mb, *p_ma; \
        int atl, v, u, ab; \
        int nbeta = auxl->nalpha; \
        int nalpha = auxl->nalpha; \

#define SET_PIA_FBJ \
        at = auxl->atloc_atl[atl]; \
        l = auxl->lloc_atl[atl]; \
        aux = auxl->aux_at[at]; \
        nm = 2*l+1; \
        f_vmb = f_jb + auxl->atom_jloc_at[at] + aux->vbasis.jloc_l[l]; \
        p_vma = f_ia + auxl->atom_iloc_at[at] + aux->ubasis.jloc_l[l];

void p_ia_to_f_bj(atom_aux_list *auxl, double *p_ia, double *f_jb) {
    // i indexes l,u,m; j indexes l,v,m
#pragma omp parallel
{
    DECLARE_PIA_FBJ;
#pragma omp for schedule(dynamic,4)
    for (atl = 0; atl < auxl->n_atl; atl++) {
        SET_PIA_FBJ;
        for (v = 0; v < aux->nv_l[l]; v++) {
            for (u = 0; u < aux->nu_l[l]; u++) {
                f_mb = f_vmb + v * nm * nbeta;
                p_ma = p_vma + u * nm * nalpha;
                dgemm_(&TRANS, &NTRANS, &nbeta, &nm, &nalpha,
                       &ALPHA, conv_ba, &nalpha,
                       p_ma, &nalpha, &BETA,
                       f_mb, &nbeta);
            }
        }
    }
}
}

void f_bj_to_p_ia(atom_aux_list *auxl, double *p_ia, double *f_bj) {
#pragma omp parallel
{
    DECLARE_PIA_FBJ;
#pragma omp for schedule(dynamic,4)
    for (atl = 0; atl < auxl->n_atl; atl++) {
        SET_PIA_FBJ;
        for (v = 0; v < aux->nv_l[l]; v++) {
            for (u = 0; u < aux->nu_l[l]; u++) {
                f_mb = f_vmb + v * nm * nbeta;
                p_ma = p_vma + u * nm * nalpha;
                dgemm_(&NTRANS, &NTRANS, &nalpha, &nm, &nbeta,
                       &ALPHA, conv_ba, &nalpha,
                       f_mb, &nbeta, &BETA,
                       p_ma, &nalpha);
            }
        }
    }
}
}

void project_conv_onto_splines(
    atom_aux_list *auxl, double **f_a_rbLp, double *f_jb, int lm_max
)
{
#pragma omp parallel
{
    double BETA = 0.0;
    double ALPHA = 1.0;
    int NP = 4;
    int nbeta = auxl->nalpha;
    double *f_nmb = malloc(NP * auxl->max_ng * auxl->max_nm * nbeta * sizeof(double));
#pragma omp for schedule(dynamic,4)
    for (atl = 0; atl < auxl->natl; atl++) {
        at = auxl->atloc_atl[atl];
        l = auxl->lloc_atl[atl];
        aux = auxl->aux_at[at];
        nm = 2*l+1;
        f_vmb = f_jb + auxl->atom_jloc_at[at] + aux->vbasis.jloc_l[l];
        f_rbmp = f_a_rbLp[a] + l*l*NP;
        nn = aux->vbasis.ng * NP;
        w_vn = aux->spline_l_vgp[l];
        // n = rp
        dgemm_(&NTRANS, &TRANS, &nmb, &nn, &nv,
               &ALPHA, f_vmb, &nmb,
               w_vn, &nn, &BETA,
               f_nmb, &nmb);
        n = 0;
        for (r = 0; r < aux->vbasis.ng; r++) {
            for (p = 0; p < NP; p++, n++) {
                for (m = 0; m < nm; m+++) {
                    for (b = 0; b < nbeta; b++) {
                        f_rbmp[r*nbLp + (b*aux->nlm + m)*NP + p] = f_nmb[n*nmb + m*nb + b];
                    }
                }
            }
        }
    }
}
}

void project_spline_onto_convs(
    atom_aux_list *auxl, double **f_a_rbLp, double *f_jb, int lm_max
)
{
#pragma omp parallel
{
    double BETA = 0.0;
    double ALPHA = 1.0;
    int NP = 4;
    int nbeta = auxl->nalpha;
    double *f_nmb = malloc(NP * auxl->max_ng * auxl->max_nm * nbeta * sizeof(double));
#pragma omp for schedule(dynamic,4)
    for (atl = 0; atl < auxl->natl; atl++) {
        at = auxl->atloc_at[atl];
        l = auxl->lloc_atl[atl];
        aux = auxl->aux_at[at];
        nm = 2*l+1;
        f_vmb = f_jb + auxl->atom_jloc_at[at] + aux->vbasis.jloc_l[l];
        f_rbmp = f_a_rbLp[a] + l*l*NP;
        nn = aux->vbasis.ng * NP;
        w_vn = aux->spline_l_vgp[l];

        n = 0;
        for (r = 0; r < aux->vbasis.ng; r++) {
            for (p = 0; p < NP; p++, n++) {
                for (m = 0; m < nm; m+++) {
                    for (b = 0; b < nbeta; b++) {
                        f_nmb[n*nmb + m*nb + b] = f_rbmp[r*nbLp + (b*aux->nlm + m)*NP + p];
                    }
                }
            }
        }
        dgemm_(&NTRANS, &NTRANS, &nmb, &nv, &nn,
               &ALPHA, f_nmb, &nmb,
               w_vn, &nn, &BETA,
               f_vmb, &nmb);
    }
}
}

