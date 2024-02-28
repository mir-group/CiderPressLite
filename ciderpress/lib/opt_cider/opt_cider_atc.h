#ifndef OPT_CIDER_ATC_H
#define OPT_CIDER_ATC_H

/** Modes:
 * 0-3 - Standard Gaussian basis
 * 0 - Perform convolution in basis space, apply ovlp^-1 to func_l_*g
 * 1 - (NOT IMPLEMENTED) Perform convolution in basis space,
 *     apply ovlp^-0.5 to func_l_*g and ovlp^-0.5 to convmat
 * 2 - (NOT IMPLEMENTED) Perform convolution in basis space,
 *     apply ovlp^-1 to convmat
 * 3 - Perform convolution in reciprocal space, apply ovlp^-0.5 to func_l_*g
 *     and ovlp^-0.5 to func_l_*k
 * 4-7 - Diff Gaussian basis
 * 4 - Perform convolution in basis space, apply ovlp^-1 to func_l_*g
 * 5 - (NOT IMPLEMENTED)
 * 6 - (NOT IMPLEMENTED)
 * 7 - Perform convolution in reciprocal space, apply ovlp^-0.5 to func_l_*g
 *     and ovlp^-0.5 to func_l_*k
 */
typedef struct {
    int mode; // 0 for Gaussians, 1 for Gaussian differences
    int lmax;
    int *nv_l;
    int *jloc_l; // location of first orbital for each l, starting at 0
    int nv; // equivalent to nbas in pyscf
    int nj; // equivalent to nao in pyscf
    double **expnts_l_v;
    double **coefs_l_v;
    double **ovlp_l_vv; // (lmax+1) matrices for basis, see 'mode' for details
    double *r_g;
    double *k_g;
    double **func_l_vg;
    double **func_l_vk;
} aux_basis;

typedef struct {
    int ia; // atom index
    int lmax; // max l for this atom
    int mode; // If 0, ovlp_l_.. contains overlap; if 1, contains L, if 2, contains L^-1
    double **convmat_l_vuba; // (lmax+1) tensors, each is n_beta(l) x n_v(l) x n_u(l) x n_alpha(l)
    double *convmat_kba; // reciprocal-space value of convolutional kernels on k_g,
                         // To use this, k_g must be the same for vbasis and ubasis
    aux_basis vbasis; // conv basis
    aux_basis ubasis; // aux basis
    double **spline_l_vgp; // spline for vbasis realspace
} aux_set;

typedef struct {
    int *ao_loc_v; // location of each shell for v set, in order:
                   // [loc(atom=0,l=0,v=0), loc(atom=0,l=0,v=1), ...,
                   //  loc(atom=0,l=1,v=0), loc(atom=0,l=1,v=1), ...,
                   //  loc(atom=1,l=0,v=0), loc(atom=1,l=0,v=1), ...,]
    int *ao_loc_u; // location of each shell for u set, same order as ao_loc_v
    int *atom_jloc_at; // corresponds to v bas
    int *atom_iloc_at; // corresponds to u bas
    int norb_v;
    int norb_u;
    int n_atl; // number of atom/l pairs
    int *at_atl; // atom corresponding to each atl pair
    int *l_atl; // l corresponding to each atl pair
    int natm;
    aux_set *aux_at;
    char UPLO;
} atom_aux_list;

#endif
