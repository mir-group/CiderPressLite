#ifndef _NR_CIDER_NUMINT_H
#define _NR_CIDER_NUMINT_H

#define BOXSIZE         56
#define BLKSIZE         104
#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))
#define BAS_SLOTS       8
#define ATOM_OF         0
#define ANG_OF          1
#define PTR_EXP         5
#define PTR_COEFF       6
#define OVLP_OFFSET(a,ish)  ovlp_mats + (a*nbas + ish) * ngamma*nalpha
#define OVLP_OFFSET2(a,ish)  ovlp_mats + (a*natc + atco->atc_loc[ish]) * nalpha
#define OVLP_OFFSET3(a,ish)  ovlp_mats + (nalpha*atco->atc_loc[ish] + a) * nalpha
#define OVLP_OFFSET4(a,ish)  ovlp_mats + a*natc + atco->atc_loc[ish]

typedef struct {
    int ngsh;
    int ngsh2;
    int ia; // atom index
    int ia_loc; // first index in conv_aux_loc for this atom
    int lmax; // maximum l for this atom
    int *l_loc; // start indexes of gammas/gcoefs at each l
    int *l_loc2; // start indexes of gamma ovlp at each l
    double *gammas; // exponents
    double *gcoefs; // coefficients
    double *gtrans; // transformation (inverted overlap) matrix for gammas;
    double *gtrans_m; // transformation (inverted overlap) matrix for gammas;
    double *gtrans_p; // transformation (inverted overlap) matrix for gammas;
    int *gamma_ids;
} atc_conv_set;

typedef struct {
    atc_conv_set* atc_convs;
    int *conv_aux_loc; // loc for conv_aux basis functions
    int *bas; // pseudo-pyscf bas for calling compute_gaussians
    double *env; // pseudo-pyscf env for calling compute_gaussians
    int *atc_loc; // loc for ovlp between bas_aux and conv_aux
    int natc;
    int natm;
    int naux_conv;
    int nshl_conv;
    char UPLO;
    double *gamma_set; // set of gammas
    int gamma_set_size; // size of gamma_set
} atc_ovlp_set;

#endif