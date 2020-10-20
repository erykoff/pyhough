#ifndef _PYHOUGHBACK
#define _PYHOUGHBACK

#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

struct back {
    unsigned short *transform;
    long ntheta,nrho;
    double *theta, *rho;
    long nrow_transform,ncol_transform;

    long nrow,ncol;

};

struct back *back_new(long *dims,
		      unsigned short *transform,
		      long ntheta,
		      double *theta,
		      long nrho,
		      double *rho,
		      long ncol,
		      long nrow);

struct back *back_free(struct back *self);
void _backproject(struct back *self, unsigned short *data);

#endif
