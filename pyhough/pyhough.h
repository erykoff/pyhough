#ifndef _PYHOUGH
#define _PYHOUGH

#include <stdbool.h>

struct hough {
    long nrow,ncol;
    bool *image;
    
    long ntheta,nrho;
    double drho;
    double *theta, *rho;
    //unsigned short *transform;
};



struct hough *hough_new(long *dims,
			bool *image,
			long ntheta,
			double *theta,
			long nrho,
			double drho,
			double *rho);

struct hough *hough_free(struct hough *self);
struct hough *hough_transform(const struct hough *self);
void _hough_transform(struct hough *self, unsigned short *data);








#endif