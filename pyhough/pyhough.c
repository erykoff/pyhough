#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include "pyhough.h"

struct hough *hough_new(long *dims,
			bool *image,
			long ntheta,
			double *theta,
			long nrho,
			double drho,
			double *rho) {
    struct hough *self;

    if ((self = (struct hough *) calloc(1,sizeof(struct hough))) == NULL) {
	fprintf(stderr,"Failed to allocate struct hough\n");
	exit(1);
    }

    self->image = NULL;
    self->theta = NULL;
    self->rho = NULL;

    self->nrow = dims[0];
    self->ncol = dims[1];

    if ((self->image = (bool *) calloc(self->nrow*self->ncol,sizeof(bool))) == NULL) {
	fprintf(stderr,"Failed to allocate image memory\n");
	exit(1);
    }
    memcpy(self->image,image,self->nrow*self->ncol*sizeof(bool));

    self->ntheta = ntheta;
    if ((self->theta = (double *) calloc(self->ntheta,sizeof(double))) == NULL) {
	fprintf(stderr,"Failed to allocate theta memory\n");
	exit(1);
    }
    memcpy(self->theta,theta,self->ntheta*sizeof(double));

    self->nrho = nrho;
    if ((self->rho = (double *) calloc(self->nrho,sizeof(double))) == NULL) {
	fprintf(stderr,"Failed to allocate rho memory\n");
	exit(1);
    }
    memcpy(self->rho,rho,self->nrho*sizeof(double));

    //self->drho = self->rho[1] - self->rho[0];
    self->drho = drho;

    //printf("rhos: %.2f, %.2f, %.2f\n",self->rho[0],self->rho[1],self->rho[2]);

    return self;
}

void _hough_transform(struct hough *self, unsigned short *data) {
    //long i,m,n;
    //long i,j,m,n;
    long i,j;
    double *costheta,*sintheta;
    //long pind;
    //long rhoprime;
    long index,max_index;
    long *pix_indices,*pix_m,*pix_n;
    long offset;

    // precalculate cos/sin
    
    if ((costheta = (double *)calloc(self->ntheta,sizeof(double))) == NULL) {
	fprintf(stderr,"Failed to allocate costheta memory\n");
	exit(1);
    }
    if ((sintheta = (double *)calloc(self->ntheta,sizeof(double))) == NULL ){
	fprintf(stderr,"Failed to allocate sintheta memory\n");
	exit(1);
    }

    for (i=0;i<self->ntheta;i++) {
	costheta[i] = cos(self->theta[i]);
	sintheta[i] = sin(self->theta[i]);
    }

    // find the pixels
    if ((pix_indices = (long *)calloc(self->nrow*self->ncol,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }
    if ((pix_m = (long *)calloc(self->nrow*self->ncol,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }
    if ((pix_n = (long *)calloc(self->nrow*self->ncol,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }

    index = 0;
    for (i=0;i<self->nrow*self->ncol;i++) {
	if (self->image[i]) {
	    pix_indices[index] = i;
	    pix_m[index] = i % self->ncol;
	    pix_n[index++] = i / self->ncol;
	}
    }
    max_index = index;

    offset = self->nrho / 2;
    printf("offset = %ld\n",offset);

    // loop over good pixels
    for (i=0;i<self->ntheta;i++) {
	//printf(".");
	//fflush(stdout);
	for (j=0;j<max_index;j++) {

	    index = (long) round(pix_m[j]*costheta[i] + pix_n[j]*sintheta[i]) + offset;
	    if ((index >= 0) && (index < self->nrho)) {
		data[index*self->ntheta + i]++;
	    }
	}
    }
    // free sin/cos
    free(costheta);
    free(sintheta);
    free(pix_indices);
    free(pix_m);
    free(pix_n);

}


struct hough *hough_free(struct hough *self) {
    if (self) {
	if (self->image) {
	    free(self->image);
	    self->image = NULL;
	}
	if (self->theta) {
	    free(self->theta);
	    self->theta = NULL;
	}
	if (self->rho) {
	    free(self->rho);
	    self->rho = NULL;
	}

	free(self);
	self=NULL;
    }
    return self;
}
