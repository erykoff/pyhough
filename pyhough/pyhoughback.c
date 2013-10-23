#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>

#include "pyhoughback.h"


struct back *back_new(long *dims,
		      unsigned short *transform,
		      long ntheta,
		      double *theta,
		      long nrho,
		      double *rho,
		      long ncol,
		      long nrow) {
    struct back *self;

    if ((self = (struct back *) calloc(1,sizeof(struct back))) == NULL) {
	fprintf(stderr,"Failed to allocate struct back\n");
	exit(1);
    }

    self->transform = NULL;
    self->theta = NULL;
    self->rho = NULL;

    self->nrow_transform = dims[0];
    self->ncol_transform = dims[1];
    
    self->nrow = nrow;
    self->ncol = ncol;

    if ((self->transform = (unsigned short *) calloc(self->nrow_transform*self->ncol_transform,sizeof(unsigned short))) == NULL) {
	fprintf(stderr,"Failed to allocate transform memory\n");
	exit(1);
    }
    memcpy(self->transform,transform,self->nrow_transform*self->ncol_transform*sizeof(unsigned short));

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

    return self;
}

struct back *back_free(struct back *self) {
    if (self) {
	if (self->transform) {
	    free(self->transform);
	    self->transform = NULL;
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

void _backproject(struct back *self, unsigned short *data) {
    long i,m,n;
    long index,max_index;
    double sintheta,costheta;
    double sqrt22;

    long *pix_theta,*pix_rho,*pix_index;

    double a,b;

    long data_index;

    // find the pixels
    if ((pix_index = (long *)calloc(self->nrow_transform*self->ncol_transform,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }
    if ((pix_theta = (long *)calloc(self->nrow_transform*self->ncol_transform,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }
    if ((pix_rho = (long *)calloc(self->nrow_transform*self->ncol_transform,sizeof(long))) == NULL) {
	fprintf(stderr,"Failed to allocate index memory\n");
	exit(1);
    }

    index = 0;
    for (i=0;i<self->nrow_transform*self->ncol_transform;i++) {
	if (self->transform[i] > 0) {
	    pix_index[index] = i;
	    pix_theta[index] = i % self->ncol_transform;
	    pix_rho[index++] = i / self->ncol_transform;
	    //printf("%ld %ld: %d -- ",pix_theta[index-1],pix_rho[index-1],self->transform[pix_index[index-1]]);
	    //printf("%.2f, %.2f\n",self->theta[pix_theta[index-1]],self->rho[pix_rho[index-1]]);
	}
    }
    max_index = index;

    sqrt22 = sqrt(2)/2.;
    
    // loop over good pixels
    for (i=0;i<max_index;i++) {
	sintheta = sin(self->theta[pix_theta[i]]);
	costheta = cos(self->theta[pix_theta[i]]);
	//printf("theta = %.2f, sintheta = %.2f, costheta = %.2f, rho = %.2f\n",self->theta[pix_theta[i]],sintheta,costheta,self->rho[pix_rho[i]]);
	if (fabs(sintheta) > sqrt22) {
	    // delta(n, [am+b])
	    a = -costheta/sintheta;
	    b = self->rho[pix_rho[i]]/sintheta;
	    //printf("a = %.2f, b=%.2f\n",a,b);
	    // loop over m values
	    for (m=0;m<self->ncol;m++) {
		n = (long) (a*m + b + 0.5);
		data_index = n*self->ncol + m;
		//data_index = m*self->ncol + n;
		if ((n >= 0) && (n < self->nrow)) {
		    data[data_index] += self->transform[pix_index[i]];
		}// else {
		 //   printf("Warning: %ld, %ld\n",m,n);
		//	}
	    }
	} else {
	    // delta(m, [an+b])
	    a = -sintheta/costheta;
	    b = self->rho[pix_rho[i]]/costheta;
	    // loop over n values
	    for (n=0;n<self->nrow;n++) {
		m = (long) (a*n + b + 0.5);
		data_index = n*self->ncol + m;
		//data_index = m*self->ncol + n;
		if ((m >= 0) && (m < self->ncol)) {
		    data[data_index] += self->transform[pix_index[i]];
		}// else {
		// printf("*Warning: %ld, %ld\n", m,n);
		//}
	    }
	}
	    /*
	    for (j=0;j<self->nrow;j++) {
		index = (long) (a*j + b + 0.5);
		data_index = j*self->ncol + index;
	
		if ((index < self->ncol) && (index >= 0)) {
		    if (data_index >= (self->nrow*self->ncol) || data_index < 0) {
			printf("Shit: %ld, %ld, %ld\n", j, index, data_index);
			exit(1);
		    }
		    //printf("index = %ld\n",index);
		    // j->m, index->n
		    data[j*self->ncol + index] += self->transform[pix_index[i]];
		}
	    }
	    
	} else {
	    // delta(m, [an+b])
	    a = sintheta/costheta;
	    b = self->rho[pix_rho[i]]/costheta;
	    printf("* a = %.2f, b=%.2f\n",a,b);
	    for (j=0;j<self->ncol;j++) {
		index = (long) (a*j + b + 0.5);
		if ((index < self->nrow) && (index >=0)) {
		    data_index = index*self->ncol + j;

		    if (data_index >= (self->nrow*self->ncol) || data_index < 0) {
			printf("* Shit: %ld, %ld, %ld\n", j, index, data_index);
			exit(1);
		    }
		    //if ((a>2.0) && (b>200)) {
		    //	printf("* index = %ld, j= %ld, allind = %ld, pixind = %ld\n",index,j,index*self->ncol+j,pix_index[i]);
		    //}
		    // j->n, index->m
		    data[index*self->ncol + j] += self->transform[pix_index[i]];
		}
	    }
	    
	}
	    */
    }
    

    // free memory
    free(pix_index);
    free(pix_theta);
    free(pix_rho);

}
