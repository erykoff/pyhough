#include <Python.h>
#include <stdbool.h>
#include "pyhoughback.h"
#include <numpy/arrayobject.h>

struct PyBackObject {
    PyObject_HEAD
    struct back* back;
};


static void
PyBackObject_dealloc(struct PyBackObject* self) {
    self->back = back_free(self->back);
    self->ob_type->tp_free((PyObject*)self);
}


static int
PyBackObject_init(struct PyBackObject *self, PyObject *args) {
    // write this...
    PyObject *transform_obj = NULL;
    PyObject *theta_obj = NULL;
    PyObject *rho_obj = NULL;
    long ncol, nrow;

    
    npy_intp *dims,ntheta,nrho;
    unsigned short *transform;
    double *theta;
    double *rho;
    

    if (!PyArg_ParseTuple(args,
			  (char*)"OOOll",
			  &transform_obj,
			  &theta_obj,
			  &rho_obj,
			  &ncol,&nrow)) {
	printf("Failed to parse init.\n");
	return -1;
    }

    dims = PyArray_DIMS(transform_obj);
    ntheta = *PyArray_DIMS(theta_obj);
    nrho = *PyArray_DIMS(rho_obj);

    transform = (unsigned short *) PyArray_DATA(transform_obj);
    theta = (double *) PyArray_DATA(theta_obj);
    rho = (double *) PyArray_DATA(rho_obj);

    self->back = back_new(dims,transform,
			  ntheta,theta,
			  nrho,rho,
			  ncol,nrow);

    if (!self->back) {
	return -1;
    }

    return 0;
}


static PyObject *
PyBackObject_repr(struct PyBackObject *self) {
    return PyString_FromString("");
}


static PyObject *make_image(const struct back *self) {
    PyObject *image = NULL;
    int ndims=2;
    npy_intp dims[2];
    dims[0] = self->nrow;
    dims[1] = self->ncol;
    image = PyArray_ZEROS(ndims, dims, NPY_UINT16, 0);
    return image;
}


static PyObject *PyBackObject_backproject(struct PyBackObject *self) {
    
    PyObject *image = make_image(self->back);
    unsigned short *data = (unsigned short*) PyArray_DATA(image);

    _backproject(self->back, data);

    return image;
}


static PyMethodDef PyBackObject_methods[] = {
    {"backproject", (PyCFunction) PyBackObject_backproject, METH_NOARGS, "backproject\n"},
    {NULL} /* Sentinel */
};


static PyTypeObject PyBackType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_pyhoughback_pywrap.Back",             /*tp_name*/
    sizeof(struct PyBackObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyBackObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyBackObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Back Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyBackObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyBackObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyPSFExObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef PyBack_type_methods[] = {
    {NULL}  /* Sentinel */
};


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_pyhoughback_pywrap(void) 
{
    PyObject* m;

    PyBackType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyBackType) < 0)
        return;

    m = Py_InitModule3("_pyhoughback_pywrap", PyBack_type_methods, "Define Back type and methods.");

    Py_INCREF(&PyBackType);
    PyModule_AddObject(m, "Back", (PyObject *)&PyBackType);

    import_array();
}

