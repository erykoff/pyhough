#include <Python.h>
#include <stdbool.h>
#include "pyhoughback.h"
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

struct PyBackObject {
    PyObject_HEAD
    struct back* back;
};

static void
PyBackObject_dealloc(struct PyBackObject* self) {
    self->back = back_free(self->back);
#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static int
PyBackObject_init(struct PyBackObject *self, PyObject *args) {
    // write this...
    PyArrayObject *transform_obj = NULL;
    PyArrayObject *theta_obj = NULL;
    PyArrayObject *rho_obj = NULL;
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
    char repr[256];
    sprintf(repr, "PyHoughBack Object");
#if PY_MAJOR_VERSION >= 3
    return Py_BuildValue("y", repr);
#else
    return Py_BuildValue("s", repr);
#endif
}

static PyArrayObject *make_image(const struct back *self) {
    PyArrayObject *image = NULL;
    int ndims=2;
    npy_intp dims[2];
    dims[0] = self->nrow;
    dims[1] = self->ncol;
    image = (PyArrayObject*) PyArray_ZEROS(ndims, dims, NPY_UINT16, 0);
    return image;
}


static PyArrayObject *PyBackObject_backproject(struct PyBackObject *self) {

    PyArrayObject *image = make_image(self->back);
    unsigned short *data = (unsigned short*) PyArray_DATA(image);

    _backproject(self->back, data);

    return image;
}


static PyMethodDef PyBackObject_methods[] = {
    {"backproject", (PyCFunction) PyBackObject_backproject, METH_NOARGS, "backproject\n"},
    {NULL} /* Sentinel */
};


static PyTypeObject PyBackType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_pyhoughback_pywrap.Back",             /*tp_name*/
    sizeof(struct PyBackObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyBackObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
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
    (initproc)PyBackObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef PyBack_module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pyhoughback_pywrap",      /* m_name */
    "Back Transforms",  /* m_doc */
    -1,                        /* m_size */
    PyBack_module_methods,    /* m_methods */
    NULL,                      /* m_reload */
    NULL,                      /* m_traverse */
    NULL,                      /* m_clear */
    NULL,                      /* m_free */
};
#endif


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__pyhoughback_pywrap(void)
#else
init_pyhoughback_pywrap(void)
#endif
{
    PyObject* m;

    PyBackType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyBackType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    if (PyType_Ready(&PyBackType) < 0)
        return;

    m = Py_InitModule3("_pyhoughback_pywrap", PyBack_module_methods, "Define Back type and methods.");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyBackType);
    PyModule_AddObject(m, "Back", (PyObject *)&PyBackType);

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

