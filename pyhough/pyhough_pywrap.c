#include <Python.h>
#include <stdbool.h>
#include "pyhough.h"
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

struct PyHoughObject {
    PyObject_HEAD
    struct hough* hough;
};

static void
PyHoughObject_dealloc(struct PyHoughObject* self) {
    self->hough = hough_free(self->hough);
#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static int
PyHoughObject_init(struct PyHoughObject *self, PyObject *args) {
    // write this...
    PyArrayObject *image_obj = NULL;
    PyArrayObject *theta_obj = NULL;
    PyArrayObject *rho_obj = NULL;

    npy_intp *dims,ntheta,nrho;
    bool *image;
    double *theta;
    double *rho;

    if (!PyArg_ParseTuple(args,
			  (char*)"OOO",
			  &image_obj,
			  &theta_obj,
			  &rho_obj)) {
	printf("Failed to parse init.\n");
	return -1;
    }

    dims = PyArray_DIMS(image_obj);
    ntheta = *PyArray_DIMS(theta_obj);
    nrho = *PyArray_DIMS(rho_obj);

    image = (bool *) PyArray_DATA(image_obj);
    theta = (double *) PyArray_DATA(theta_obj);
    rho = (double *) PyArray_DATA(rho_obj);

    self->hough = hough_new(dims,image,
			    ntheta,theta,
			    nrho,rho);

    if (!self->hough) {
	return -1;
    }

    return 0;
}

static PyObject *
PyHoughObject_repr(struct PyHoughObject *self) {
    char repr[256];
    sprintf(repr, "PyHough Object");
#if PY_MAJOR_VERSION >= 3
    return Py_BuildValue("y", repr);
#else
    return Py_BuildValue("s", repr);
#endif
}

static PyArrayObject *make_transform_image(const struct hough *self) {
    PyArrayObject *transform = NULL;
    int ndims=2;
    npy_intp dims[2];
    dims[0] = self->nrho;
    dims[1] = self->ntheta;
    transform = (PyArrayObject*) PyArray_ZEROS(ndims, dims, NPY_UINT16, 0);
    return transform;
}

static PyArrayObject *PyHoughObject_transform(struct PyHoughObject *self) {

    PyArrayObject *transform = make_transform_image(self->hough);
    unsigned short *data = (unsigned short*) PyArray_DATA(transform);

    _hough_transform(self->hough, data);

    return transform;
}

static PyMethodDef PyHoughObject_methods[] = {
    {"transform", (PyCFunction) PyHoughObject_transform, METH_NOARGS, "transform\n"},
    {NULL} /* Sentinel */
};

static PyTypeObject PyHoughType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_pyhough_pywrap.Hough",             /*tp_name*/
    sizeof(struct PyHoughObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyHoughObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)PyHoughObject_repr,                         /*tp_repr*/
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
    "Hough Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyHoughObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyHoughObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef PyHough_module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pyhough_pywrap",      /* m_name */
    "Hough Transforms",  /* m_doc */
    -1,                        /* m_size */
    PyHough_module_methods,    /* m_methods */
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
PyInit__pyhough_pywrap(void)
#else
init_pyhough_pywrap(void)
#endif
{
    PyObject* m;

    PyHoughType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyHoughType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyHoughType) < 0)
        return;

    m = Py_InitModule3("_pyhough_pywrap", PyHough_module_methods, "Define Hough type and methods.");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyHoughType);
    PyModule_AddObject(m, "Hough", (PyObject *)&PyHoughType);

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

