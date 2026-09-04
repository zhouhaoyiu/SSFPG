#include "mex.h"
#include <stddef.h>

extern size_t ssfpg_gram_core(
    const double *, const double *, const double *, const double *,
    const double *, size_t, size_t, size_t, double, double, double, int,
    const double *, double *, double *, double *, double *, double *, double *);

static void require_real_double(const mxArray *a, const char *name)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a)) {
        mexErrMsgIdAndTxt("SSFPG_gram_rust:type", "%s must be a full real double array.", name);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    size_t m, n, iter, niter;
    double *xall = NULL;
    if (nrhs != 10 || (nlhs != 6 && nlhs != 7)) {
        mexErrMsgIdAndTxt("SSFPG_gram_rust:usage", "Call SSFPG_gram_rust instead of this MEX file directly.");
    }
    for (int i = 0; i < nrhs; ++i) require_real_double(prhs[i], "input");

    m = mxGetM(prhs[0]);
    n = mxGetM(prhs[2]);
    iter = mxGetNumberOfElements(prhs[4]);
    if (m == 0 || mxGetN(prhs[0]) != m || mxGetNumberOfElements(prhs[1]) != m ||
        mxGetN(prhs[2]) != m || mxGetNumberOfElements(prhs[3]) != n ||
        mxGetNumberOfElements(prhs[9]) != m || iter == 0) {
        mexErrMsgIdAndTxt("SSFPG_gram_rust:dimensions", "Incompatible MEX input dimensions.");
    }
    if (mxGetNumberOfElements(prhs[5]) != 1 || mxGetNumberOfElements(prhs[6]) != 1 ||
        mxGetNumberOfElements(prhs[7]) != 1 || mxGetNumberOfElements(prhs[8]) != 1) {
        mexErrMsgIdAndTxt("SSFPG_gram_rust:scalars", "tmax, obE, Xtol0, and isscaling must be scalars.");
    }

    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(iter, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(iter, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(iter, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(iter, 1, mxREAL);
    plhs[5] = mxCreateDoubleScalar(0.0);
    if (nlhs == 7) {
        plhs[6] = mxCreateDoubleMatrix(m, iter, mxREAL);
        xall = mxGetDoubles(plhs[6]);
    }

    niter = ssfpg_gram_core(
        mxGetDoubles(prhs[0]), mxGetDoubles(prhs[1]), mxGetDoubles(prhs[2]),
        mxGetDoubles(prhs[3]), mxGetDoubles(prhs[4]), m, n, iter,
        mxGetScalar(prhs[5]), mxGetScalar(prhs[6]), mxGetScalar(prhs[7]),
        mxGetScalar(prhs[8]) != 0.0, mxGetDoubles(prhs[9]),
        mxGetDoubles(plhs[0]), mxGetDoubles(plhs[1]), mxGetDoubles(plhs[2]),
        mxGetDoubles(plhs[3]), mxGetDoubles(plhs[4]), xall);
    *mxGetDoubles(plhs[5]) = (double)niter;
}
