#ifndef MACROS_H
#define MACROS_H
    //#ifdef USE_FLOAT
        #define real float
        #define Exp expf
    //#endif
    //#ifdef USE_DOUBLE
    //    #define real double
    //    #define Exp exp
    //#endif

    // 2D indexing for 1D arrays
    #define Id2(A,i,j) (A)[ (i)*nz + (j) ]
    // 3D indexing with the time axis as well (indexed by k
    #define Id3(A,k,i,j) (A) [ (k)*nz*nx + (i)*nz + (j) ]

    //BOOLEANS
    #define boolean int
    #define true 1
    #define false 0
#endif
