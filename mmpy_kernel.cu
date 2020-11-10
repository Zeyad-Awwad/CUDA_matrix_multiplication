// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

#define TW 32

using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{

    __shared__ double As[TW][TW], Bs[TW][TW];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int I = blockIdx.y * TW + 8*ty;
    int J =  blockIdx.x * TW + 2*tx;
    int border = N - I;
    if (border > 8)
        border = 8;
    int kk;
    
    if((I < N) && (J < N))
    {
        double c00, c01, c10, c11, c20, c30, c21, c31;
        double c40, c50, c41, c51, c60, c70, c61, c71;
        c00 = 0; c01 = 0; c10 = 0; c11 = 0;
        c20 = 0; c30 = 0; c21 = 0; c31 = 0;
        c40 = 0; c50 = 0; c41 = 0; c51 = 0;
        c60 = 0; c70 = 0; c61 = 0; c71 = 0;
        
        for (kk=0; kk<N/TW; kk++) // go through each block
        {
            // read each block into shared memory
            
            #pragma unroll
            for (int i=0; i<border; i++)
            {
                As[8*ty+i][2*tx] = A[(I+i)*N + kk*TW + 2*tx];
                As[8*ty+i][2*tx+1] = A[(I+i)*N + kk*TW + 2*tx + 1];
            }
            #pragma unroll
            for (int i=0; i<border; i++)
            { 
                Bs[8*ty+i][2*tx] = B[(kk*TW + 8*ty+i)*N + J];
                Bs[8*ty+i][2*tx+1] = B[(kk*TW + 8*ty+i)*N + J + 1];
            }
            __syncthreads();
            
            // dot product
            #pragma unroll
            for (int k=0; k < TW; k++)
            {
                c00 += As[8*ty][k] * Bs[k][2*tx];
                c01 += As[8*ty][k] * Bs[k][2*tx+1];
                c10 += As[8*ty+1][k] * Bs[k][2*tx];
                c11 += As[8*ty+1][k] * Bs[k][2*tx+1];
                c20 += As[8*ty+2][k] * Bs[k][2*tx];
                c21 += As[8*ty+2][k] * Bs[k][2*tx+1];
                c30 += As[8*ty+3][k] * Bs[k][2*tx];
                c31 += As[8*ty+3][k] * Bs[k][2*tx+1];
                c40 += As[8*ty+4][k] * Bs[k][2*tx];
                c41 += As[8*ty+4][k] * Bs[k][2*tx+1];
                c50 += As[8*ty+5][k] * Bs[k][2*tx];
                c51 += As[8*ty+5][k] * Bs[k][2*tx+1];
                c60 += As[8*ty+6][k] * Bs[k][2*tx];
                c61 += As[8*ty+6][k] * Bs[k][2*tx+1];
                c70 += As[8*ty+7][k] * Bs[k][2*tx];
                c71 += As[8*ty+7][k] * Bs[k][2*tx+1];
            }
            __syncthreads();
        }
        ///*
        int diff = N - kk*TW;
        if ( diff > 0 )
        {   
            for (int i=0; i<border; i++)
            {
                As[8*ty+i][2*tx] = 0;
                As[8*ty+i][2*tx+1] = 0;
                Bs[8*ty+i][2*tx] = 0;
                Bs[8*ty+i][2*tx+1] = 0;
            }
            
            if (2*tx < diff)
            {
                for (int i=0; i<border; i++)
                {
                    As[8*ty+i][2*tx] = A[(I+i)*N + kk*TW + 2*tx];
                }
            }
            if (2*tx + 1 < diff)
            {
                for (int i=0; i<border; i++)
                {
                    As[8*ty+i][2*tx+1] = A[(I+i)*N + kk*TW + 2*tx + 1];
                }
            }
            
            for (int i=0; i<border; i++)
            {
                if (8*ty + i < diff + 2 )
                {
                    Bs[8*ty+i][2*tx] = B[(kk*TW + 8*ty+i)*N + J];
                }
            }
            if ( (J+1) < N)
            {
                for (int i=0; i<border; i++)
                {
                    if (8*ty + i < diff + 2)
                    {
                        Bs[8*ty+i][2*tx+1] = B[(kk*TW + 8*ty+i)*N + J+1];
                    }
                }
            }
            
            __syncthreads();

            // dot product
            #pragma unroll
            for (int k=0; k < TW; k++)
            {
                c00 += As[8*ty][k] * Bs[k][2*tx];
                c01 += As[8*ty][k] * Bs[k][2*tx+1];
                c10 += As[8*ty+1][k] * Bs[k][2*tx];
                c11 += As[8*ty+1][k] * Bs[k][2*tx+1];
                c20 += As[8*ty+2][k] * Bs[k][2*tx];
                c21 += As[8*ty+2][k] * Bs[k][2*tx+1];
                c30 += As[8*ty+3][k] * Bs[k][2*tx];
                c31 += As[8*ty+3][k] * Bs[k][2*tx+1];
                c40 += As[8*ty+4][k] * Bs[k][2*tx];
                c41 += As[8*ty+4][k] * Bs[k][2*tx+1];
                c50 += As[8*ty+5][k] * Bs[k][2*tx];
                c51 += As[8*ty+5][k] * Bs[k][2*tx+1];
                c60 += As[8*ty+6][k] * Bs[k][2*tx];
                c61 += As[8*ty+6][k] * Bs[k][2*tx+1];
                c70 += As[8*ty+7][k] * Bs[k][2*tx];
                c71 += As[8*ty+7][k] * Bs[k][2*tx+1];
            }
            
            __syncthreads();
        }
        //*/
        
        C[I*N + J] = c00;
        if ( (I+1) < N )
            C[(I+1)*N + J] = c10;
        if ( (I+2) < N )
            C[(I+2)*N + J] = c20;
        if ( (I+3) < N )
            C[(I+3)*N + J] = c30;
        if ( (I+4) < N )
            C[(I+4)*N + J] = c40;
        if ( (I+5) < N )
            C[(I+5)*N + J] = c50;
        if ( (I+6) < N )
            C[(I+6)*N + J] = c60;
        if ( (I+7) < N )
            C[(I+7)*N + J] = c70;
        if ( (J+1) < N )
        {
            C[I*N + J + 1] = c01;
            if ( (I+1) < N )
                C[(I+1)*N + J + 1] = c11;
            if ( (I+2) < N )
                C[(I+2)*N + J + 1] = c21;
            if ( (I+3) < N )
                C[(I+3)*N + J + 1] = c31;
            if ( (I+4) < N )
                C[(I+4)*N + J + 1] = c41;
            if ( (I+5) < N )
                C[(I+5)*N + J + 1] = c51;
            if ( (I+6) < N )
                C[(I+6)*N + J + 1] = c61;
            if ( (I+7) < N )
                C[(I+7)*N + J + 1] = c71;
        }
    }
}