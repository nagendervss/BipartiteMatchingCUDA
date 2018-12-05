#include<stdio.h>
#include "bipartiteMatchingKernels.h"

__global__ void initLabelsKernel(int *psi, int *mu, int numRows, int numCols){
  int totalVertices = numRows + numCols;
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < totalVertices){
    psi[threadId] = threadId<numRows?0:1; //Assign label 0 to psi corresponding to rows and label one corresponding to columns
    mu[threadId] = -1; //Initialize all labels to -1 indicating no matching
  }
}

__global__ void gpuPRKernel(int numCols, int numRows, int *mu, bool *actExists, int *CSROffsetCol, int *CSRIndicesCol, int *psi, int loop){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < numCols){
    if(blockIdx.x == 0 && threadIdx.x == 0 && loop%10000 == 0){
      int countNonMatch = 0;
      for(int i=0; i<(numCols); i++){
        if(mu[numRows + i] == -1 || mu[mu[numRows + i]] != (numRows + i)){
	  countNonMatch++;
	}
      }
      printf("Num non match cols in loop %d is %d \n", loop, countNonMatch);
    }
    int muColIndex = numRows + threadId;
    int psiMin = numRows + numCols;
    int rowMatch = -1;
    if(mu[muColIndex] != -2 && (mu[muColIndex] == -1 || mu[mu[muColIndex]] != muColIndex)){
      *actExists = true;
      for(int i=CSROffsetCol[threadId]; i<CSROffsetCol[threadId + 1]; i++){
        if(psi[CSRIndicesCol[i]] < psiMin){
	  psiMin = psi[CSRIndicesCol[i]];
	  rowMatch = CSRIndicesCol[i];
	  if(psiMin == (psi[muColIndex] - 1)){
	    break;
	  }
	}
      }

      if(psiMin < (numRows + numCols)){
        mu[rowMatch] = muColIndex;
	mu[muColIndex] = rowMatch;
	psi[muColIndex] = psiMin + 1;
	psi[rowMatch] = psiMin + 2;
      }
      else{
        mu[muColIndex] = -2;
      }
    }
  }
}

void gpuPR(int *psi, int *mu, int numRows, int numCols, int *CSROffsetCol, int *CSRIndicesCol, int nnz){
  int totalVertices = numRows + numCols;
  int *dpsi, *dmu, *dCSROffsetCol, *dCSRIndicesCol;
  cudaMalloc(&dpsi, totalVertices*sizeof(int));
  cudaMalloc(&dmu, totalVertices*sizeof(int));
  cudaMalloc(&dCSROffsetCol, (numCols + 1)*sizeof(int));
  cudaMalloc(&dCSRIndicesCol, nnz*sizeof(int));

  cudaMemcpy(dCSROffsetCol, CSROffsetCol, (numCols+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dCSRIndicesCol, CSRIndicesCol, nnz*sizeof(int), cudaMemcpyHostToDevice);
  
  int numBlocks = ceil((float)totalVertices/MAXTHREADS);
  initLabelsKernel<<<numBlocks, MAXTHREADS>>>(dpsi, dmu, numRows, numCols);
  
  cudaDeviceSynchronize(); //Placing the function here in case cudaMemcpy is removed later during code dev
  /*cudaMemcpy(psi, dpsi, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mu, dmu, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);*/

  bool *actExists; //Boolean variable indicating existence of active column
  cudaHostAlloc((void **)&actExists, sizeof(bool), 0);
  *actExists = true;
  int i=0;
  while(*actExists){
    i++;
    *actExists = false;
    int numBlocks = ceil((float)numCols/MAXTHREADS);
    gpuPRKernel<<<numBlocks, MAXTHREADS>>>(numCols, numRows, dmu, actExists, dCSROffsetCol, dCSRIndicesCol, dpsi, i);
    cudaDeviceSynchronize();
  }
  cudaMemcpy(psi, dpsi, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mu, dmu, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
}
