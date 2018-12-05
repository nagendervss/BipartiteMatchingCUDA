#include<stdio.h>
#include "bipartiteMatchingKernels.h"

#define iterGRk 0.7

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

__global__ void initRelabel(int *mu, int *psi, int totalVertices, int numRows){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < totalVertices){
    psi[threadId] = (threadId < numRows && mu[threadId] == -1)?0:(totalVertices);
  }
}

__global__ void gpuGRKernel(int numRows, int totalVertices, int *psi, int *mu, int bfsLevel, int *CSROffset, int *CSRIndices, bool *rowVertexAdded){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < numRows){
    if(psi[threadId] == bfsLevel){
      for(int i=CSROffset[threadId]; i<CSROffset[threadId + 1]; i++){
        if(psi[numRows + CSRIndices[i]] == totalVertices){
	  psi[numRows + CSRIndices[i]] = bfsLevel + 1;
	  if(mu[numRows + CSRIndices[i]] > -1 && mu[mu[numRows + CSRIndices[i]]] == (numRows + CSRIndices[i])){
	    psi[mu[numRows + CSRIndices[i]]] = bfsLevel + 2;
	    *rowVertexAdded = true;
	  }
	}
      }
    }
  }
}

__global__ void fixMatchingKernel(int totalVertices, int *mu){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < totalVertices){
    if(mu[threadId] > -1 && mu[mu[threadId]] != threadId){
      mu[threadId] = -1;
    }
  }
}

void gpuGR(int totalVertices, int numRows, int *dmu, int *dpsi, int *dCSROffset, int *dCSRIndices, int *maxLevel){
  int numBlocks = ceil((float)totalVertices/MAXTHREADS);
  initRelabel<<<numBlocks, MAXTHREADS>>>(dmu, dpsi, totalVertices, numRows);
  cudaDeviceSynchronize();

  bool *rowVertexAdded;
  cudaHostAlloc(&rowVertexAdded, sizeof(bool), 0);
  *rowVertexAdded = true;

  int bfsLevel = 0;

  while(*rowVertexAdded){
    *rowVertexAdded = false;
    int numBlocks = ceil((float)numRows/MAXTHREADS);
    gpuGRKernel<<<numBlocks, MAXTHREADS>>>(numRows, totalVertices, dpsi, dmu, bfsLevel, dCSROffset, dCSRIndices, rowVertexAdded);
    cudaDeviceSynchronize();
    bfsLevel = bfsLevel + 2;
  }
  *maxLevel = bfsLevel;
}

void gpuPR(int *psi, int *mu, int numRows, int numCols, int *CSROffsetCol, int *CSRIndicesCol, int nnz, int *CSROffset, int *CSRIndices){
  int totalVertices = numRows + numCols;
  int *dpsi, *dmu, *dCSROffsetCol, *dCSRIndicesCol, *dCSROffset, *dCSRIndices;
  cudaMalloc(&dpsi, totalVertices*sizeof(int));
  cudaMalloc(&dmu, totalVertices*sizeof(int));
  cudaMalloc(&dCSROffsetCol, (numCols + 1)*sizeof(int));
  cudaMalloc(&dCSRIndicesCol, nnz*sizeof(int));
  cudaMalloc(&dCSROffset, (numRows + 1)*sizeof(int));
  cudaMalloc(&dCSRIndices, nnz*sizeof(int));

  cudaMemcpy(dCSROffsetCol, CSROffsetCol, (numCols+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dCSRIndicesCol, CSRIndicesCol, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dCSROffset, CSROffset, (numRows+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dCSRIndices, CSRIndices, nnz*sizeof(int), cudaMemcpyHostToDevice);
  
  int numBlocks = ceil((float)totalVertices/MAXTHREADS);
  initLabelsKernel<<<numBlocks, MAXTHREADS>>>(dpsi, dmu, numRows, numCols);
  
  cudaDeviceSynchronize(); //Placing the function here in case cudaMemcpy is removed later during code dev
  /*cudaMemcpy(psi, dpsi, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mu, dmu, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);*/

  bool *actExists; //Boolean variable indicating existence of active column
  cudaHostAlloc((void **)&actExists, sizeof(bool), 0);
  *actExists = true;
  int loop=0;
  int iterGR = 0;
  while(*actExists){
    
    if(loop == iterGR){
      int maxLevel;
      gpuGR(totalVertices, numRows, dmu, dpsi, dCSROffset, dCSRIndices, &maxLevel);
      iterGR += ceil(iterGRk*maxLevel);
    }
    *actExists = false;
    int numBlocks = ceil((float)numCols/MAXTHREADS);
    gpuPRKernel<<<numBlocks, MAXTHREADS>>>(numCols, numRows, dmu, actExists, dCSROffsetCol, dCSRIndicesCol, dpsi, loop);
    loop++;
    cudaDeviceSynchronize();
  }

  numBlocks = ceil((float)totalVertices/MAXTHREADS);
  fixMatchingKernel<<<numBlocks, MAXTHREADS>>>(totalVertices, dmu);
  cudaMemcpy(mu, dmu, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(psi, dpsi, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
}
