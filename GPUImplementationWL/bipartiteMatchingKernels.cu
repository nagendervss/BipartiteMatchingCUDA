#include<stdio.h>
#include "bipartiteMatchingKernels.h"
#include<thrust/scan.h>
#include<thrust/device_ptr.h>

#define iterGRk 0.7

__global__ void initLabelsKernel(int *psi, int *mu, int numRows, int numCols){
  int totalVertices = numRows + numCols;
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < totalVertices){
    psi[threadId] = threadId<numRows?0:1; //Assign label 0 to psi corresponding to rows and label one corresponding to columns
    mu[threadId] = -1; //Initialize all labels to -1 indicating no matching
  }
}

__global__ void gpuPRKernel(int numCols, int numRows, int *mu, bool *actExists, int *CSROffsetCol, int *CSRIndicesCol, int *psi, int loop, int *numA, int *Ac, int *Ap, int *iA){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < *numA){
    int colIdx = Ac[threadId];
    if(colIdx != -1){
      int muColIndex = numRows + colIdx;
      int psiMin = numRows + numCols;
      int rowMatch = -1;
      //if(mu[muColIndex] != -2 && (mu[muColIndex] == -1 || mu[mu[muColIndex]] != muColIndex)){
        //*actExists = true;
        for(int i=CSROffsetCol[colIdx]; i<CSROffsetCol[colIdx + 1]; i++){
          if(psi[CSRIndicesCol[i]] < psiMin){
	    psiMin = psi[CSRIndicesCol[i]];
	    rowMatch = CSRIndicesCol[i];
	    if(psiMin == (psi[muColIndex] - 1)){
	      break;
	    }
	  }
        }

        if(psiMin < (numRows + numCols)){
	  int w = mu[rowMatch];
	  if(w == -1 || iA[w - numRows] != loop){
	    //int w = mu[rowMatch];
            mu[rowMatch] = muColIndex;
	    mu[muColIndex] = rowMatch;
	    psi[muColIndex] = psiMin + 1;
	    psi[rowMatch] = psiMin + 2;
	    
            Ap[threadId] = (w == -1)?w:(w - numRows);
            //if(w != -1) Ap[threadId] = (w - numRows);
	  }
	  /*int w = 
	  if(mu[rowMatch] == -1){
            mu[rowMatch] = muColIndex;
	    mu[muColIndex] = rowMatch;
	    psi[muColIndex] = psiMin + 1;
	    psi[rowMatch] = psiMin + 2;
	    
            Ap[threadId] = -1;
            //if(w != -1) Ap[threadId] = (w - numRows);
	  }
	  else if(iA[mu[rowMatch] - numRows] != loop){
	    int w = mu[rowMatch];
            mu[rowMatch] = muColIndex;
	    mu[muColIndex] = rowMatch;
	    psi[muColIndex] = psiMin + 1;
	    psi[rowMatch] = psiMin + 2;
	    
            Ap[threadId] = (w == -1)?w:(w - numRows);
            //if(w != -1) Ap[threadId] = (w - numRows);
	  }*/
        }
        else{
          mu[muColIndex] = -2;
	  Ac[threadId] = -1;
	  Ap[threadId] = -1;
        }
      //}
    }
    else{
      Ap[threadId] = -1;
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

__global__ void initLists(int *dAp, int *dAc, int *diA, int numCols){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < numCols){
    dAp[threadId] = threadId;
    dAc[threadId] = threadId;
    diA[threadId] = 0;
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

__global__ void gpuPRInitKernel(int *numA, int *Ap, int *Ac, int *mu, int numRows, int *iA, bool *actExists, int loop){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < *numA){
    int colIdx = Ap[threadId];
    if(colIdx != -1){
      if(mu[numRows + colIdx] == -1 || mu[mu[numRows + colIdx]] != (numRows + colIdx)){
        Ac[threadId] = Ap[threadId];
      }
      else{
        colIdx = Ac[threadId];
      }

      if(colIdx != -1){
        iA[colIdx] = loop;
        *actExists = true;
      }
    }
  }
}

__global__ void gpuPRShrinkKernel(int *numA, int numElements, int *indArray, int *Ap, int *Ac, int *mu, int numRows){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId == 0){
    *numA = indArray[numElements - 1] + 1;
  }
  if(threadId < numElements){
    int colIdx = Ap[threadId]; 
    if(colIdx != -1){
      if(mu[numRows + colIdx] == -1 || mu[mu[numRows + colIdx]] != (numRows + colIdx)){
        //Ac[threadId] = Ap[threadId];
      }
      else{
        colIdx = Ac[threadId];
      }

      if(colIdx != -1){
        /*iA[colIdx] = loop;
        *actExists = true;
         indArray[threadId] = 1;*/
	 Ac[indArray[threadId]] = colIdx;
      }
    }
  }
}

__global__ void initIndicatorArray(int numElements, int *indArray, int *Ap, int *Ac, int *mu, int numRows, int *iA, bool *actExists, int loop){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < numElements){ 
    indArray[threadId] = 0;
    int colIdx = Ap[threadId];
    if(colIdx != -1){
      if(mu[numRows + colIdx] == -1 || mu[mu[numRows + colIdx]] != (numRows + colIdx)){
        //Ac[threadId] = Ap[threadId];
      }
      else{
        colIdx = Ac[threadId];
      }

      if(colIdx != -1){
        iA[colIdx] = loop;
        *actExists = true;
         indArray[threadId] = 1;
      }
    }
  }
}

void gpuPRShrink(int *dAp, int *dAc, int *dmu, int *diA, int *numA, int numRows, bool *actExists, int loop){

  //printf("Runtime check 1");
  int numElements = *numA;
  int *dActiveElemIndicatorArray;
  
  cudaMalloc(&dActiveElemIndicatorArray, numElements*sizeof(int));
  
  int numBlocks = ceil((float)numElements/MAXTHREADS);
  initIndicatorArray<<<numBlocks, MAXTHREADS>>>(numElements, dActiveElemIndicatorArray, dAp, dAc, dmu, numRows, diA, actExists, loop);
  cudaDeviceSynchronize();

  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(dActiveElemIndicatorArray);
  //thrust::exclusive_scan(dActiveElemIndicatorArray, dActiveElemIndicatorArray+numElements, dActiveElemIndicatorArray);
  thrust::exclusive_scan(dev_ptr, dev_ptr+numElements, dev_ptr);

  gpuPRShrinkKernel<<<numBlocks, MAXTHREADS>>>(numA, numElements, dActiveElemIndicatorArray, dAp, dAc, dmu, numRows);
  cudaDeviceSynchronize();
}

__global__ void swapKernel(int *Ap, int *Ac, int *numA){
  int threadId = blockIdx.x*blockDim.x + threadIdx.x;
  if(threadId < *numA){
    int temp = Ap[threadId];
    Ap[threadId] = Ac[threadId];
    Ac[threadId] = temp;
  }
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

  int *dAp, *dAc, *diA;
  cudaMalloc(&dAp, numCols*sizeof(int));
  cudaMalloc(&dAc, numCols*sizeof(int));
  cudaMalloc(&diA, numCols*sizeof(int));
  numBlocks = ceil((float)numCols/MAXTHREADS);
  initLists<<<numBlocks, MAXTHREADS>>>(dAp, dAc, diA, numCols);
  cudaDeviceSynchronize();

  int *numA;
  cudaHostAlloc(&numA, sizeof(int), 0);
  *numA = numCols;
  
  /*int *temp;
  cudaMalloc(&temp, sizeof(int));*/

  bool shrink = false;
  
  while(*actExists){
    
    if(loop == iterGR){
      int maxLevel;
      gpuGR(totalVertices, numRows, dmu, dpsi, dCSROffset, dCSRIndices, &maxLevel);
      iterGR += ceil(iterGRk*maxLevel);
      shrink = true;
    }
    *actExists = false;
    if(shrink && *numA > 512){
      gpuPRShrink(dAp, dAc, dmu, diA, numA, numRows, actExists, loop);
      shrink = false;
    }
    else{
      numBlocks = ceil((float)(*numA)/MAXTHREADS);
      gpuPRInitKernel<<<numBlocks, MAXTHREADS>>>(numA, dAp, dAc, dmu, numRows, diA, actExists, loop);
      cudaDeviceSynchronize();
    }
    if(*actExists){
      int numBlocks = ceil((float)(*numA)/MAXTHREADS);
      gpuPRKernel<<<numBlocks, MAXTHREADS>>>(numCols, numRows, dmu, actExists, dCSROffsetCol, dCSRIndicesCol, dpsi, loop, numA, dAc, dAp, diA);
      cudaDeviceSynchronize();

      /*temp = dAp;
      dAp = dAc;
      dAc = temp;*/
      swapKernel<<<numBlocks, MAXTHREADS>>>(dAp, dAc, numA);
      cudaDeviceSynchronize();
    }
    loop++;
  }

  numBlocks = ceil((float)totalVertices/MAXTHREADS);
  fixMatchingKernel<<<numBlocks, MAXTHREADS>>>(totalVertices, dmu);
  cudaMemcpy(mu, dmu, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(psi, dpsi, totalVertices*sizeof(int), cudaMemcpyDeviceToHost);
}
