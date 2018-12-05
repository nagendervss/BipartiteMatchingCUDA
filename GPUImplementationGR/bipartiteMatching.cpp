#include<iostream>
#include<string>
#include<fstream>
#include<cstdlib>
#include<sstream>
#include<math.h>
#include<time.h>
#include"bipartiteMatchingKernels.h"

#define MAXTHREADS 1024

using namespace std;

typedef struct pair {
  long long f;
  long long s;
} Pair;

int cmpPair(const void *v1, const void *v2){
  long long diff = (((Pair*)v1)->f - ((Pair*)v2)->f);
  if(diff != 0){
    return diff;
  }
  else{
    return (((Pair *)v1)->s - ((Pair *)v2)->s);
  }
}

void readGraphFromMMFile(string fileName, int *nrow, int *ncol, int *nnz, int **CSROffset, int **CSRIndices, int **CSROffsetCol, int **CSRIndicesCol){
  string line;
  ifstream inpFileStream(fileName.c_str(), fstream::in);
  if(!inpFileStream.is_open()){
    cout<<" File "<<fileName<<" cannot be opened."<<endl;
    exit(1);
  }
  do{
    getline(inpFileStream, line);
    if(inpFileStream.eof()){
      cout<<"unexpected EOF"<<endl;
      exit(1);
    }
  }while(line[0] == '%');

  int numRows, numCols, numNonZeroElements;
  stringstream ss (stringstream::in | stringstream::out);
  ss << line;
  ss >> numRows >> numCols >> numNonZeroElements;
  
  Pair *coords = (Pair *)malloc(sizeof(Pair) * numNonZeroElements);
  Pair *coordsCol = (Pair *)malloc(sizeof(Pair) * numNonZeroElements);

  int i = 0;
  while(1){
    getline(inpFileStream, line);
    if(inpFileStream.eof()){
      break;
    }

    int v1,v2;
    stringstream ss;
    ss << line;
    ss >> v1 >> v2;

    coords[i].f = v1 - 1;
    coords[i].s = v2 - 1;

    coordsCol[i].f = v2 - 1;
    coordsCol[i].s = v1 - 1;

    ++i;
  }
  inpFileStream.close();
  qsort(coords, numNonZeroElements, sizeof(Pair), cmpPair);
  qsort(coordsCol, numNonZeroElements, sizeof(Pair), cmpPair);
  //eliminating duplicate edges
  int numOrigNonZeroElements = 1;
  int numOrigNonZeroElementsCol = 1;
  for(int i=1; i<numNonZeroElements; i++){
    if(coords[i].f != coords[numOrigNonZeroElements - 1].f || coords[i].s != coords[numOrigNonZeroElements - 1].s){
      coords[numOrigNonZeroElements].f = coords[i].f;
      coords[numOrigNonZeroElements++].s = coords[i].s;  
    }
    
    if(coordsCol[i].f != coordsCol[numOrigNonZeroElementsCol - 1].f || coordsCol[i].s != coordsCol[numOrigNonZeroElementsCol - 1].s){
      coordsCol[numOrigNonZeroElementsCol].f = coordsCol[i].f;
      coordsCol[numOrigNonZeroElementsCol++].s = coordsCol[i].s;  
    }
  }
  
  if(numOrigNonZeroElements != numOrigNonZeroElementsCol){
    cout<<"Error in computing numOrigNonZeroElements"<<endl;
    exit(1);
  }

  *nnz = numOrigNonZeroElements;
  *nrow = numRows;
  *ncol = numCols;
  
  int *xCSROffset = *CSROffset = (int *)malloc((numRows + 1)*sizeof(int));
  int *xCSRIndices = *CSRIndices = (int *)malloc((numOrigNonZeroElements)*sizeof(int));

  int prevSource = 0;
  xCSROffset[0] = 0;
  int adjInd = 0;
  for(int i=0; i<numOrigNonZeroElements; i++){
    int target = coords[i].s;
    int source = coords[i].f;

    while(prevSource < source){
      xCSROffset[++prevSource] = adjInd;
    }
    xCSRIndices[adjInd++] = target;
  }

  while(prevSource < numRows){
    xCSROffset[++prevSource] = adjInd;
  }

  xCSROffset[numRows] = adjInd;
  
  int *xCSROffsetCol = *CSROffsetCol = (int *)malloc((numCols + 1)*sizeof(int));
  int *xCSRIndicesCol = *CSRIndicesCol = (int *)malloc((numOrigNonZeroElementsCol)*sizeof(int));

  prevSource = 0;
  xCSROffsetCol[0] = 0;
  adjInd = 0;
  for(int i=0; i<numOrigNonZeroElementsCol; i++){
    int target = coordsCol[i].s;
    int source = coordsCol[i].f;

    while(prevSource < source){
      xCSROffsetCol[++prevSource] = adjInd;
    }
    xCSRIndicesCol[adjInd++] = target;
  }

  while(prevSource < numCols){
    xCSROffsetCol[++prevSource] = adjInd;
  }

  xCSROffsetCol[numCols] = adjInd;

  free(coords);
  free(coordsCol);

  return;
 
}

int main(int argc, char** argv){
  string fileName = argv[1];
 
  int nrows, ncols, nnz, *CSROffset, *CSRIndices, *CSROffsetCol, *CSRIndicesCol;
  cout<<"Started reading file "<<fileName<<endl;
  readGraphFromMMFile(fileName, &nrows, &ncols, &nnz, &CSROffset, &CSRIndices, &CSROffsetCol, &CSRIndicesCol);
  cout<<"Completed reading file "<<fileName<<endl;
  

  int totalVertices = nrows + ncols;

  //psi - pointer to array holding the label of each vertex. mu - pointer to array holding the matching index of each vertex
  int *psi = (int *)malloc(totalVertices*sizeof(int));
  int *mu = (int *)malloc(totalVertices*sizeof(int));
  clock_t start = clock();
  gpuPR(psi, mu, nrows, ncols, CSROffsetCol, CSRIndicesCol, nnz, CSROffset, CSRIndices);
  clock_t end = clock();
  double time = (double)(end-start)/CLOCKS_PER_SEC;

  int matchCount = 0;
  for(int i=0; i<nrows; i++){
    if(mu[i] != -1){
      //cout<<"Row "<<i<<" col "<<(mu[i]-nrows)<<endl;
      ++matchCount;
    }
    else{
      //cout<<"Row "<<i<<" col "<<mu[i]<<endl;
    }
  }
  cout<<"Cardinality "<<matchCount<<endl;
  cout<<"Running time for Matching "<<time<<" seconds"<<endl;

  return 1;
}
