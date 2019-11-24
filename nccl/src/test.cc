#include <stdio.h>
#include <stdlib.h>
#include "nccl.h"


int main(int argc, char* argv[])
{
  ncclComm_t comms[4];


  //managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };


  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));

  //initializing NCCL
  ncclCommInitAll(comms, nDev, devs);


   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  ncclGroupStart();

#if 0
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());
#endif

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
