#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "nccl.h"

typedef unsigned long uint64_t;

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  ncclComm_t comm;
  ncclUniqueId id;
  float *sendbuf, *recvbuf;
  int myrank = 0, nranks= 1;
  uint64_t hostHashs[nranks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myrank] = getHostHash(hostname);
  ncclGetUniqueId(&id); 
  ncclCommInitRank(&comm, nranks, id, myrank);

  //initializing NCCL
  //ncclCommInitAll(comm, nDev, devs);


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
  ncclCommDestroy(comm);


  printf("Success \n");
  return 0;
}
