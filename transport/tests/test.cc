#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "nccl_net.h"
#include "transport.h"
#include "comm.h"
#include "utils.h"
#include <socket.h>

typedef unsigned long uint64_t;

#define STR2(v) #v
#define STR(v) STR2(v)

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  //if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}



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


struct bootstrapNetComm {
  int fd;
};

/* Init functions */
static char bootstrapNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress bootstrapNetIfAddrs[MAX_IFS];
static int bootstrapNetIfs = -1;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetIfs == -1) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetIfs == -1) {
      bootstrapNetIfs = findInterfaces(bootstrapNetIfNames, bootstrapNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (bootstrapNetIfs <= 0) {
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<bootstrapNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, bootstrapNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&bootstrapNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
      }
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}


ncclResult_t initNetPlugin(ncclNet_t** net) {
  void* netPluginLib = dlopen("libtransport.so", RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == NULL) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      printf("NET/Plugin : No plugin found (libnccl-net.so), using internal implementation");
    } else {
      printf("NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }
  ncclNet_t* extNet = (ncclNet_t*) dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
  if (extNet == NULL) {
    printf("NET/Plugin: Failed to find " STR(NCCL_PLUGIN_SYMBOL) " symbol.");
    goto cleanup;
  }
  if (initNet(extNet) == ncclSuccess) {
    *net = extNet;
    return ncclSuccess;
  }
cleanup:
  if (netPluginLib != NULL) dlclose(netPluginLib);
  return ncclSuccess;
}



ncclNet_t* ncclNet = NULL;
int main(int argc, char* argv[])
{
  ncclComm_t comm;
  bootstrapNetInit();
  initNetPlugin(&ncclNet);
    

  printf("Success \n");
  return 0;
}
