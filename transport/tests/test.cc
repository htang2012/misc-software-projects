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
#include "net.h"
#include <socket.h>


typedef unsigned long uint64_t;

#define STR2(v) #v
#define STR(v) STR2(v)


/* Init functions */
static char bootstrapNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress bootstrapNetIfAddrs[MAX_IFS];
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;
static int bootstrapNetIfs = -1;

struct bootstrapNetComm {
  int fd;
};

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

static ncclResult_t bootstrapNetNewComm(struct bootstrapNetComm** comm) {
        (*comm) = (struct bootstrapNetComm *) calloc(1, sizeof(struct bootstrapNetComm));
        (*comm)->fd = -1;
        return ncclSuccess;
}

static ncclResult_t bootstrapNetAccept(void* listenComm, void** recvComm) {
  struct bootstrapNetComm* lComm = (struct bootstrapNetComm*)listenComm;
  struct bootstrapNetComm* rComm;
  bootstrapNetNewComm(&rComm);
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen);
  *recvComm = rComm;
  return ncclSuccess;
}



static ncclResult_t bootstrapNetGetSocketAddr(int dev, union socketAddress* addr) {
      if (dev >= bootstrapNetIfs) return ncclInternalError;
        memcpy(addr, bootstrapNetIfAddrs+dev, sizeof(*addr));
          return ncclSuccess;
}

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  //if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t bootstrapNetCreateHandle(ncclNetHandle_t* netHandle, const char* str) {
      union socketAddress* connectAddr = (union socketAddress*) netHandle;
      GetSocketAddrFromString(connectAddr, str);
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


ncclResult_t bootstrapNetListen(int dev, ncclNetHandle_t* netHandle, void** listenComm);
void *bootstrapRoot(void* listenComm);

ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;
  void* listenComm;
  bootstrapNetListen(idFromEnv ? dontCareIf : 0, netHandle, &listenComm);
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, listenComm);
  return ncclSuccess;
}


ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(ncclNetHandle_t) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    if (bootstrapNetCreateHandle(netHandle, env) != 0) {
      printf("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}



ncclResult_t bootstrapNetListen(int dev, ncclNetHandle_t* netHandle, void** listenComm) {
  union socketAddress* connectAddr = (union socketAddress*) netHandle;
  static_assert(sizeof(union socketAddress) < NCCL_NET_HANDLE_MAXSIZE, "union socketAddress size is too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    bootstrapNetGetSocketAddr(dev, connectAddr);
  } else if (dev == findSubnetIf) {
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, connectAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      printf("NET/Socket : No usable listening interface found");
      return ncclSystemError;
    }
    // pass the local address back
    memcpy(connectAddr, &localAddr, sizeof(localAddr));
  } // Otherwise, handle stores a local address
  struct bootstrapNetComm* comm;
  bootstrapNetNewComm(&comm);
  createListenSocket(&comm->fd, connectAddr);
  *listenComm = comm;
  return ncclSuccess;
}


struct extInfo info;

void *bootstrapRoot(void* listenComm) {
  ncclNetHandle_t *rankHandles = NULL;
  ncclNetHandle_t *rankHandlesRoot = NULL; // for initial rank <-> root information exchange
  ncclNetHandle_t zero = { 0 }; // for sanity checking
  void* tmpComm;
  ncclResult_t res;
  setFilesLimit();

  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
    bootstrapNetAccept(listenComm, &tmpComm);
    bootstrapNetRecv(tmpComm, &info, sizeof(info));
    bootstrapNetCloseRecv(tmpComm);

    if (c == 0) {
      nranks = info.nranks;
      ncclCalloc(&rankHandles, nranks);
      ncclCalloc(&rankHandlesRoot, nranks);
    }

    if (nranks != info.nranks) {
      printf("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(&zero, &rankHandlesRoot[info.rank], sizeof(ncclNetHandle_t)) != 0) {
      printf("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
   // Save the connection handle for that rank
    memcpy(rankHandlesRoot+info.rank, info.extHandleListenRoot, sizeof(ncclNetHandle_t));
    memcpy(rankHandles+info.rank, info.extHandleListen, sizeof(ncclNetHandle_t));

    ++c;
  } while (c < nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    void *tmpSendComm;
    bootstrapNetConnect(0, rankHandlesRoot+r, &tmpSendComm);
    bootstrapNetSend(tmpSendComm, rankHandles+next, sizeof(ncclNetHandle_t));
    bootstrapNetCloseSend(tmpSendComm));
  }

out:
  bootstrapNetCloseListen(listenComm);
  if (rankHandles) free(rankHandles);
  if (rankHandlesRoot) free(rankHandlesRoot);

  return NULL;
}


ncclNet_t* ncclNet = NULL;
int main(int argc, char* argv[])
{
  ncclComm_t comm;
  bootstrapNetInit();
  extern ncclNet_t NCCL_PLUGIN_SYMBOL;
  initNetPlugin(&ncclNet);
  initNet(&NCCL_PLUGIN_SYMBOL);
  ncclNet_t & net = NCCL_PLUGIN_SYMBOL;
  ncclUniqueId out;
  bootstrapGetUniqueId(&out);


    

  printf("Success \n");
  return 0;
}
