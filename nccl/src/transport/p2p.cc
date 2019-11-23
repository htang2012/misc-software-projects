/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"

struct p2pConnectInfo {
  int direct;
  union {
    void* directPtr;
  };
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
};

#include <sys/types.h>

NCCL_PARAM(P2pLevel, "P2P_LEVEL", -2);
NCCL_PARAM(P2pDisable, "P2P_DISABLE", -2);

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {

  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int sendSize = sizeof(struct ncclSendMem);
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, sendSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/common device", channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    } else {
      // Enable P2P access
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    info.direct = 0;
    // Map IPC and enable P2P access
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/IPC",
        channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int buffSize, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, recvSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      TRACE(NCCL_INIT|NCCL_P2P,"%d <- %d via P2P/common device", myInfo->rank, peerInfo->rank);
    }
      // Enable P2P access
  } 
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclRecvMem*)(info->directPtr);
    send->conn.direct = 1;
  } 
  

  send->conn.buff = remDevMem->buff;
  send->conn.llBuff = remDevMem->llBuff;
  send->conn.ll128Buff = remDevMem->ll128Buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCountRem = &remDevMem->opCount;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.opCountLoc = &resources->devMem->opCount;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclSendMem*)(info->directPtr);
    recv->conn.direct = 1;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  } 

  recv->conn.buff = resources->devMem->buff;
  recv->conn.llBuff = resources->devMem->llBuff;
  recv->conn.ll128Buff = resources->devMem->ll128Buff;
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.opCountLoc = &resources->devMem->opCount;
  recv->conn.head = &remDevMem->head;
  recv->conn.opCountRem = &remDevMem->opCount;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
  CUDACHECK(cudaFree(sendRes->devMem));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
  CUDACHECK(cudaFree(recvRes->devMem));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
