/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"

extern ncclNet_t* ncclNet;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];
#ifndef NCCLCHECK
// Propagate errors up
#define NCCLCHECK(call) do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
                /* Print the back trace*/ \
                return res; \
        } \
} while (0)

#endif



#define CUDACHECKGOTO(cmd, res, label) do { cmd; } while(false)

// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetPciPath(int dev, char** path) { NCCLCHECK(ncclNet->pciPath(dev, path)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(void* comm, void* mhandle) { NCCLCHECK(ncclNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetFlush(void* recvComm, void* data, int size, void* mhandle) { NCCLCHECK(ncclNet->flush(recvComm, data, size, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { NCCLCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

#define GPU_BUF_SIZE (2*1024*1024)
static ncclResult_t ncclNetPtrSupport(int dev, int* supportedTypes) {
  int support;
  NCCLCHECK(ncclNet->ptrSupport(dev, &support));
  *supportedTypes = support & ~NCCL_PTR_CUDA;
  // The network supports GPU Direct RDMA ; verify the GPU supports it as well.
  if (support & NCCL_PTR_CUDA) {
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    ncclResult_t res;
    ncclNetListen(dev, &handle, &lComm);
    ncclNetConnect(dev, &handle, &sComm);
    ncclNetAccept(lComm, &rComm);
    ncclNetRegMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle);
    if (res != ncclSuccess) goto cleanup;
    ncclNetDeregMr(sComm, mHandle);
    ncclNetRegMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle);
    ncclNetDeregMr(rComm, mHandle);
    *supportedTypes |= NCCL_PTR_CUDA;
cleanup:
    if (rComm) ncclNetCloseRecv(rComm);
    if (sComm) ncclNetCloseSend(sComm);
    if (lComm) ncclNetCloseListen(lComm);
  }
  return ncclSuccess;
}

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
