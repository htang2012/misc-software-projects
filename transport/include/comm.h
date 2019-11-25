/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "transport.h"
#incldue "devcomm.h"

#define DEFAULT_BUFFER_SIZE_BYTES (1LL << 22) /* 4MiB */

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      char pad2[CACHE_LINE_SIZE-sizeof(void*)];
      uint64_t opCount;
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      uint64_t opCount;
      char pad2[CACHE_LINE_SIZE-sizeof(uint64_t)];
    };
    char pad4[MEM_ALIGN];
  };
  char buff[1]; // Actually larger than that
};

#define MAXCHANNELS 8

struct ncclComm {
  struct ncclChannel channels[MAXCHANNELS];

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  void* bootstrap;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format

  int node;
  int nNodes;
  int localRanks;

  enum { GROUP, PARALLEL } launchMode;
  bool userStreamSet;
  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;
  uint64_t lastOpCount;

  // Channels for collectives
  int nChannels;

  // Only nvlink is used for inter-GPU communication
  int nvlink;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_PROTOCOLS];

  // An internal CUDA stream for NCCL kernel CGMD launches
  int groupCudaStream;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  void* argsptr;

  // Global proxy thread
  pthread_t proxyThread;
  struct ncclProxyState proxyState;
};

typedef struct ncclComm* ncclComm_t;

#endif
