/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
      void* p = malloc(nelem*sizeof(T));
        if (p == NULL) {
                printf("Failed to malloc %ld bytes", nelem*sizeof(T));
                    return ncclSystemError;
                      }
          memset(p, 0, nelem*sizeof(T));
            *ptr = (T*)p;
              return ncclSuccess;
}


ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, void** commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif
