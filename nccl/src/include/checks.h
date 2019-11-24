/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"


enum cudaError {
     cudaSuccess,
     cudaErrorOk,
     cudaErrorTimeout
};

typedef enum cudaError cudaError_t;
      

// Check CUDA calls
#define CUDACHECK(cmd)  do { } while(false)

#define CUDACHECKGOTO(cmd, res, label) do {  } while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

// Propagate errors up
#define NCCLCHECK(call) do { } while (0);

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#endif
