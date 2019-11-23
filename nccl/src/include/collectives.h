/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "core.h"
#include "info.h"

#define FUNC_INDEX(coll, redop, dtype, al, pr) ((((((coll)*ncclNumOps + (redop))*ncclNumTypes) + (dtype))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_COLL_NAME(coll, op, dtype) \
  coll##_##op##_##dtype

#define NCCL_KERN_NAME(coll, op, dtype) \
  coll##Kernel_##op##_##dtype

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1

#endif
