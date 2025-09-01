/*************************************************************************
 * Copyright (c) 2025, Mock Topology Implementation for NCCL
 *
 * Modified SYSCHECK macros that can be intercepted for mocking
 ************************************************************************/

#ifndef NCCL_MOCK_CHECKS_H_
#define NCCL_MOCK_CHECKS_H_

#define NCCL_MOCK_TOPOLOGY
#ifdef NCCL_MOCK_TOPOLOGY

// Mock-aware SYSCHECK macros that use our mock functions
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

#else

// Include original checks.h when not mocking
#include "../include/checks.h"

#endif // NCCL_MOCK_TOPOLOGY

#endif // NCCL_MOCK_CHECKS_H_
