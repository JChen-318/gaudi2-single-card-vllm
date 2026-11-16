// SPDX-License-Identifier: Apache-2.0
// T=16 build of the fused full-query MTP core (DSpine K=15 verification). Same kernel source; the state tile
// stays in registers across all 16 tokens, so the recurrent state is read once (the T8 composition read it twice
// and wrote the 8th checkpoint back before re-reading it). Selected by the shared GUID's glue when T=16.
#define TOKENS 16
#include "gdn_mtp_fused_f32_gaudi2.c"
