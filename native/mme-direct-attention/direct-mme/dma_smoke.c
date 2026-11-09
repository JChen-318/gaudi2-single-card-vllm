#define HLTESTS_LIB_MODE
#include "hlthunk_tests.h"
#include <string.h>
int main(void) {
 setbuf(stdout,NULL);
 if(hltests_init()) return 1;
 int fd=hltests_open("0000:08:00.0");
 printf("OPEN fd=%d\n",fd); if(fd<0)return 2;
 struct hlthunk_hw_ip_info hw={0};
 int rc=hlthunk_get_hw_ip_info(fd,&hw);
 printf("HW rc=%d sram=0x%lx size=%u mme_mask=0x%x legacy=%d\n",rc,hw.sram_base_address,hw.sram_size,hw.mme_enabled_mask,hltests_is_legacy_mode_enabled(fd));
 const unsigned n=4096;
 void *src=hltests_allocate_host_mem(fd,n,NOT_HUGE_MAP);
 void *dst=hltests_allocate_host_mem(fd,n,NOT_HUGE_MAP);
 void *dev=hltests_allocate_device_mem(fd,n,0,NOT_CONTIGUOUS);
 if(!src||!dst||!dev)return 3;
 for(unsigned j=0;j<n/4;j++) ((unsigned*)src)[j]=j^0x12345678;
 memset(dst,0,n);
 rc=hltests_dma_transfer(fd,hltests_get_dma_down_qid(fd,STREAM0),EB_FALSE,MB_TRUE,hltests_get_device_va_for_host_ptr(fd,src),(uintptr_t)dev,n,DMA_DIR_HOST_TO_DRAM);
 printf("DMA_DOWN rc=%d\n",rc);if(rc)return 4;
 rc=hltests_dma_transfer(fd,hltests_get_dma_up_qid(fd,STREAM0),EB_FALSE,MB_TRUE,(uintptr_t)dev,hltests_get_device_va_for_host_ptr(fd,dst),n,DMA_DIR_DRAM_TO_HOST);
 printf("DMA_UP rc=%d equal=%d\n",rc,memcmp(src,dst,n)==0);
 int failed=rc||memcmp(src,dst,n);
 hltests_free_device_mem(fd,dev);hltests_free_host_mem(fd,dst);hltests_free_host_mem(fd,src);hltests_close(fd);hltests_fini();
 return failed?5:0;
}
