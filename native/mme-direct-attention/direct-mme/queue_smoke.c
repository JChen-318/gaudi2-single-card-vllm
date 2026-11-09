#define HLTESTS_LIB_MODE
#include "hlthunk_tests.h"
#include <string.h>
int main(void) {
 setbuf(stdout,NULL);
 if(hltests_init())return 1;
 int fd=hltests_open("0000:08:00.0"); if(fd<0)return 2;
 struct hlthunk_hw_ip_info hw={0};hlthunk_get_hw_ip_info(fd,&hw);
 unsigned mq=hltests_get_mme_qid(fd,0,STREAM0), dq=hltests_get_dma_up_qid(fd,STREAM0);
 uint16_t sob=hltests_get_first_avail_sob(fd), mon=hltests_get_first_avail_mon(fd);
 void *common=hltests_allocate_host_mem(fd,4096,NOT_HUGE_MAP);
 void *upper=hltests_create_cb(fd,4096,INTERNAL,hw.sram_base_address+0x2000);
 void *done=hltests_create_cb(fd,4096,EXTERNAL,0);
 if(!common||!upper||!done)return 3;
 struct hltests_pkt_info p={0};p.eb=EB_TRUE;p.mb=MB_TRUE;p.write_to_sob.sob_id=sob;p.write_to_sob.value=1;p.write_to_sob.mode=SOB_ADD;
 unsigned nc=hltests_add_write_to_sob_pkt(fd,common,0,&p);
 memset(&p,0,sizeof(p));p.cp_dma.src_addr=hltests_get_device_va_for_host_ptr(fd,common);p.cp_dma.size=nc;
 unsigned nu=hltests_add_cp_dma_pkt(fd,upper,0,&p);
 int rc=hltests_dma_transfer(fd,hltests_get_dma_down_qid(fd,STREAM0),EB_FALSE,MB_TRUE,hltests_get_device_va_for_host_ptr(fd,upper),hw.sram_base_address+0x2000,nu,DMA_DIR_HOST_TO_SRAM);
 printf("UPLOAD rc=%d mq=%u dq=%u sob=%u mon=%u\n",rc,mq,dq,sob,mon);if(rc)return 4;
 hltests_clear_sobs(fd,1);
 struct hltests_monitor_and_fence m={0};m.queue_id=dq;m.sob_id=sob;m.mon_id=mon;m.sob_val=1;m.dec_fence=true;m.mon_payload=1;m.mon_mode=SOB_EQUAL;
 unsigned nd=hltests_add_monitor_and_fence(fd,done,0,&m);
 memset(&p,0,sizeof(p));p.eb=EB_TRUE;p.mb=MB_TRUE;nd=hltests_add_nop_pkt(fd,done,nd,&p);
 struct hltests_cs_chunk chunks[2]={{upper,nu,mq},{done,nd,dq}};
 uint64_t seq=0;rc=hltests_submit_cs(fd,NULL,0,chunks,2,0,&seq);printf("SUBMIT rc=%d seq=%lu\n",rc,seq);if(rc)return 5;
 rc=hltests_wait_for_cs(fd,seq,5000000);printf("WAIT status=%d expected=%d\n",rc,HL_WAIT_CS_STATUS_COMPLETED);
 if(rc!=HL_WAIT_CS_STATUS_COMPLETED)return 6;
 hltests_destroy_cb(fd,done);hltests_destroy_cb(fd,upper);hltests_free_host_mem(fd,common);hltests_close(fd);hltests_fini();puts("RESULT MME_QUEUE_PASS");return 0;
}
