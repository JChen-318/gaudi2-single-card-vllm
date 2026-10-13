// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "tpc_kernel_lib_interface.h"
using namespace tpc_lib_api;
extern "C" GlueCodeReturn GetKernelGuids(DeviceId,uint32_t*,GuidInfo*);
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
void shape(Tensor& t,TensorDataType dtype,std::initializer_list<uint64_t> dims) {
    t.geometry.dataType=dtype; t.geometry.dims=dims.size(); unsigned i=0;
    for(auto n:dims) t.geometry.maxSizes[i++]=n;
}
int main() {
    uint32_t count=0; assert(GetKernelGuids(DEVICE_ID_GAUDI2,&count,nullptr)==GLUE_SUCCESS && count==2);
    GuidInfo guids[2]{}; assert(GetKernelGuids(DEVICE_ID_GAUDI2,&count,guids)==GLUE_SUCCESS);
    int cases=0;
    for(bool race:{true,false}) for(uint64_t b:{1,3,16}) for(uint64_t w:{1,4,7}) for(uint64_t v:{64,2113,248320}) {
        const uint64_t tiles=(v+1023)/1024;
        auto p=std::make_unique<HabanaKernelParams>(); auto o=std::make_unique<HabanaKernelInstantiation>();
        Tensor inputs[7]{},outputs[2]{}; TensorAccessPattern ip[7]{},op[2]{};
        p->deviceId=DEVICE_ID_GAUDI2; std::strcpy(p->guid.name,guids[race?0:1].name);
        p->inputTensors=inputs; p->outputTensors=outputs; p->inputTensorNr=race?5:7; p->outputTensorNr=race?2:1;
        o->inputTensorAccessPattern=ip; o->outputTensorAccessPattern=op;
        shape(inputs[0],DATA_F32,{v,w,b}); shape(inputs[1],DATA_I32,{w,b}); shape(inputs[2],DATA_I32,{b}); shape(inputs[3],DATA_F32,{w,b});
        if(race) { shape(inputs[4],DATA_F32,{v,b}); shape(outputs[0],DATA_F32,{tiles,b}); shape(outputs[1],DATA_I32,{tiles,b}); }
        else { shape(inputs[4],DATA_I32,{1,b}); shape(inputs[5],DATA_F32,{tiles,b}); shape(inputs[6],DATA_I32,{tiles,b}); shape(outputs[0],DATA_I32,{w+1,b}); }
        assert(InstantiateTpcKernel(p.get(),o.get())==GLUE_INSUFFICIENT_ELF_BUFFER);
        std::vector<unsigned char> elf(o->kernel.elfSize); o->kernel.kernelElf=elf.data();
        assert(InstantiateTpcKernel(p.get(),o.get())==GLUE_SUCCESS);
        assert(o->indexSpaceRank==(race?2u:1u)); assert(o->indexSpaceGeometry[0]==(race?tiles:b));
        // Contract rejects wrong output type and lengths/geometry before any code is loaded.
        auto old_type=outputs[0].geometry.dataType; outputs[0].geometry.dataType=DATA_BF16;
        assert(InstantiateTpcKernel(p.get(),o.get())==GLUE_INCOMPATIBLE_OUTPUT_SIZE); outputs[0].geometry.dataType=old_type;
        inputs[1].geometry.maxSizes[0]=w+1;
        assert(InstantiateTpcKernel(p.get(),o.get())==GLUE_INCOMPATIBLE_INPUT_SIZE); ++cases;
    }
    std::cout<<"GLUE_CONTRACT_CASES "<<cases<<" PASSED\n";
}
