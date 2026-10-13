// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "tpc_kernel_lib_interface.h"
extern unsigned char _binary_rejection_race_o_start, _binary_rejection_race_o_end;
extern unsigned char _binary_rejection_pack_o_start, _binary_rejection_pack_o_end;
namespace tpc_lib_api { struct _TensorManipulationSuggestion; }
using namespace tpc_lib_api;
constexpr const char* race_name="mtp_rejection_race_f32_i32_gaudi2_v2";
constexpr const char* pack_name="mtp_rejection_pack_f32_i32_gaudi2_v2";

static bool type_rank(const Tensor& t, TensorDataType type, unsigned rank) {
    return t.geometry.dataType==type && t.geometry.dims==rank;
}
static void mapping(TensorAccessPattern& p, unsigned dim, unsigned axis, int a, int lo, int hi) {
    p.mapping[dim].indexSpaceDim=axis; p.mapping[dim].a=a;
    p.mapping[dim].start_b=lo; p.mapping[dim].end_b=hi;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t* n,GuidInfo* g) {
    if(!n) return GLUE_FAILED;
    auto capacity=*n; *n=d==DEVICE_ID_GAUDI2 ? 2:0;
    if(!g || !capacity || !*n) return GLUE_SUCCESS;
    if(capacity<2) return GLUE_FAILED;
    std::memset(g,0,2*sizeof(*g)); std::strcpy(g[0].name,race_name); std::strcpy(g[1].name,pack_name);
    return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* p,HabanaKernelInstantiation* o) {
    if(!p || !o || !p->inputTensors || !p->outputTensors) return GLUE_FAILED;
    if(p->deviceId!=DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    const bool race=std::strcmp(p->guid.name,race_name)==0;
    const bool pack=std::strcmp(p->guid.name,pack_name)==0;
    if(!race && !pack) return GLUE_NODE_NOT_FOUND;
    if(p->inputTensorNr!=(race?5u:7u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorNr!=(race?2u:1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& prob=p->inputTensors[0];
    if(!type_rank(prob,DATA_F32,3) || !type_rank(p->inputTensors[1],DATA_I32,2) ||
       !type_rank(p->inputTensors[2],DATA_I32,1) || !type_rank(p->inputTensors[3],DATA_F32,2))
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto v=prob.geometry.maxSizes[0], w=prob.geometry.maxSizes[1], b=prob.geometry.maxSizes[2];
    const auto tiles=(v+1023)/1024;
    if(v<64 || v>1048576 || w<1 || w>8 || b<1 || b>16 ||
       p->inputTensors[1].geometry.maxSizes[0]!=w || p->inputTensors[1].geometry.maxSizes[1]!=b ||
       p->inputTensors[2].geometry.maxSizes[0]!=b ||
       p->inputTensors[3].geometry.maxSizes[0]!=w || p->inputTensors[3].geometry.maxSizes[1]!=b)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if(race) {
        if(!type_rank(p->inputTensors[4],DATA_F32,2) ||
           p->inputTensors[4].geometry.maxSizes[0]!=v || p->inputTensors[4].geometry.maxSizes[1]!=b)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    } else {
        if(!type_rank(p->inputTensors[4],DATA_I32,2) ||
           p->inputTensors[4].geometry.maxSizes[0]!=1 || p->inputTensors[4].geometry.maxSizes[1]!=b ||
           !type_rank(p->inputTensors[5],DATA_F32,2) || !type_rank(p->inputTensors[6],DATA_I32,2) ||
           p->inputTensors[5].geometry.maxSizes[0]!=tiles || p->inputTensors[5].geometry.maxSizes[1]!=b ||
           p->inputTensors[6].geometry.maxSizes[0]!=tiles || p->inputTensors[6].geometry.maxSizes[1]!=b)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for(unsigned i=0;i<p->outputTensorNr;++i) {
        const auto& t=p->outputTensors[i];
        if(!type_rank(t,race && i==0?DATA_F32:DATA_I32,2) ||
           t.geometry.maxSizes[0]!=(race?tiles:w+1) || t.geometry.maxSizes[1]!=b)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    o->indexSpaceRank=race?2:1; o->indexSpaceGeometry[0]=race?tiles:b;
    if(race) o->indexSpaceGeometry[1]=b;
    const unsigned batch_axis=race?1:0;
    for(unsigned i=0;i<p->inputTensorNr;++i) std::memset(&o->inputTensorAccessPattern[i],0,sizeof(TensorAccessPattern));
    for(unsigned i=0;i<p->outputTensorNr;++i) std::memset(&o->outputTensorAccessPattern[i],0,sizeof(TensorAccessPattern));
    // Acceptance gathers anywhere in the request row; declare the entire row.
    mapping(o->inputTensorAccessPattern[0],0,batch_axis,0,0,v-1);
    mapping(o->inputTensorAccessPattern[0],1,batch_axis,0,0,w-1);
    mapping(o->inputTensorAccessPattern[0],2,batch_axis,1,0,0);
    mapping(o->inputTensorAccessPattern[1],0,batch_axis,0,0,w-1);
    mapping(o->inputTensorAccessPattern[1],1,batch_axis,1,0,0);
    mapping(o->inputTensorAccessPattern[2],0,batch_axis,0,0,b-1);
    mapping(o->inputTensorAccessPattern[3],0,batch_axis,0,0,w-1);
    mapping(o->inputTensorAccessPattern[3],1,batch_axis,1,0,0);
    for(unsigned i=4;i<p->inputTensorNr;++i) {
        auto columns=p->inputTensors[i].geometry.maxSizes[0];
        mapping(o->inputTensorAccessPattern[i],0,batch_axis,0,0,columns-1);
        mapping(o->inputTensorAccessPattern[i],1,batch_axis,1,0,0);
    }
    for(unsigned i=0;i<p->outputTensorNr;++i) {
        mapping(o->outputTensorAccessPattern[i],0,0,race?1:0,0,race?0:w);
        mapping(o->outputTensorAccessPattern[i],1,batch_axis,1,0,0);
    }
    auto begin=race?&_binary_rejection_race_o_start:&_binary_rejection_pack_o_start;
    auto end=race?&_binary_rejection_race_o_end:&_binary_rejection_pack_o_end;
    unsigned size=end-begin, capacity=o->kernel.elfSize;
    o->kernel.paramsNr=0; o->kernel.elfSize=size;
    if(capacity<size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(o->kernel.kernelElf,begin,size); return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams*,_TensorManipulationSuggestion*) {
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId device,const ShapeInferenceParams* p,ShapeInferenceOutput* o) {
    if(device!=DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if(!p || !o || !p->inputTensors || !o->outputTensors) return GLUE_FAILED;
    const char* guid=p->pGuid?p->pGuid->name:p->guid.name;
    const bool race=std::strcmp(guid,race_name)==0,pack=std::strcmp(guid,pack_name)==0;
    if(!race && !pack) return GLUE_NODE_NOT_FOUND;
    if(p->inputTensorsNr!=(race?5u:7u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if(p->outputTensorsNr!=(race?2u:1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if(!p->inputTensors[0]) return GLUE_FAILED;
    const auto& prob=p->inputTensors[0]->geometry;
    const auto v=prob.maxSizes[0],w=prob.maxSizes[1],b=prob.maxSizes[2];
    for(unsigned i=0;i<p->outputTensorsNr;++i) {
        if(!o->outputTensors[i]) return GLUE_FAILED;
        auto& g=o->outputTensors[i]->geometry; g.dataType=race && i==0?DATA_F32:DATA_I32; g.dims=2;
        g.maxSizes[0]=g.minSizes[0]=race?(v+1023)/1024:w+1;
        g.maxSizes[1]=g.minSizes[1]=b;
    }
    return GLUE_SUCCESS;
}
