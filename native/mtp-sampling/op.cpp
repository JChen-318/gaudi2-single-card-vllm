// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"
namespace {
const char* race_schema="custom_op::mtp_rejection_tiles";
const char* pack_schema="custom_op::mtp_rejection_pack";
bool registration() {
    auto race_meta=[](const at::Stack& x) {
        const auto& p=x.at(0).toTensor();
        habana::PartialOutputMetaData s,i;
        s.dtype=at::kFloat; i.dtype=at::kInt;
        s.shape={p.size(0),(p.size(2)+1023)/1024}; i.shape=s.shape;
        return habana::PartialOutputMetaDataVector{s,i};
    };
    auto pack_meta=[](const at::Stack& x) {
        const auto& p=x.at(0).toTensor();
        habana::PartialOutputMetaData o; o.dtype=at::kInt; o.shape={p.size(0),p.size(1)+1};
        return habana::PartialOutputMetaDataVector{o};
    };
    habana::custom_op::registerUserCustomOp(race_schema,"mtp_rejection_race_f32_i32_gaudi2_v2",race_meta,nullptr);
    habana::custom_op::registerUserCustomOp(pack_schema,"mtp_rejection_pack_f32_i32_gaudi2_v2",pack_meta,nullptr);
    return true;
}
const bool registered=registration();
void validate(const at::Tensor& p,const at::Tensor& d,const at::Tensor& c,const at::Tensor& u) {
    TORCH_CHECK(p.dim()==3 && p.scalar_type()==at::kFloat && p.is_contiguous());
    TORCH_CHECK(p.size(0)>=1 && p.size(0)<=16 && p.size(1)>=1 && p.size(1)<=8 && p.size(2)>=64 && p.size(2)<=1048576);
    TORCH_CHECK(d.sizes()==p.sizes().slice(0,2) && d.scalar_type()==at::kInt && d.is_contiguous());
    TORCH_CHECK(c.dim()==1 && c.size(0)==p.size(0) && c.scalar_type()==at::kInt && c.is_contiguous());
    TORCH_CHECK(u.sizes()==d.sizes() && u.scalar_type()==at::kFloat && u.is_contiguous());
}
std::tuple<at::Tensor,at::Tensor> tiles(const at::Tensor& p,const at::Tensor& d,const at::Tensor& c,
    const at::Tensor& u,const at::Tensor& n) {
    validate(p,d,c,u);
    TORCH_CHECK(n.sizes()==at::IntArrayRef({p.size(0),p.size(2)}) && n.scalar_type()==at::kFloat && n.is_contiguous());
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(race_schema);
    auto o=descriptor.execute({p,d,c,u,n}); return {o.at(0),o.at(1)};
}
at::Tensor pack(const at::Tensor& p,const at::Tensor& d,const at::Tensor& c,const at::Tensor& u,
    const at::Tensor& bonus,const at::Tensor& s,const at::Tensor& i) {
    validate(p,d,c,u);
    TORCH_CHECK(bonus.sizes()==at::IntArrayRef({p.size(0),1}) && bonus.scalar_type()==at::kInt && bonus.is_contiguous());
    TORCH_CHECK(s.sizes()==at::IntArrayRef({p.size(0),(p.size(2)+1023)/1024}) && s.scalar_type()==at::kFloat && s.is_contiguous());
    TORCH_CHECK(i.sizes()==s.sizes() && i.scalar_type()==at::kInt && i.is_contiguous());
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(pack_schema);
    return descriptor.execute({p,d,c,u,bonus,s,i}).at(0);
}
std::tuple<at::Tensor,at::Tensor> tiles_meta(const at::Tensor& p,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&) {
    const auto shape=std::vector<int64_t>{p.size(0),(p.size(2)+1023)/1024};
    return {at::empty(shape,p.options()),at::empty(shape,p.options().dtype(at::kInt))};
}
at::Tensor pack_meta(const at::Tensor& p,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&) {
    return at::empty({p.size(0),p.size(1)+1},p.options().dtype(at::kInt));
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("mtp_rejection_tiles(Tensor p, Tensor d, Tensor c, Tensor u, Tensor n) -> (Tensor, Tensor)");
    m.def("mtp_rejection_pack(Tensor p, Tensor d, Tensor c, Tensor u, Tensor bonus, Tensor scores, Tensor ids) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { m.impl("mtp_rejection_tiles",tiles); m.impl("mtp_rejection_pack",pack); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { m.impl("mtp_rejection_tiles",tiles_meta); m.impl("mtp_rejection_pack",pack_meta); }
