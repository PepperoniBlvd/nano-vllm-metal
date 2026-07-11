#include "kernels.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace nano_vllm_metal {

void load_library(mx::Device d, const char *path) {
#ifdef _METAL_
    auto &md = mx::metal::device(d);
    md.get_library("nano_vllm_metal", path);
#endif
}

}  // namespace nano_vllm_metal
