cc_library(
    name = "main",
    srcs = ["lib/intel64/libiomp5.so"],
    hdrs = glob(["include/*.h"]),
    visibility = ["//visibility:public"],
    linkopts = ["-pthread -lpthread"],
)
