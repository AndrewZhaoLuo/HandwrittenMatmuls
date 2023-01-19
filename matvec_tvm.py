import os
from typing import *

import numpy as np
import tvm
from tvm import meta_schedule, nd, relay
from tvm.relay import vm
from tvm.runtime import profiler_vm
from tvm.runtime import vm as vm_rt
from tvm.utils import roofline


def toy_model(M: int, K: int, transposed) -> tvm.IRModule:
    if transposed:
        data = relay.var("data", shape=[K, M])
        weight = relay.var("weight", shape=[K, 1])
        result = relay.nn.matmul(data, weight, transpose_a=True)
    else:
        data = relay.var("data", shape=[M, K])
        weight = relay.var("weight", shape=[K, 1])
        result = relay.nn.matmul(data, weight)
    mod = tvm.IRModule.from_expr(result)
    mod = relay.transform.InferType()(mod)
    return mod


def tune(mod, params, target: tvm.target.Target, max_trials_global=128, work_dir=None):
    if work_dir is None:
        raise ValueError("ERROR")

    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)

    print(f"Tuning mod:\n{mod}")

    with meta_schedule.Profiler() as profiler:
        with target:
            database = meta_schedule.relay_integration.tune_relay(
                mod,
                params,
                target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
            )

    print("Tuning Time:")
    print(profiler.table())
    return database


def build_and_get_tir(mod, params, database, target: tvm.target.Target):
    disabled_passes_tir_read = ["tir.CommonSubexprElimTir", "tir.UnrollLoop"]

    saved_tir = roofline.SaveLoweredTIR()
    # Build once to get TIR (with simplifications if needed)
    with target, database, tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
            "relay.FuseOps.max_depth": 30,
        },
        instruments=[saved_tir],
        disabled_pass=disabled_passes_tir_read,
        opt_level=3,
    ):
        vm.compile(
            mod,
            target,
            params=params,
        )
    return saved_tir


def benchmark(
    mod,
    params,
    database,
    target: tvm.target.Target,
    input_dict: Dict[str, np.ndarray],
):
    # TVM takes nd arrays
    input_dict = {k: nd.array(v) for k, v in input_dict.items()}

    # Here we build with all stuff
    with target, database, tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
            "relay.FuseOps.max_depth": 30,
        },
        opt_level=3,
    ):
        exe = vm.compile(
            mod,
            target,
            params=params,
        )
        dev = tvm.device("cuda" if "cuda" in str(target) else "cpu", 0)
        runtime = vm_rt.VirtualMachine(exe, dev)

    results = runtime.benchmark(
        tvm.cpu(),
        func_name="main",
        number=100,
        repeat=1,
        end_to_end=True,
        **input_dict,
    )  # End to end for being fair vs. onnxrt
    # only vm profiler right now
    vm_profiler = profiler_vm.VirtualMachineProfiler(exe, dev)
    results_profiler = vm_profiler.profile(**input_dict)

    return results, results_profiler


if __name__ == "__main__":
    M = 1024 * 9
    K = 1024 * 9
    trials = 256
    toy_mod = toy_model(M, K, transposed=False)
    # toy_mod_T = toy_model(1024 * 9, 1024 * 9, transposed=True)

    target = tvm.target.Target("llvm -num-cores=1")
    tuning_database = tune(
        toy_mod, {}, target, max_trials_global=trials, work_dir="tmp_out"
    )
    saved_tir = build_and_get_tir(toy_mod, {}, tuning_database, target)
    for global_var, tir in saved_tir.functions.items():
        print(global_var, "*" * 50)
        print(tir.script())
        print()

    results, results_proflier = benchmark(
        toy_mod,
        {},
        tuning_database,
        target,
        {
            "data": np.random.random([M, K]).astype("float32"),
            "weight": np.random.random([K, 1]).astype("float32"),
        },
    )
    print(results)
    print(results_proflier)

    """
     ID |            Name |      FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done 
    -------------------------------------------------------------------------------------------------------------------
    0 | fused_nn_matmul | 169869312 |      1 |         6.9099 |   24583.5954 |            24583.5954 |    256 |    Y 
    -------------------------------------------------------------------------------------------------------------------
    Total trials: 256
    Total latency (us): 24583.6
    """