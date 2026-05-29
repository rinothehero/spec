# Xronos SPEC

Standalone extraction of the Xronos distributed speculative decoding inference stack.

This directory intentionally keeps the Python import path as `xronos.infer` so the original implementation can run with minimal changes. It does not include the Xronos training framework (`client`, `server`, `runner`, `profiler`).

## Included

- `xronos/infer/`: drafter, verifier, driver, analysis, reporting, doctor, and Kubernetes helper tools.
- `xronos/proto/spec.proto`: the gRPC contract for drafter/verifier/driver communication.
- `xronos/proto/spec_pb2*.py`: generated Python gRPC/protobuf bindings.
- `k8s/`: the original Kubernetes speculative decoding experiment manifests.
- `docker/infer.Dockerfile`: inference container build file.

## Install

```bash
cd SPEC
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[infer]"
```

For protobuf regeneration and development checks:

```bash
pip install -e ".[infer,dev]"
make protobuf
make infer-check
```

## Run Distributed Inference

Drafter host:

```bash
python -m xronos.infer.drafter_server \
  --model <drafter-model> \
  --host 0.0.0.0 \
  --port 50061
```

Verifier host:

```bash
python -m xronos.infer.verifier_server \
  --model <verifier-model> \
  --host 0.0.0.0 \
  --port 50062
```

Driver host:

```bash
python -m xronos.infer.spec_driver \
  --drafter-addr <drafter-host>:50061 \
  --verifier-addr <verifier-host>:50062 \
  --tokenizer <tokenizer-or-model> \
  --prompt "Hello" \
  --gammas 1,2,4 \
  --runs 1 \
  --max-new-tokens 32
```

This is distributed inference/decoding, not training. It does not update model weights.

## Hardware Notes

- Drafter power measurement uses Jetson INA3221 sysfs rails when available.
- Verifier power measurement uses NVML or `nvidia-smi` when available.
- On unsupported hardware, decoding can still run, but power samples may be empty.

The original detailed inference documentation remains in `xronos/infer/README.md`.
