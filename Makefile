PYTHON ?= python3
GIT_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
GPU_IMAGE ?= xronos-spec:gpu
JETSON_IMAGE ?= xronos-spec:jetson
JETSON_BASE_IMAGE ?=
RUNBOOK_CHECK_DIR ?= /tmp/xronos-spec-runbook-check

PROTO_SRC = xronos/proto/spec.proto

all: protobuf

protobuf:
	$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --mypy_out=. --mypy_grpc_out=. $(PROTO_SRC)

infer-check:
	$(PYTHON) -m py_compile xronos/infer/*.py xronos/proto/spec_pb2.py xronos/proto/spec_pb2_grpc.py
	$(PYTHON) -m xronos.infer.k8s_manifest_audit --manifest k8s/spec-decoding.yaml
	mkdir -p "$(RUNBOOK_CHECK_DIR)"
	$(PYTHON) -m xronos.infer.k8s_runbook \
		--manifest k8s/spec-decoding.yaml \
		--out "$(RUNBOOK_CHECK_DIR)/k8s_runbook.json" \
		--markdown-out "$(RUNBOOK_CHECK_DIR)/k8s_runbook.md"
	$(PYTHON) -m xronos.infer.self_test

docker-gpu:
	docker build \
		-f docker/infer.Dockerfile \
		--build-arg VCS_REF="$(GIT_COMMIT)" \
		--build-arg IMAGE_NAME="$(GPU_IMAGE)" \
		-t "$(GPU_IMAGE)" .

docker-jetson:
	@test -n "$(JETSON_BASE_IMAGE)" || (echo "Set JETSON_BASE_IMAGE to the JetPack/L4T PyTorch base image." >&2; exit 1)
	docker build \
		-f docker/infer.Dockerfile \
		--build-arg BASE_IMAGE="$(JETSON_BASE_IMAGE)" \
		--build-arg VCS_REF="$(GIT_COMMIT)" \
		--build-arg IMAGE_NAME="$(JETSON_IMAGE)" \
		-t "$(JETSON_IMAGE)" .

clean:
	rm -rf xronos/proto/spec_pb2*.py*

.PHONY: all protobuf infer-check docker-gpu docker-jetson clean
