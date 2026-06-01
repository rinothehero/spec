# Phase 1 Research Plan: Gamma-Dependent Frequency Behavior in Distributed Speculative Decoding

작성일: 2026-06-01

## 0. 한 문장 요약

Modern DSD 알고리즘은 각 decoding round에서 gamma 또는 speculative workload를 동적으로 선택한다. 1차 목표는, 이렇게 gamma가 결정되었을 때 DSD workload마다 에너지 관점의 최적 주파수가 따로 존재하며, 그 값이 단순히 최대 주파수 또는 최저 주파수로 고정되지 않음을 실험적으로 보이는 것이다.

이 문서는 2차 목표인 DSD-aware governor 구현 전까지의 계획이다. 즉, 아직 governor를 만들기 전에 "왜 DSD-aware governor가 필요한가"를 증명하는 실험 계획이다.

## 1. 연구 배경

### 1.1 Distributed Speculative Decoding 구조

Distributed Speculative Decoding, 줄여서 DSD는 작은 drafter와 큰 verifier를 나누어 실행한다.

```text
drafter  = 작은 모델, 여러 token을 먼저 예측
verifier = 큰 모델, drafter가 만든 token을 검증
driver   = 전체 decoding round를 조율
```

우리 실험 환경에서는 다음처럼 역할을 나눈다.

```text
Nano pod = drafter
AGX pod  = verifier
Mas pod  = driver
```

한 round의 흐름은 다음과 같다.

```text
1. driver가 현재 prefix를 기준으로 round 시작
2. drafter가 gamma개 token을 순차적으로 draft
3. verifier가 gamma개 후보를 한 번의 parallel forward로 검증
4. verifier가 맞는 token은 accept
5. 틀린 지점 이후는 verifier token으로 교체
6. 다음 round로 이동
```

여기서 gamma는 한 round에서 drafter가 미리 생성하는 token 수다.

```text
gamma = 1  -> 1개 token draft
gamma = 4  -> 4개 token draft
gamma = 8  -> 8개 token draft
```

### 1.2 Modern DSD에서 gamma는 고정값이 아니다

최근 adaptive DSD 계열 연구들은 gamma 또는 draft workload를 동적으로 고른다. 알고리즘마다 세부 목적은 다르지만 공통적으로 "항상 같은 gamma를 쓰지 않는다"는 특징이 있다.

예시는 다음과 같다.

```text
Dynamic speculation lookahead:
  현재 상황에서 몇 token을 미리 볼지 동적으로 결정

SpecDec++류:
  rejection 위험이 커지기 전에 candidate length를 조절

BanditSpec류:
  online reward를 보고 gamma 또는 speculative config를 선택

EAGLE-2류:
  단순 scalar gamma가 아니라 dynamic draft tree를 구성
```

이 연구에서는 기존 adaptive DSD 알고리즘 자체를 새로 만들거나 모두 구현하지 않는다. 대신 다음 전제를 둔다.

```text
DSD algorithm = gamma/workload를 고르는 black box
우리 연구     = 주어진 gamma/workload에서 주파수-에너지 특성을 분석
```

즉 1차 목표의 질문은 다음과 같다.

```text
기존 adaptive DSD 알고리즘이 어떤 이유로든 gamma를 정했다고 하자.
그 gamma가 정해졌을 때, drafter/verifier 주파수는 항상 최대가 좋은가?
아니면 gamma마다 에너지 최적 주파수가 따로 존재하는가?
```

## 2. 1차 연구 목표

### 2.1 목표 문장

1차 목표는 다음 문장으로 정의한다.

```text
In DSD workloads, once gamma or speculative workload is determined, the energy-optimal frequency is not necessarily the maximum or minimum frequency. Instead, each gamma/workload can have its own best frequency point under a latency constraint.
```

한국어로는 다음과 같다.

```text
DSD 워크로드에서는 gamma 또는 speculative workload가 결정되었을 때,
최대 주파수나 최저 주파수가 항상 최적인 것이 아니다.
latency 조건을 만족하는 범위 안에서,
gamma/workload마다 에너지 최적인 주파수 지점이 따로 존재함을 보인다.
```

### 2.2 이 목표가 중요한 이유

이 목표는 2차 연구 목표인 DSD-aware governor의 필요성을 정당화한다.

만약 실험 결과가 다음처럼 나온다면:

```text
gamma=1  -> 낮은 주파수 또는 중간 주파수가 에너지 최적
gamma=4  -> 다른 주파수가 에너지 최적
gamma=8  -> 또 다른 주파수가 에너지 최적
```

그러면 다음 주장을 할 수 있다.

```text
DSD workload의 gamma-level signal을 모르는 기본 governor 또는 fixed-frequency policy는
항상 좋은 선택을 할 수 없다.
```

반대로 gamma가 무엇이든 항상 같은 주파수가 최적이라면 2차 governor 연구의 필요성은 약해진다. 따라서 1차 목표는 전체 연구의 motivation experiment다.

## 3. 범위 제한

이 문서의 범위는 1차 목표까지다.

### 3.1 포함하는 것

```text
1. fixed gamma 조건에서 frequency-energy-latency landscape 측정
2. gamma별 energy-optimal frequency 비교
3. max frequency, min frequency, mid frequency 비교
4. latency constraint를 둔 energy/token 분석
5. drafter+verifier total measured energy 분석
6. DSD-aware governor가 필요한지 판단하는 motivation evidence 생성
```

### 3.2 포함하지 않는 것

```text
1. full adaptive DSD 알고리즘 구현
2. SpecDec++, EAGLE-2, BanditSpec full 재현
3. online DSD-aware governor 구현
4. CPU/GPU/EMC 전체 joint optimization
5. 모든 모델 조합에 대한 exhaustive search
6. 논문 최종 수준의 대규모 prompt benchmark
```

이 범위를 제한하는 이유는 분명하다. 1차 목표는 "최적 frequency가 workload마다 달라지는지"를 보이는 것이다. 이를 위해 adaptive DSD 알고리즘 전체를 구현할 필요는 없다. gamma를 외부 black-box DSD가 이미 정한 값으로 보고, fixed gamma별로 충분히 촘촘하게 측정하면 된다.

## 4. 주요 개념 정의

### 4.1 Gamma

gamma는 한 DSD round에서 drafter가 미리 생성하는 token 수다.

```text
gamma_t = round t에서 drafter가 생성할 draft token 수
```

1차 실험에서는 gamma를 고정하여 조건을 만든다.

```text
gamma in {1, 2, 4, 8}
```

이 값들은 adaptive DSD 알고리즘이 실제로 선택할 수 있는 대표적인 workload 크기라고 본다.

### 4.2 Frequency

1차 실험의 핵심 제어 대상은 GPU frequency다.

처음에는 verifier GPU frequency를 주된 변수로 둔다.

```text
f_verify_gpu in {low, mid, high}
```

현재 AGX에서 확인된 GPU frequency 후보는 다음과 같다.

```text
low  = 408 MHz
mid  = 816 MHz
high = 1300.5 MHz
```

drafter GPU frequency는 우선 고정한다.

현재 Nano에서 확인된 기본 상태는 다음과 같다.

```text
Nano GPU current high = 624.75 MHz
```

시간이 허용되면 보조 실험으로 drafter GPU frequency도 2수준만 추가한다.

```text
f_draft_gpu in {408 MHz, 624.75 MHz}
```

하지만 1차 목표의 메인 주장은 verifier frequency만으로도 충분히 시작할 수 있다. proposal에서 verifier는 gamma에 따라 memory-bound에서 compute-bound로 성격이 바뀔 수 있다고 되어 있기 때문이다.

### 4.3 Total Measured Energy

이 연구에서 말하는 "전체 시스템 에너지"는 driver와 network를 제외한 measured drafter+verifier energy다.

```text
E_system = E_drafter_device + E_verifier_device
```

현재 측정 범위는 다음과 같다.

```text
drafter energy  = Nano INA3221 rail 기반 측정
verifier energy = AGX Jetson rail 기반 측정
```

Mas driver의 CPU energy, network switch energy, Kubernetes overhead는 포함하지 않는다. 따라서 문서와 결과에서는 반드시 다음 표현을 쓴다.

```text
total measured drafter+verifier energy
```

### 4.4 Energy per Generated Token

주요 energy metric은 energy per generated token이다.

```text
energy/token = E_system / generated_tokens
```

이 값을 쓰는 이유는 prompt마다 생성 token 수가 다를 수 있기 때문이다.

### 4.5 Latency Constraint

에너지만 최소화하면 너무 낮은 주파수에서 아주 느린 설정이 최적처럼 보일 수 있다. 따라서 1차 목표에서도 latency constraint를 둔다.

기본 constraint는 다음 중 하나로 둔다.

```text
Option A:
  latency <= always-high-frequency latency * 1.05

Option B:
  tokens/sec >= always-high-frequency tokens/sec * 0.95
```

초기 분석에서는 둘 다 계산한다. 최종 본문에서는 더 해석이 쉬운 하나를 선택한다.

권장 기본값:

```text
latency overhead <= 5 percent
```

즉 "최고 주파수 대비 5% 이상 느려지지 않는 조건 안에서 energy/token이 최소인 주파수"를 energy-optimal frequency로 정의한다.

## 5. 연구 가설

### 5.1 Hypothesis H1: gamma별 verifier workload 성격이 다르다

```text
gamma가 작을 때:
  verifier forward가 작은 batch/token 수를 처리한다.
  memory-bound 성격이 강할 수 있다.
  GPU frequency를 낮춰도 latency 손해가 작을 수 있다.

gamma가 클 때:
  verifier가 한 번에 더 많은 token을 검증한다.
  compute-bound 성격이 커질 수 있다.
  GPU frequency를 낮추면 latency 손해가 커질 수 있다.
```

따라서 gamma별로 latency-frequency curve가 다를 것으로 예상한다.

### 5.2 Hypothesis H2: 최대 주파수가 항상 에너지 최적이 아니다

최대 주파수는 latency를 줄일 수 있지만 power를 크게 올린다.

```text
E = Power * Time
```

만약 frequency 증가로 latency가 조금만 줄고 power가 크게 증가하면, total energy는 오히려 증가한다.

따라서 다음 현상이 가능하다.

```text
high freq:
  fastest일 수 있음
  하지만 energy/token이 나쁠 수 있음

mid freq:
  latency는 거의 유지
  power가 낮아 energy/token이 좋을 수 있음

low freq:
  power는 낮지만 너무 느려져 energy/token이 나빠질 수 있음
```

1차 목표는 이 non-monotonic behavior를 보이는 것이다.

### 5.3 Hypothesis H3: gamma별 energy-optimal frequency가 다르다

가장 중요한 가설이다.

```text
best_freq(gamma=1) != best_freq(gamma=4)
또는
best_freq(gamma=2) != best_freq(gamma=8)
```

이 현상이 보이면 DSD-specific workload signal을 이용하는 frequency policy의 필요성이 생긴다.

### 5.4 Hypothesis H4: default governor 또는 fixed frequency는 workload-level signal을 반영하지 못한다

1차 목표에서는 governor를 구현하지 않지만, default governor/fixed frequency가 갖는 한계를 motivation으로 언급할 수 있다.

기본 governor는 일반적으로 utilization, idle time, load, thermal state를 본다. 하지만 다음 DSD-specific signal은 모른다.

```text
gamma
accepted tokens
accept rate
verify workload size
draft/verify phase boundary
round-level latency slack
```

따라서 gamma별 optimal frequency가 다르다는 결과가 나오면, 다음 2차 목표가 자연스럽게 이어진다.

```text
DSD-aware governor should use gamma/workload-level signal.
```

## 6. 현재 pilot evidence

이미 작은 pilot 실험은 성공했다.

### 6.1 실험 환경

```text
drafter  = Qwen/Qwen2.5-0.5B on Nano
verifier = Qwen/Qwen2.5-3B on AGX
driver   = Mas
gamma    = 1, 2, 4, 8
prompt   = 1개
runs     = 1
```

AGX verifier GPU frequency는 다음 세 조건으로 직접 고정했다.

```text
408 MHz
816 MHz
1300.5 MHz
```

### 6.2 Pilot 결과

아래 표는 `tot_power` rail 기준의 pilot 결과다.

| verifier freq | gamma | latency ms | tokens/s | energy/token mJ |
|---:|---:|---:|---:|---:|
| 408 MHz | 1 | 8378.43 | 3.3419 | 4703.9 |
| 408 MHz | 2 | 7871.59 | 3.5571 | 4119.6 |
| 408 MHz | 4 | 8947.03 | 3.1295 | 3949.9 |
| 408 MHz | 8 | 10581.03 | 2.6462 | 4155.0 |
| 816 MHz | 1 | 7898.13 | 3.5451 | 4357.9 |
| 816 MHz | 2 | 7313.34 | 3.8286 | 3933.8 |
| 816 MHz | 4 | 8671.15 | 3.2291 | 3988.7 |
| 816 MHz | 8 | 10463.29 | 2.6760 | 3860.6 |
| 1300.5 MHz | 1 | 7718.84 | 3.6275 | 5444.7 |
| 1300.5 MHz | 2 | 7349.23 | 3.8099 | 4241.4 |
| 1300.5 MHz | 4 | 8824.53 | 3.1730 | 4250.6 |
| 1300.5 MHz | 8 | 10731.42 | 2.6092 | 4434.0 |

Pilot 기준으로 보면 다음 패턴이 있었다.

```text
gamma=1:
  fastest = 1300.5 MHz
  best energy/token = 816 MHz

gamma=2:
  fastest = 816 MHz
  best energy/token = 816 MHz

gamma=4:
  fastest = 816 MHz
  best energy/token = 408 MHz

gamma=8:
  fastest = 816 MHz
  best energy/token = 816 MHz
```

이 결과는 아직 prompt 1개, run 1개라 최종 결론은 아니다. 하지만 1차 목표가 성립할 가능성을 보여주는 pilot evidence다.

특히 중요한 점은 다음이다.

```text
최대 주파수 1300.5 MHz가 항상 energy-optimal이 아니었다.
최저 주파수 408 MHz도 항상 energy-optimal이 아니었다.
중간 주파수 816 MHz가 여러 조건에서 좋은 tradeoff를 보였다.
gamma=4에서는 408 MHz가 energy/token 기준으로 가장 좋았다.
```

따라서 정식 실험에서는 prompt 수와 run 수를 늘려 이 패턴이 안정적인지 검증한다.

## 7. 실험 설계

### 7.1 Primary model pair

메인 실험 모델 pair는 하나로 고정한다.

```text
drafter  = Qwen/Qwen2.5-0.5B
verifier = Qwen/Qwen2.5-3B
```

이 조합을 선택하는 이유는 다음과 같다.

```text
1. 이미 Nano와 AGX에서 CUDA fp16 forward가 확인됨
2. 실제 distributed speculative decoding 성공
3. 전력 측정 성공
4. 모델이 너무 크지 않아 반복 실험 가능
5. Qwen2.5 계열이라 tokenizer/model family가 일관됨
```

### 7.2 Optional validation model pair

시간이 남으면 보조 pair 하나를 추가한다.

```text
drafter  = Qwen/Qwen2.5-1.5B
verifier = Qwen/Qwen2.5-7B
```

이 보조 실험의 목적은 일반성 확인이다. 하지만 1차 목표의 필수 조건은 아니다.

### 7.3 Gamma values

메인 gamma set은 다음으로 둔다.

```text
gamma = 1, 2, 4, 8
```

이유:

```text
1. 이미 pilot 실험에서 성공
2. low to high workload 크기를 대표
3. 실험 시간이 과도하게 길지 않음
4. adaptive DSD가 선택할 수 있는 representative 후보로 보기 좋음
```

시간이 남으면 다음 값을 추가한다.

```text
gamma = 16
```

하지만 gamma=16은 drafter overhead가 커질 수 있고, prompt 길이/stop token 영향도 커질 수 있으므로 필수는 아니다.

### 7.4 Verifier GPU frequency values

AGX verifier GPU frequency는 세 수준으로 둔다.

```text
low  = 408 MHz
mid  = 816 MHz
high = 1300.5 MHz
```

이유:

```text
1. 실제 write/restore 확인됨
2. low/mid/high의 차이가 충분히 큼
3. pilot에서 non-monotonic energy behavior가 관찰됨
4. 실험 조건 수가 manageable함
```

### 7.5 Drafter GPU frequency

메인 실험에서는 Nano drafter GPU frequency를 고정한다.

```text
f_draft_gpu = 624.75 MHz 또는 현재 기본 high 상태
```

이렇게 하는 이유:

```text
1. 1차 목표는 gamma별 verifier frequency 최적점 존재 여부 확인
2. 변수 수를 줄여 결과 해석을 명확하게 함
3. proposal에서도 verifier compute regime 변화가 핵심 motivation
```

시간이 남으면 보조 실험으로 drafter frequency 2수준을 추가한다.

```text
f_draft_gpu = 408 MHz
f_draft_gpu = 624.75 MHz
```

하지만 이 경우 실험 조건 수가 두 배가 된다.

```text
4 gamma * 3 verifier freq = 12 conditions
4 gamma * 2 drafter freq * 3 verifier freq = 24 conditions
```

따라서 먼저 12 condition을 안정적으로 끝내는 것이 우선이다.

### 7.6 Prompt set

Pilot은 prompt 1개였기 때문에 정식 실험에서는 prompt set을 늘린다.

권장 prompt 수:

```text
20 prompts
```

prompt type은 다음처럼 섞는다.

```text
1. 짧은 설명형
2. 긴 설명형
3. 상식 QA
4. 코드 생성
5. 수학/논리 추론
6. 요약
7. 반복적이고 예측 쉬운 prompt
8. 창의적이고 예측 어려운 prompt
```

이렇게 나누는 이유는 accept rate가 prompt 성격에 따라 달라질 수 있기 때문이다.

Prompt 예시는 다음과 같다.

```text
Explain distributed speculative decoding in one paragraph.
Write a Python function that checks whether a string is a palindrome.
Summarize why memory bandwidth matters in LLM decoding.
List three reasons why edge devices need energy-efficient inference.
Explain the difference between latency and throughput.
Write a short email asking for a meeting reschedule.
Describe how matrix multiplication works in simple terms.
What are the main causes of climate change?
Explain why GPU frequency can affect energy consumption.
Give a concise definition of dynamic voltage and frequency scaling.
```

최종 prompt set은 JSONL로 저장한다.

```text
SPEC/experiments/prompts/phase1_20_prompts.jsonl
```

각 줄 형식:

```json
{"id": "explain_dsd", "prompt": "Explain distributed speculative decoding in one paragraph."}
```

### 7.7 Runs

각 condition당 반복 횟수:

```text
runs = 3
```

조건 수:

```text
4 gamma * 3 verifier freq * 20 prompts * 3 runs
= 720 measured prompt-runs
```

이것은 작지 않지만 3일 내에 가능할 것으로 본다. 시간이 부족하면 다음 fallback을 쓴다.

```text
Fallback A:
  prompts = 10, runs = 3

Fallback B:
  prompts = 20, runs = 2

Fallback C:
  prompts = 10, runs = 2
```

최소 성공 기준은 다음이다.

```text
4 gamma * 3 verifier freq * 10 prompts * 3 runs
= 360 measured prompt-runs
```

### 7.8 Max new tokens

권장값:

```text
max_new_tokens = 64
```

이유:

```text
1. pilot과 비교 가능
2. 실험 시간이 과도하지 않음
3. prompt 간 stop token 차이를 줄일 수 있음
```

시간이 허용되면 일부 조건에서 128 token도 확인한다.

```text
max_new_tokens = 128
```

하지만 1차 목표에는 64 token이면 충분하다.

## 8. 실험 Matrix

### 8.1 Main matrix

메인 matrix는 다음이다.

| Factor | Values |
|---|---|
| model pair | Qwen2.5-0.5B -> Qwen2.5-3B |
| gamma | 1, 2, 4, 8 |
| verifier GPU freq | 408, 816, 1300.5 MHz |
| drafter GPU freq | fixed high |
| prompts | 20 |
| runs | 3 |
| max new tokens | 64 |

총 condition 수:

```text
4 gamma * 3 verifier freq = 12 conditions
```

총 measured runs:

```text
12 conditions * 20 prompts * 3 runs = 720
```

### 8.2 Reduced matrix

시간이 부족하면 다음 reduced matrix로 간다.

| Factor | Values |
|---|---|
| gamma | 1, 2, 4, 8 |
| verifier GPU freq | 408, 816, 1300.5 MHz |
| prompts | 10 |
| runs | 3 |

총 measured runs:

```text
12 * 10 * 3 = 360
```

### 8.3 Optional drafter frequency matrix

메인 결과가 충분히 빨리 나오면 다음 보조 matrix를 추가한다.

| Factor | Values |
|---|---|
| gamma | 1, 2, 4, 8 |
| drafter GPU freq | 408, 624.75 MHz |
| verifier GPU freq | 408, 816, 1300.5 MHz |
| prompts | 10 |
| runs | 2 |

총 measured runs:

```text
4 * 2 * 3 * 10 * 2 = 480
```

이 실험은 1차 목표를 verifier 중심에서 joint drafter+verifier 방향으로 확장하는 역할을 한다.

## 9. Baselines and Comparisons

1차 목표에서는 governor를 만들지 않지만, 다음 frequency baseline을 비교한다.

### 9.1 Always low

```text
verifier GPU freq = 408 MHz
```

질문:

```text
항상 최저 주파수가 에너지 최적인가?
```

### 9.2 Always mid

```text
verifier GPU freq = 816 MHz
```

질문:

```text
중간 주파수가 대부분의 gamma에서 좋은 tradeoff인가?
```

### 9.3 Always high

```text
verifier GPU freq = 1300.5 MHz
```

질문:

```text
항상 최고 주파수가 latency와 energy 모두에서 좋은가?
```

### 9.4 Gamma-specific oracle

이건 실제 운영 policy가 아니라 분석용 oracle이다.

```text
각 gamma마다 energy/token이 가장 낮은 verifier freq를 선택
```

latency constraint를 적용하면:

```text
각 gamma마다 latency overhead <= 5 percent인 frequency 중 energy/token 최소
```

이 oracle은 2차 목표의 upper bound로 쓸 수 있다.

```text
DSD-aware governor가 이 oracle에 가까워질 수 있으면 좋다.
```

1차 목표에서는 이 oracle을 통해 "gamma별 최적점이 다르다"를 보인다.

## 10. 측정 지표

### 10.1 Primary metrics

가장 중요한 지표:

```text
system_total_energy_mj_per_generated_token
wall_latency_ms
tokens_per_s
```

### 10.2 Secondary metrics

해석을 위해 필요한 지표:

```text
accept_rate
generated_tokens
steps
draft_tokens
accepted_draft_tokens
replacement_tokens
drafter_total_energy_mj
verifier_total_energy_mj
system_total_energy_mj
drafter_power_samples
verifier_power_samples
system_energy_complete
```

### 10.3 Runtime metadata

thermal과 frequency 안정성을 위해 다음도 기록한다.

```text
drafter temperature
verifier temperature
drafter reported GPU frequency
verifier reported GPU frequency
frequency lock success
pod name
model id
tokenizer id
```

중요한 검증 조건:

```text
system_energy_complete == 1
frequency actually locked to requested value
no missing power samples
```

## 11. 최적 주파수 정의

### 11.1 Energy-only best frequency

가장 단순한 정의:

```text
best_freq_energy(gamma) =
  argmin_f mean(energy_per_token | gamma, f)
```

이 정의는 에너지 측면만 본다.

### 11.2 Latency-constrained best frequency

논문에서 더 안전한 정의:

```text
best_freq_constrained(gamma) =
  argmin_f mean(energy_per_token | gamma, f)
  subject to mean(latency | gamma, f)
          <= mean(latency | gamma, high_freq) * 1.05
```

즉 최고 주파수 대비 5% 이상 느려지지 않는 조건에서 에너지 최적인 frequency를 찾는다.

### 11.3 Pareto-optimal frequency

추가 분석으로 latency-energy Pareto frontier를 그린다.

어떤 frequency가 다른 frequency보다 다음 두 조건 모두에서 나쁘면 dominated라고 본다.

```text
latency가 더 느림
energy/token도 더 큼
```

Pareto plot은 다음 질문에 답한다.

```text
어떤 frequency가 gamma별로 의미 있는 tradeoff point인가?
```

## 12. 데이터 처리 계획

### 12.1 Raw CSV

각 실험 결과는 SPEC의 CSV 포맷으로 저장한다.

권장 파일명:

```text
results/phase1_qwen25_0p5b_to_3b_g{gammas}_fv{freqs}_p20_r3.csv
```

주파수별로 나누어 실행하면:

```text
results/phase1_fv408_qwen25_0p5b_to_3b.csv
results/phase1_fv816_qwen25_0p5b_to_3b.csv
results/phase1_fv1300p5_qwen25_0p5b_to_3b.csv
```

### 12.2 Row filtering

분석에는 `rail == "tot_power"` row를 우선 사용한다.

필터 조건:

```text
rail == "tot_power"
system_energy_complete == 1
generated_tokens > 0
frequency_lock_ok == 1 또는 실제 sysfs frequency 확인 완료
```

### 12.3 Aggregation

집계 단위:

```text
group by gamma, verifier_freq
```

계산할 값:

```text
mean energy/token
std energy/token
mean latency
std latency
mean tokens/sec
std tokens/sec
mean accept rate
mean drafter energy
mean verifier energy
```

### 12.4 Outlier handling

다음 경우는 별도 표시한다.

```text
1. system_energy_complete != 1
2. power sample count가 너무 적음
3. generated token 수가 극단적으로 다름
4. temperature가 시작 기준보다 과도하게 높음
5. frequency가 요청값과 다름
```

Outlier를 임의로 제거하지 않는다. 먼저 표시하고 원인을 적는다. 제거가 필요하면 기준을 문서화한다.

## 13. 그림과 표 계획

### 13.1 Figure 1: Gamma별 energy/token vs frequency

x축:

```text
verifier GPU frequency
```

y축:

```text
energy/token
```

line:

```text
gamma=1
gamma=2
gamma=4
gamma=8
```

보이고 싶은 것:

```text
각 gamma의 minimum point가 다를 수 있다.
```

### 13.2 Figure 2: Gamma별 latency vs frequency

x축:

```text
verifier GPU frequency
```

y축:

```text
latency
```

목적:

```text
energy saving이 latency를 얼마나 희생하는지 확인
```

### 13.3 Figure 3: Energy-latency Pareto plot

x축:

```text
latency
```

y축:

```text
energy/token
```

색:

```text
gamma
```

marker:

```text
frequency
```

목적:

```text
최저/최고 주파수가 항상 Pareto-optimal인지 확인
```

### 13.4 Table 1: Gamma별 best frequency

| gamma | fastest freq | energy-only best freq | latency-constrained best freq | energy saving vs high |
|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... |

### 13.5 Table 2: Energy breakdown

| gamma | freq | drafter energy | verifier energy | system energy | verifier energy share |
|---:|---:|---:|---:|---:|---:|

목적:

```text
frequency 변경이 drafter/verifier energy split에 어떤 영향을 주는지 확인
```

## 14. 성공 기준

1차 목표가 성공했다고 판단하려면 다음 중 최소 2개 이상이 보여야 한다.

### 14.1 Criterion A: 최대 주파수가 항상 energy-optimal이 아님

예:

```text
for at least 2 gamma values:
  energy/token(mid or low) < energy/token(high)
```

### 14.2 Criterion B: 최저 주파수도 항상 energy-optimal이 아님

예:

```text
for at least 2 gamma values:
  energy/token(mid or high) < energy/token(low)
```

### 14.3 Criterion C: gamma별 best frequency가 다름

예:

```text
best_freq(gamma=1) != best_freq(gamma=4)
```

또는:

```text
latency-constrained best_freq가 gamma에 따라 달라짐
```

### 14.4 Criterion D: gamma-specific oracle이 fixed frequency보다 좋음

예:

```text
gamma-specific oracle energy/token
  < always-high energy/token
```

latency 조건:

```text
latency overhead <= 5 percent
```

### 14.5 Criterion E: 결과가 prompt 평균에서도 유지됨

pilot처럼 prompt 1개에서만 보이는 현상이 아니라, prompt set 평균에서도 같은 방향이 나와야 한다.

최소 기준:

```text
10 prompts * 3 runs 평균에서 유지
```

권장 기준:

```text
20 prompts * 3 runs 평균에서 유지
```

## 15. 실패 가능성과 해석

### 15.1 모든 gamma에서 816 MHz가 항상 최적이면?

이 경우에도 완전 실패는 아니다.

해석:

```text
max/min frequency는 좋지 않고, mid frequency가 robust optimum일 수 있다.
```

이 경우 1차 claim을 약간 바꾼다.

기존 claim:

```text
gamma마다 최적 frequency가 다르다.
```

수정 claim:

```text
DSD workload에서는 max/min이 아닌 intermediate frequency가 robust energy optimum이다.
```

하지만 2차 governor의 필요성은 약해질 수 있다. 이 경우 governor보다는 "DSD-aware static tuning" 연구가 된다.

### 15.2 항상 최저 주파수가 energy-optimal이면?

이 경우 latency constraint를 확인해야 한다.

가능한 해석:

```text
1. verifier workload가 대부분 memory-bound
2. high frequency가 energy만 낭비
3. latency constraint가 너무 느슨함
```

대응:

```text
1. latency overhead 5% constraint 적용
2. gamma=16 또는 verifier 7B로 workload를 키움
3. max_new_tokens=128로 확인
```

### 15.3 항상 최고 주파수가 energy-optimal이면?

이 경우 DSD-aware DVFS claim이 약해진다.

가능한 원인:

```text
1. AGX high frequency에서 latency 감소가 power 증가보다 큼
2. prompt가 너무 짧음
3. verifier workload가 compute-bound
4. energy 측정 rail이 전체 소비를 충분히 반영하지 못함
```

대응:

```text
1. prompt 수 확대
2. gamma 작은 조건 재확인
3. Qwen2.5-7B verifier로 확장
4. idle baseline 보정 확인
```

### 15.4 결과 variance가 너무 크면?

대응:

```text
1. runs를 5로 증가
2. condition order shuffle
3. thermal start condition 제한
4. idle baseline 측정 추가
5. prompt별 paired comparison 사용
```

## 16. 실험 운영 계획

### 16.1 서버 구성

기본 서버:

```text
Nano drafter:
  model = Qwen/Qwen2.5-0.5B
  port  = 50061

AGX verifier:
  model = Qwen/Qwen2.5-3B
  port  = 50062

Mas driver:
  runs experiment driver
```

### 16.2 Frequency setting 방식

현재 코드의 `--verifier-clocks-mhz`는 `nvidia-smi -lgc` 방식에 가깝다. AGX Jetson에서는 `nvidia-smi` clock control이 적합하지 않다.

따라서 1차 실험에서는 다음 방식 중 하나를 쓴다.

권장:

```text
실험 실행 script에서 AGX /sys/class/devfreq/17000000.gpu/min_freq/max_freq를 직접 고정
```

추후 정리:

```text
verifier_server.py의 SetFrequency가 Jetson devfreq도 지원하도록 코드 수정
```

1차 목표만 보면 직접 sysfs 고정으로 충분하다. 다만 CSV에 실제 frequency 기록이 빠질 수 있으므로, 분석 metadata에 반드시 별도 기록한다.

### 16.3 Thermal control

가능하면 다음 옵션을 사용한다.

```text
--sample-runtime-metadata
```

열 조건이 흔들리면 다음도 고려한다.

```text
--max-start-temp-c
```

하지만 너무 엄격한 temperature wait는 실험 시간을 크게 늘릴 수 있다. 먼저 metadata만 수집하고, 온도 drift가 심하면 제한을 건다.

### 16.4 Condition order

권장:

```text
--shuffle-conditions
--shuffle-runs
```

이유:

```text
항상 408 -> 816 -> 1300 순서로 돌리면 thermal/order bias가 생길 수 있음
```

다만 주파수 변경을 외부 script가 담당하면 완전한 shuffle이 어려울 수 있다. 이 경우 frequency block 순서를 최소한 바꿔서 반복한다.

예:

```text
run block 1: 408 -> 816 -> 1300
run block 2: 816 -> 1300 -> 408
run block 3: 1300 -> 408 -> 816
```

## 17. 일정 계획

### Day 1: 실험 안정화와 prompt set 준비

목표:

```text
정식 phase1 실험을 안정적으로 돌릴 수 있는 상태 확보
```

작업:

```text
1. prompt JSONL 20개 작성
2. AGX frequency set/restore script 정리
3. 현재 Qwen2.5-0.5B/3B 서버 재확인
4. pilot command를 정식 command로 정리
5. 3 frequency * 4 gamma * 5 prompt dry run 또는 small run
6. CSV schema와 frequency metadata 확인
```

산출물:

```text
phase1_20_prompts.jsonl
frequency_control_notes.md
small sanity CSV
```

### Day 2: Main matrix 실행

목표:

```text
메인 12 condition 실험 완료
```

작업:

```text
1. 408 MHz block 실행
2. 816 MHz block 실행
3. 1300.5 MHz block 실행
4. 각 block 완료 후 AGX frequency 상태 확인
5. CSV row 수, energy_complete, sample count 확인
6. 실패 condition 재실행
```

권장 규모:

```text
20 prompts, runs=3
```

시간 부족 시:

```text
10 prompts, runs=3
```

산출물:

```text
phase1 raw CSV files
run log
basic summary table
```

### Day 3: 분석과 1차 claim 정리

목표:

```text
1차 목표가 성립하는지 판정
```

작업:

```text
1. CSV merge
2. tot_power row filtering
3. gamma/frequency별 평균과 표준편차 계산
4. best frequency 계산
5. latency constraint 적용
6. figure 1, 2, 3 생성
7. success criteria 판정
8. 2차 governor 연구로 이어지는 motivation paragraph 작성
```

산출물:

```text
phase1_summary.csv
phase1_best_frequency_by_gamma.csv
phase1_energy_latency_plots/
phase1_findings.md
```

## 18. 최종 산출물 구조

1차 목표 완료 시 다음 파일들이 있으면 좋다.

```text
SPEC/experiments/prompts/phase1_20_prompts.jsonl
SPEC/results/phase1_raw/
SPEC/results/phase1_summary.csv
SPEC/results/phase1_best_frequency_by_gamma.csv
SPEC/results/phase1_plots/energy_per_token_by_gamma.png
SPEC/results/phase1_plots/latency_by_gamma.png
SPEC/results/phase1_plots/pareto_energy_latency.png
SPEC/docs/phase1_findings.md
```

## 19. 1차 목표 결과를 쓰는 방식

결과가 예상대로 나오면 논문/발표 문장은 다음처럼 쓴다.

```text
We first characterize the frequency-energy behavior of distributed speculative decoding under fixed gamma values. Our results show that the energy-optimal verifier frequency is neither always the minimum nor always the maximum frequency. Instead, the best frequency depends on the DSD workload size represented by gamma. This motivates a DSD-aware frequency control policy that uses workload-level signals rather than only low-level utilization.
```

한국어:

```text
우리는 먼저 fixed gamma 조건에서 DSD의 주파수-에너지 특성을 분석했다.
실험 결과, verifier의 에너지 최적 주파수는 항상 최저 또는 최고 주파수가 아니며,
gamma로 표현되는 DSD workload 크기에 따라 달라질 수 있음을 확인했다.
이는 단순 utilization 기반 governor가 아니라,
DSD workload-level signal을 사용하는 주파수 제어 정책이 필요함을 보여준다.
```

## 20. 2차 목표로 연결되는 지점

1차 목표가 보여주는 것은 "관찰"이다.

```text
gamma별로 좋은 frequency가 다르다.
```

2차 목표는 이 관찰을 이용해 "정책"을 만드는 것이다.

```text
DSD-aware governor가 gamma/workload signal을 보고 frequency를 고른다.
```

따라서 1차 목표의 최종 산출물은 2차 목표의 lookup table 또는 training data가 된다.

예:

```text
gamma=1 -> 816 MHz
gamma=2 -> 816 MHz
gamma=4 -> 408 MHz
gamma=8 -> 816 MHz
```

이런 table은 나중에 가장 단순한 gamma-aware governor가 된다.

## 21. 최종 체크리스트

1차 목표 완료 전 확인할 것:

```text
[ ] prompt set 작성 완료
[ ] Nano drafter 서버 정상 실행
[ ] AGX verifier 서버 정상 실행
[ ] Mas driver 정상 실행
[ ] AGX GPU frequency set/restore 정상
[ ] 각 frequency block 후 원복 확인
[ ] 모든 CSV에 system_energy_complete=1 확인
[ ] 각 gamma/frequency condition에 충분한 sample 존재
[ ] prompt 평균 기준 energy/token 계산
[ ] latency constraint 적용한 best frequency 계산
[ ] max/min frequency가 항상 최적이 아님을 확인
[ ] gamma별 best frequency가 달라지는지 확인
[ ] figure/table 생성
[ ] 2차 governor motivation 문단 작성
```

## 22. 결론

1차 연구는 governor를 바로 만드는 것이 아니다. 먼저 DSD workload에서 frequency 최적점이 존재하고, 그 최적점이 gamma/workload에 따라 달라질 수 있음을 보이는 것이다.

이 결과가 확보되면 다음 주장이 가능해진다.

```text
기본 governor는 DSD-specific signal인 gamma를 모른다.
fixed frequency policy도 workload 변화에 대응하지 못한다.
따라서 adaptive DSD 위에는 gamma/workload-aware DVFS governor가 필요하다.
```

즉 1차 목표는 2차 목표의 정당성을 만드는 실험이다.
