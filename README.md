# Oracle Class Routing for PTM-based CIL

ImageNet-A class-incremental learning에서 기존 feature/codebook routing이 같은 클래스를 여러 LoRA expert로 분산시키는 문제를 검증하기 위한 공개 실험 저장소입니다.

> 이 단계의 목적은 Neural Collapse를 적용하는 것이 아닙니다. 먼저 GT class를 알고 있다고 가정한 Oracle routing으로 **class-consistent expert 배정 자체가 유효한지** 검증합니다.

## 1. 출발점

초기 `SD-LoRA + Static Top-2, K=5` 실험에서 다음 문제가 관찰되었습니다.

- ImageNet-A, 200 classes, T=20 (10–10 split)
- Seed 1993, epoch 20, no replay, no classifier alignment
- Final Top-1 50.03%, Average Accuracy 61.58%, Final Top-5 76.83%
- expert별 테스트 샘플 수: 239 / 340 / 480 / 301 / 159
- expert별 정확도: 54.8 / 46.8 / 50.4 / 56.1 / 37.1%
- 각 expert가 200개 클래스 중 191–196개를 처리

즉 K=5임에도 class specialization이 형성되지 않았고, 동일 클래스의 샘플이 거의 모든 expert에 흩어졌습니다.

## 2. 검증 질문

1. 같은 클래스의 모든 샘플을 항상 같은 expert에 배정하면 성능이 향상되는가?
2. class-consistent 배정이 expert별 샘플 수와 정확도 편차를 줄이는가?
3. 완벽한 class routing에서도 K=5가 K=1보다 낮다면, 다중 expert 분리 가정 자체가 약한가?

## 3. 고정 조건

| 항목 | 값 |
|---|---|
| Dataset | ImageNet-A (`imageneta`) |
| Task split | T=20, `init_cls=10`, `increment=10` |
| Seed | 1993 |
| Memory | 0 |
| Backbone | `pretrained_vit_b16_224_in21k` |
| Backend | SD-LoRA |
| LoRA rank | 8 |
| Batch size | 64 |
| Epoch | task당 20 |
| Optimizer | AdamW |
| Initial LR | 0.001 |
| Weight decay | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Classifier Alignment | off (`ca_epochs=0`) |

Backend, backbone, optimizer와 학습 조건은 고정하고 routing 방식과 top-K만 바꿉니다.

> **재현 주의:** 원본 R-LoRA main에는 ImageNet-A 전용 K=5 JSON이 커밋되어 있지 않습니다. 이 저장소의 config는 Notion에 기록된 ImageNet-A T=20 조건과 기존 `crcl_topk_sdlora_noca_n5.json`의 학습 설정을 결합해 작성했습니다. 따라서 E1이 기존 Final Top-1 50.03%를 재현하는지 확인한 뒤 E2–E4를 해석해야 합니다.

## 4. 실험군

| ID | K | Routing | Top-K | 목적 |
|---|---:|---|---:|---|
| E0 | 1 | no routing | 1 | 단일 expert 기준선 |
| E1 | 5 | learned feature/codebook | 2 | 초기 Notion 결과 재현 |
| E2 | 5 | GT-class Oracle | 2 | 동일 Top-2에서 class fragmentation 제거 |
| E3 | 5 | learned feature/codebook | 1 | Top-1 learned routing 대조군 |
| E4 | 5 | GT-class Oracle | 1 | class별 단일 expert 전문화 |

핵심 비교는 다음과 같습니다.

- **E2 − E1:** K=5, Top-2, SD-LoRA와 update 수를 고정한 class-consistency 효과
- **E4 − E3:** Top-1에서 learned routing과 GT-class routing의 차이
- **E4 − E0:** class를 5개 expert로 분리하는 것 자체의 이득

## 5. Oracle mapping

Top-1은 class `c`를 `c % 5` expert에 배정합니다.

~~~python
expert = target % 5
~~~

ImageNet-A의 200개 class가 expert별 40개씩 배정됩니다.

Top-2는 class별 고정 expert pair를 사용합니다.

~~~python
expert_1 = target % 5
expert_2 = (target + 1) % 5
~~~

각 expert는 정확히 80개의 class membership을 받습니다. 같은 class의 train/test sample은 항상 같은 경로를 사용합니다.

## 6. 코드 구성

~~~text
overlay/
  LAMDA-PILOT/
    models/crcl.py                 # learned/oracle routing switch
    utils/oracle_routing.py        # deterministic GT-class mapping
    exps/*.json                    # E0–E4 configs
scripts/
  install_overlay.sh              # R-LoRA checkout에 파일 설치
  run_oracle_class_routing.sh      # E0–E4 실행
tests/
  test_oracle_routing.py           # 40/80 class balance 검증
~~~

수정된 `crcl.py`는 R-LoRA main commit `347d4b5a6b39599f53437a9e878c9a3fd0ae78ff` 기준입니다. Backbone과 SD-LoRA backend 구현은 변경하지 않습니다.

## 7. 설치

R-LoRA 접근 권한과 정상 실행 환경이 필요합니다.

~~~bash
git clone https://github.com/dongkyuhyeon/R-LoRA.git
git clone https://github.com/dongkyuhyeon/oracle-class-routing-cil.git

cd oracle-class-routing-cil
bash scripts/install_overlay.sh /path/to/R-LoRA
~~~

`install_overlay.sh`는 기존 `LAMDA-PILOT/models/crcl.py`를 최초 1회 `crcl.py.oracle-backup`으로 보존한 뒤 overlay를 설치합니다.

## 8. 테스트

~~~bash
python -m unittest tests/test_oracle_routing.py -v
python -m py_compile overlay/LAMDA-PILOT/models/crcl.py
~~~

검증 기대값:

- Oracle Top-1: expert별 40 classes
- Oracle Top-2: expert별 80 class memberships
- 반복된 동일 label은 항상 동일 경로

## 9. 실행

먼저 E1을 재현해 기존 조건이 맞는지 확인합니다.

~~~bash
bash scripts/run_oracle_class_routing.sh /path/to/R-LoRA 0 e1
~~~

그다음 전체 실험을 실행합니다.

~~~bash
bash scripts/run_oracle_class_routing.sh /path/to/R-LoRA 0 all
~~~

결과는 이 저장소의 `experiments/runs/` 아래에 실험별 폴더로 저장됩니다.
`PYTHON_BIN`과 `EXPERIMENT_RUNS_DIR` 환경변수로 실행 환경과 출력 경로를
바꿀 수 있습니다.

## 실험 기록 자동 저장

각 실행 결과는 다음 9개 항목으로만 저장됩니다.

```text
experiments/runs/YYYYMMDD_HHMMSS_<experiment>_seed1993/
├── 1_config.json
├── 2_command.sh
├── 3_git_commit_sha.txt
├── 4_metrics.json
├── 5_expert_assignments.csv
├── 6_class_coverage_and_accuracy.csv
├── 7_full.log
├── 8_graphs/
└── 9_summary.md
```

`4_metrics.json`, 두 CSV, 그래프와 `9_summary.md`의 수치 부분은 로그에서
자동 생성됩니다. Dataset, checkpoint와 pretrained weight는 저장하지 않습니다.
`9_summary.md`의 결과 해석과 다음 실험 결정만 최종 결과에 맞게 작성합니다.

## 10. 재현 확인 기준

E1에서 아래 값과 크게 다르면 Oracle 결과를 해석하기 전에 원본 config와 데이터셋을 다시 확인합니다.

| 지표 | 기존 기준 |
|---|---:|
| Final Top-1 | 50.03% |
| Average Accuracy | 61.58% |
| Final Top-5 | 76.83% |
| 최대/최소 테스트 배정 비율 | 약 3.02x |
| expert class coverage | 191–196 / 200 |

단일 seed 재현 오차가 Final Top-1 기준 ±0.3%p를 크게 넘으면 실험 조건 불일치 가능성을 우선 점검합니다.

## 11. 기록할 지표

- Final Top-1, Average Accuracy, Final Top-5
- Old-class / New-class accuracy
- task별 accuracy curve
- expert별 train/test sample 수와 최대/최소 비율
- expert별 담당 class 수
- class당 선택된 expert 수
- expert별 accuracy와 표준편차
- class routing entropy 및 dominant-expert 비율

## 12. 결과 해석

| 결과 | 해석 |
|---|---|
| E2 > E1, E4 > E3, E4 > E0 | 정확한 class partition은 유효하며 learned router가 병목 |
| E2 > E1, E4 > E3, E4 ≤ E0 | routing 오류는 있지만 다중 expert 이득은 부족 |
| E2 ≈ E1, E4 ≈ E3 | class fragmentation 제거 효과가 제한적 |
| Oracle < learned | 고정 class 분리가 adaptation 요구와 맞지 않을 가능성 |

Oracle이 K=1을 안정적으로 넘을 때 Neural Collapse 또는 class-aware representation shaping을 learned approximation으로 연결합니다. 그렇지 않으면 NC 구현 전에 expert 구조와 분리 가정을 재검토해야 합니다.

## 공개 범위

이 저장소는 실험 재현과 연구 논의를 위해 Public으로 운영됩니다. 데이터셋 자체와 pretrained model weight는 포함하지 않습니다.
