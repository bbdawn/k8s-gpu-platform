# k8s-gpu-platform

Kubernetes 환경에서 NVIDIA A100 GPU를 **Full GPU / Time-slicing / MPS / MIG** 네 가지
방식으로 공유했을 때, 동일한 AI workload의 성능이 어떻게 달라지는지 비교하는 프로젝트.

OCR 서비스를 만드는 것이 목적이 아니다. OCR은 **재현 가능하고 측정하기 쉬운 GPU
workload**로서만 존재한다. 따라서 OCR 기능은 최소한으로 유지하고, 측정의 정확성과
환경 간 동일성을 우선한다.

---

## 1. 현재 상태

| 항목 | 상태 |
|---|---|
| OCR workload (FastAPI + PaddleOCR) | ✅ 완료 |
| CPU 환경 검증 | ✅ 완료 (수치는 3장) |
| 업로드 UI (정적 HTML) | ✅ 완료 |
| Dockerfile | ❌ 미착수 |
| Kubernetes 매니페스트 | ❌ 미착수 |
| GPU(A100) 검증 | ❌ 미착수 — **아직 한 번도 GPU에서 실행된 적 없음** |
| 4개 모드 벤치마크 | ❌ 미착수 |

### 디렉터리

```
k8s-gpu-platform/
└── workloads/
    └── inference/
        └── app/
            ├── main.py              FastAPI 앱 (/, /health, /gpu, /ocr)
            ├── ocr.py               엔진 생성·워밍업·업로드 검증·추론
            ├── gpu.py               device 탐지 (paddle 호출을 전부 예외 처리)
            ├── requirements.txt     앱 의존성 + PaddlePaddle 설치법 주석
            ├── static/
            │   └── index.html       업로드 화면 (바닐라 JS, 빌드/npm 없음)
            └── tests/
                └── test_api.py      테스트 8개
```

---

## 2. 실행

### CPU (개발/검증용)

```bash
cd workloads/inference/app

pip install paddlepaddle==3.3.1          # CPU 빌드
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

### GPU

```bash
# 노드/베이스 이미지의 CUDA 런타임에 맞는 인덱스를 하나 고를 것
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

### 확인

- 업로드 화면: <http://localhost:8000/> — 여러 장 동시 업로드 가능
- Swagger UI: <http://localhost:8000/docs>
- `curl http://localhost:8000/health`
- `curl http://localhost:8000/gpu` ← **GPU를 실제로 잡았는지 여기서 확인**

### 테스트

```bash
cd workloads/inference/app
pytest tests                  # CPU 전용 7개. 모델을 로드하지 않아 어디서든 실행됨
RUN_OCR_TESTS=1 pytest tests  # 실제 모델을 로드하는 추론 테스트까지 8개
```

---

## 3. API

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 업로드 화면 |
| `GET /health` | liveness. GPU를 건드리지 않음 |
| `GET /gpu` | CUDA/paddle device 상태. **GPU가 없어도 200을 반환** |
| `POST /ocr` | jpg/png 1장 → 텍스트 + confidence |

`POST /ocr` 응답:

```json
{
  "text": [{ "text": "스타벅스 강남점", "confidence": 0.936 }],
  "count": 1,
  "elapsed_ms": 1450.12
}
```

`elapsed_ms`는 **서버 측 추론 시간**이다. 네트워크 왕복 시간을 빼고 순수 추론만
비교하기 위해 넣었다. 벤치마크에서는 이 값을 쓴다.

### 업로드 화면의 파일명 생성

영수증 경비 처리를 위해 `26.08.05(주)우아한형제들_점심_메머드` 형식의 문자열을
만들어 복사할 수 있게 해 둔다. 추출 규칙은 다음과 같다.

| 자리 | 출처 |
|---|---|
| 날짜 | **결제일시**의 날짜를 `YY.MM.DD`로 정규화 |
| 가맹점 | **가맹점 정보 > 상호** 값 (결제 주체) |
| 용도 | 결제 **시각이 17시 이후면 `저녁`, 이전이면 `점심`** |
| 판매자 | **판매자 정보 > 상호**의 **첫 번째 단어** (실제 매장) |

배달앱 영수증처럼 결제 주체(가맹점)와 실제 매장(판매자)이 다른 경우를 전제한다.
두 섹션은 서로의 구간을 침범하지 않도록 경계를 두고 각각 탐색한다. 섹션이 없으면
가맹점은 판매자 섹션 앞쪽의 `상호`를, 판매자는 영수증 첫 줄을 폴백으로 쓴다.

네 항목 모두 화면에서 수정할 수 있다. OCR 추출은 휴리스틱이라 반드시 빗나가는
경우가 있기 때문이다.

**이 후처리는 전부 프론트엔드(JS)에서 한다.** 서버에 넣으면 `elapsed_ms`에
후처리 시간이 섞여 벤치마크 측정이 오염되므로 `/ocr` API는 손대지 않는다.

> PaddleOCR은 `결제일시 : 2026-08-05 10:48:07`을 `결제일시:2026-08-0510:48:07`처럼
> **공백을 지워서** 내놓는다. 날짜 파싱은 날짜와 시각을 한 덩어리로 잡아 이 형태를
> 처리한다. 사업자번호(`120-87-65763`)를 날짜로 오인하지 않도록 후보를 순회하며
> 달/일 범위가 유효한 첫 번째 것만 채택한다.

### 환경변수

ConfigMap으로 분리하기 쉽도록 모든 설정을 환경변수로 뺐다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `OCR_DEVICE` | `auto` | `cpu` / `gpu:0` 강제 지정 |
| `OCR_LANG` | `korean` | 인식 언어 모델 |
| `OCR_MAX_CONCURRENCY` | `1` | 동시 `predict()` 허용 수 (아래 4장 참고) |
| `OCR_WARMUP` | `true` | 시작 시 워밍업 추론 |
| `OCR_MAX_IMAGE_BYTES` | `10485760` | 업로드 크기 제한 |
| `OCR_USE_TEXTLINE_ORIENTATION` | `false` | 방향 분류. 측정 일관성을 위해 off |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

---

## 4. 벤치마크 설계

### 4.1 GPU 공유 방식별 특성

**핵심: "GPU니까 병렬"이 아니다.** 기본적으로 한 GPU에서 서로 다른 프로세스의
커널은 동시에 실행되지 않고, CUDA가 컨텍스트를 시분할해 번갈아 돌린다.
네 가지 모드는 정확히 이 지점에서 갈린다.

| 모드 | 진짜 동시 실행 | 메모리 격리 | 특징 |
|---|---|---|---|
| **Full GPU** | ❌ 프로세스 1개 독점 | — | 작은 추론은 A100을 다 못 채워 GPU가 논다 |
| **Time-slicing** | ❌ 시분할 | ❌ 없음 (OOM 전파) | 처리량은 안 늘고 지연만 증가. 목적은 속도가 아니라 "노는 GPU 나눠쓰기" |
| **MPS** | ✅ 커널 동시 실행 | ❌ 약함 | CUDA 컨텍스트 공유. 각 프로세스가 GPU를 못 채울 때 처리량이 실제로 오름 |
| **MIG** | ✅ 하드웨어 파티션 | ✅ 강함 | 격리 보장. 단 파티션 1개는 A100 전체보다 작음 |

**가설 (미검증):** 이 workload는 영수증 1장 = 작은 추론이라 A100을 다 못 채운다.
따라서 **MPS에서 가장 큰 처리량 이득**이 나올 것으로 예상한다. 이 프로젝트의
핵심 관전 포인트다.

### 4.2 동시성을 어디서 올릴 것인가

CPU에서 측정한 결과가 설계 근거를 제공한다 (영수증 1장 반복, 3장 참고):

```
OCR_MAX_CONCURRENCY=1          OCR_MAX_CONCURRENCY=4
동시 1 → 0.70 req/s             동시 1 → 0.69 req/s
동시 2 → 0.72 req/s             동시 2 → 0.72 req/s
동시 4 → 0.72 req/s             동시 4 → 0.72 req/s
         추론평균 3468ms                  추론평균 5297ms
```

**처리량이 완전히 동일하다.** 연산 장치가 이미 포화 상태라 프로세스 내부에서
스레드를 늘려도 처리량은 늘지 않고 지연만 나빠진다.

→ **비교 축은 pod(프로세스) 개수여야 한다.** MPS가 이득을 내는 것도 멀티
*프로세스*에서다. `OCR_MAX_CONCURRENCY=1`을 기본값으로 두고 replica를 늘리는
것이 맞는 설계인 이유다.

- ❌ pod 내부 스레드 동시성 늘리기
- ✅ pod 개수 × GPU 공유 모드 조합 비교

### 4.3 측정 지표

| 지표 | 출처 |
|---|---|
| 추론 지연 (p50/p95/p99) | 응답의 `elapsed_ms` |
| 처리량 (req/s) | 부하 도구 집계 |
| GPU 이용률 / 메모리 | `nvidia-smi`, DCGM exporter |
| 모드별 격리 실패 | OOM 발생 여부, 이웃 pod 영향 |

부하는 `k6` / `hey` 같은 도구로 건다. **UI는 기능 확인용이지 측정용이 아니다** —
브라우저·네트워크·JS 타이머 노이즈가 섞인다.

### 4.4 측정 정확도를 위해 이미 반영한 것

- OCR 모델은 lifespan에서 **1회 초기화 후 재사용** → 모델 로딩 시간이 측정에 안 섞임
- 시작 시 **워밍업 추론 1회** → 첫 요청의 lazy-load 지연 제거
- `OCR_USE_TEXTLINE_ORIENTATION=false` → 파이프라인 단계를 고정해 환경 간 동일성 확보
- 응답에 서버 측 `elapsed_ms` 포함 → 네트워크 오버헤드 분리

---

## 5. 앞으로 확인해야 할 것

### 5.1 GPU 환경 구축

- [ ] **Dockerfile 작성** — 베이스 이미지 CUDA 버전과 paddle 인덱스(`cu118`/`cu126`/`cu129`)를 **반드시 일치**시킬 것
- [ ] 이미지 빌드 & 레지스트리 푸시
- [ ] Kubernetes 매니페스트 (Deployment, Service, ConfigMap)
- [ ] NVIDIA device plugin / GPU Operator 설치 상태 확인

### 5.2 GPU 동작 검증 (최우선)

- [ ] `GET /gpu`가 `cuda_available: true`, `gpu_name: NVIDIA A100...`을 반환하는가
- [ ] 로그에 `initializing PaddleOCR (device=gpu:0 ...)`가 찍히는가
- [ ] `RUN_OCR_TESTS=1 pytest tests` 8개 통과
- [ ] 한글 영수증 인식 결과가 CPU와 **동일한가** (GPU/CPU 커널 차이로 결과가 갈리면 비교가 무의미)
- [ ] `elapsed_ms`가 CPU 대비 얼마나 줄어드는가 (CPU 기준값: **약 1,400ms**)
- [ ] `nvidia-smi`로 실제 GPU 메모리 점유 확인

### 5.3 모드별 환경 구성

- [ ] **Full GPU** — 기준선(baseline) 확보
- [ ] **Time-slicing** — device plugin ConfigMap에 `replicas` 설정
- [ ] **MPS** — MPS daemon 구성
- [ ] **MIG** — A100 파티션 분할. 사용 가능한 프로파일이 **40GB / 80GB 모델에 따라 다르므로**
      (`nvidia-smi mig -lgip`로 확인) 먼저 보유 장비를 확정할 것. 프로파일별 비교 여부도 결정 필요

### 5.4 벤치마크 실행

- [ ] 부하 도구 선정 및 시나리오 스크립트 작성
- [ ] 고정할 변수 정의 (이미지 크기, 요청 수, 워밍업 요청 수, 측정 시간)
- [ ] 4개 모드 × pod 개수(1/2/4/8) 매트릭스 측정
- [ ] 결과 표/그래프 정리

### 5.5 열려 있는 결정 사항

- [ ] 인식 모델 조합을 고정할 것인가 — 현재 검출은 `PP-OCRv5_server_det`, 인식은 `korean_PP-OCRv5_mobile_rec`가 자동 선택된다. **server/mobile 조합에 따라 GPU 이용률이 크게 달라지므로** 벤치마크 변수로 삼을지 고정할지 결정 필요
- [ ] 입력 이미지를 1종으로 고정할지, 크기별로 나눌지 (작은 이미지는 GPU를 못 채운다)
- [ ] 모델 파일을 이미지에 굽을지, PVC/initContainer로 뺄지 (pod 기동 시간에 영향)
- [ ] 배치 추론(`predict()`에 리스트 전달) 시나리오를 별도 축으로 추가할지

---

## 6. 알려진 제약 / 주의사항

- **`paddlepaddle-gpu`는 PyPI에 없다.** CUDA별 전용 인덱스에서 받아야 하며,
  `-i` 옵션이 전체 인덱스를 덮어쓰기 때문에 `requirements.txt`에 넣을 수 없다.
  설치법은 `requirements.txt` 주석에 문서화되어 있다.
- **Python 3.9 ~ 3.13만 지원.** PaddlePaddle 3.3.x는 3.14 휠을 제공하지 않는다.
  이미지는 `python:3.11` 또는 `3.12`로 빌드할 것.
- **PaddleOCR predictor는 thread-safe하지 않다.** `OCR_MAX_CONCURRENCY` 기본값 1이
  이를 보장한다. 올리려면 결과 정합성을 먼저 검증할 것.
- 모델은 첫 실행 시 `~/.paddlex/official_models/`로 다운로드된다. 컨테이너에서는
  이 경로가 기동 시간과 이미지 크기에 직접 영향을 준다.
- OCR 엔진 초기화가 실패해도 앱은 뜬다. `/gpu`가 원인을 노출하고 `/ocr`은 503을
  반환한다. CrashLoopBackOff 대신 원인이 보이도록 한 의도적 설계다.

---

## 7. 검증 기록

### CPU (2026-09-13, macOS arm64 / Python 3.12 / paddlepaddle 3.3.1 / paddleocr 3.7.0)

```
pytest              8 passed (실제 추론 테스트 포함)
GET /gpu            cuda_available: false, ocr_device: cpu, ocr_ready: true
POST /ocr (영수증)   10건 인식, confidence 0.936 ~ 0.9999, elapsed_ms 1400~1560
POST /ocr (영수증2)  7건 인식
동시성              MAX_CONCURRENCY 1/4 모두 0.72 req/s (4.2 참고)
```

한글 영수증 10줄을 **전부 정확히** 인식했다. 실제 인식 결과:

```
스타벅스 강남점 · 아메리카노 · 4,500 · 카페라떼 · 5,000
치즈케이크 · 6,500 · 합계 · 16,000 · 2026-09-13 14:22
```

### GPU

아직 실행된 적 없음.
