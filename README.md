# Nascen

**AI가 새로운 규칙 하나를 배우려면, 왜 그렇게 많은 예제를 먼저 봐야 할까?**

Nascen은 이 질문에서 출발한 오픈소스 연구 프로젝트입니다. 작은 세계에서 직접 실험해 규칙을 발견하고, 어떤 경험을 학습할지 골라, 처음 보는 상황에도 배운 내용이 통하는지 확인합니다. ‘태어나다’를 뜻하는 라틴어 *nascī*에서 이름을 따왔습니다.

*Learning which experiences matter, and testing whether the lessons transfer.*

[![CI](https://github.com/uiwwsw/nascen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uiwwsw/nascen/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-285d48)](LICENSE)

## 지금까지의 실험

### Experiment 001 · 규칙 발견

**가설.** 밀기 전후를 몇 번 관찰하면 처음 보는 위치에서도 결과를 예측할 수 있다. 조건부 규칙 모델은 새 배치 61개의 위치를 100% 맞혔습니다. 능동·무작위 탐색 모두 평균 3회 만에 정확도 기준에 도달했고, 능동 탐색의 CPU 비용이 더 컸습니다. [실험 상세](docs/experiments.md#experiment-001--규칙-발견)

### v0.3 · 선택 학습

**가설.** 오답이 섞여 있어도 검증한 경험을 선별하면, 같은 답변 수와 갱신 횟수에서 더 잘 배울 수 있다. 미학습 조합 정확도는 검증 후 선별 39.8%, 무작위 선별 33.2%였습니다. 검증을 거친 쪽은 오답을 적게 반영했고 CPU 시간은 더 들었습니다. [실험 상세](docs/experiments.md#v03--선택-학습)

### Experiment 002 · 교사 없이 답하기

**가설.** 교사에게 배운 작은 학생이 교사를 끈 뒤에도 새로운 사례를 판단할 수 있다. 미리 정한 규칙으로 검증한 수업 33개를 사용했을 때 결정 정확도는 **80.0%**였습니다. 긴급성만 바꾸면 결정도 달라져야 하는 쌍에서는 **4.2%**를 기록했습니다. 입력에는 지정된 영어 문형을 구조화하는 파서를 사용했습니다. [실험 상세](docs/experiments.md#experiment-002--교사-없이-답하기)

### Experiment 003 · 문장과 조건의 일반화

**가설.** 배운 규칙을 새로운 표현과 학습에 없던 조건 조합에도 적용할 수 있다. 긴급성 문장 한 줄만 새 표현으로 바꾸면 파서가 **0/50문항**을 처리했습니다. 조건 조합을 학습에서 통째로 뺀 별도 시험에서는 구조화된 사실을 직접 제공했을 때도 긴급성이 달라진 쌍 **72개 중 2개**에서만 두 결정을 모두 맞혔습니다. [실험 상세](docs/experiments.md#experiment-003--문장과-조건의-일반화)

### Experiment 004 · 오답에서 배우기

**가설.** 채점기가 알려준 오답을 교사에게 질문하면 처음 보는 문제에 더 잘 답할 수 있다. 교사를 끈 뒤 새 사례 24개를 시험했습니다. 오답 중심 교정은 결정 정확도를 **40.3%에서 81.9%**로 높였고, 질문 횟수를 8회로 맞춘 무작위 비교군은 **87.5%**였습니다. [실험 상세](docs/experiments.md#experiment-004--오답에서-배우기)

[이전 버전 비교, 전체 실험 절차와 원본 결과](docs/experiments.md)

## 바로 실행하기

Python 3.12~3.14와 CPU를 사용합니다.

```sh
git clone https://github.com/uiwwsw/nascen.git
cd nascen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nascen train --steps 50
python -m nascen knowledge
```

[실행 옵션과 재현 명령](docs/experiments.md#바로-실행하기) · [Apache-2.0](LICENSE)

## 관련 연구

- **규칙 발견과 실험 선택.** 여러 관찰에서 물리 법칙을 찾는 접근과, 경쟁 인과 가설을 구분할 실험을 고르는 접근입니다. Nascen은 Experiment 001에서 작은 밀기 환경을 대상으로 탐색 전략과 비용을 측정했습니다. [AI-Newton](https://journals.aps.org/prresearch/abstract/10.1103/dvvc-4rsr) · [Tong & Koller (2001)](https://ijcai.org/proceedings/2001/118)
- **데이터 선택.** Data Agent는 훈련 중 학습할 데이터를 고르는 정책을 연구합니다. Nascen v0.3은 오답이 섞인 조건에서 검증 후 선별과 무작위 선별을 비교했습니다. [Data Agent](https://proceedings.mlr.press/v306/yang26bq.html)
- **교사의 근거에서 배우기.** Distilling Step-by-Step은 답변과 설명을 학생 모델 훈련에 사용합니다. Nascen은 Experiment 002에서 결정과 구조화된 근거로 학생을 학습시키고 교사를 종료한 뒤 평가했습니다. [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/)
