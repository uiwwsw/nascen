# Nascen

**스스로 겪은 경험에서 배우는 AI를 만들고 싶습니다.**

*An AI that learns to grow.*

Nascen은 사전학습 모델 없이 출발해, 작은 세계에서 보고, 질문하고, 행동하고, 결과를 예측하며 배우는 과정을 탐구합니다. 첫 실험에서는 색과 형태를 따로 익힌 에이전트가 처음 보는 조합도 알아보는지 확인합니다.

## 실행

Python 3.12 이상과 PyTorch CPU 환경에서 실행합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m nascen status
python -m nascen train --steps 50
python -m nascen teach 1
python -m nascen feedback +1
```

`status`는 현재 세계의 물체와 모델의 추측을, `train`은 새 경험을 반영한 정확도와 예측 오차를 보여줍니다. `teach`는 목록에 나온 물체의 이름을 알려주며, `feedback`은 최근 행동에 한 번 평가를 남깁니다. 가중치와 경험은 `.nascen/lab.sqlite3`에 저장됩니다.

```sh
python -m nascen.benchmark
python -m unittest discover -s tests -v
```

## 다음 실험

질문에도 비용을 매기고, 같은 수의 답변을 받을 때 어떤 질문을 고르는 에이전트가 더 잘 배우는지 비교하려 합니다. 장기적으로는 물체 이름을 알아보는 데서 출발해 처음 받는 요청을 작은 행동들로 나누어 수행하는 것이 목표입니다.

[Apache-2.0](LICENSE)
