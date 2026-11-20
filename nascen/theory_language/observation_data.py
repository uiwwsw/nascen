"""Training-only language views of observable state and event meanings."""

from collections import defaultdict

from .grounded_data import training_specs
from .world import ACTIONS, STATES, describe_action, describe_state, transition

STATE_CODES = tuple((world, state) for world in ACTIONS for state in STATES)
ACTION_CODES = tuple(action for actions in ACTIONS.values() for action in actions)

# These paraphrases belong to training. The held Korean wording lives in the separate evaluator.
_STATE_VIEWS = {
    "physical": (
        (("물체가 멈춰 있다", "물체가 움직이고 있다"),
         ("이동로에 장애물이 있다", "이동로에 장애물이 없다"),
         ("아직 목표선에 간 적이 없다", "목표선에 한 번 도착했다")),
        (("대상이 제자리에 서 있다", "대상이 이동한다"),
         ("경로가 장애물로 차단됐다", "경로가 막히지 않았다"),
         ("도착 이력이 없다", "도착 이력이 있다")),
        (("움직임이 없다", "움직임이 계속된다"),
         ("진행할 길이 가로막혔다", "진행할 길이 확보됐다"),
         ("목표 지점에 이른 적이 없다", "목표 지점에 이른 적이 있다")),
        (("실험 물체가 정지 상태다", "실험 물체가 운동 중이다"),
         ("통로가 폐쇄 상태다", "통로가 개방 상태다"),
         ("이전에 목표 표시를 통과하지 않았다", "이전에 목표 표시를 통과했다")),
    ),
    "social": (
        (("소유자의 허락을 받지 않았다", "소유자의 허락을 받았다"),
         ("지금 비상이 아니다", "지금 비상 상황이다"),
         ("물건을 이용한 적이 없다", "물건을 이용한 적이 있다")),
        (("주인의 승인이 나지 않았다", "주인의 승인이 났다"),
         ("현재 긴급 상황이 없다", "현재 긴급 상황이 있다"),
         ("사용한 이력이 없다", "사용한 이력이 남아 있다")),
        (("물건 주인이 사용에 동의하지 않았다", "물건 주인이 사용에 동의했다"),
         ("급히 대처할 일이 없는 상태다", "급히 대처할 일이 생긴 상태다"),
         ("이 물품은 아직 사용되지 않았다", "이 물품은 전에 사용됐다")),
        (("자원 소유자가 승낙하지 않았다", "자원 소유자가 승낙했다"),
         ("지금은 평상시다", "지금은 비상시다"),
         ("이용 기록이 남지 않았다", "이용 기록이 남았다")),
    ),
}
_ACTION_VIEWS = {
    "push": ("담당자가 큰 힘으로 물체를 앞으로 밀어 주었다.", "직원이 실험 물체를 강하게 떠밀었다.",
             "사람이 물체에 충분히 큰 밀기 힘을 가했다.", "실험자가 물건을 힘껏 밀었다."),
    "no_push": ("담당자가 물체를 밀지 않았다.", "실험자는 물체를 그대로 두고 힘을 가하지 않았다.",
                "사람이 물체를 떠밀지 않았다.", "직원이 물건에 미는 힘을 주지 않았다."),
    "unblock": ("장애물을 경로 밖으로 옮겼다.", "담당자가 장애물을 제거했다.",
                "실험자가 통로를 가로막던 물건을 없앴다.", "직원이 통행을 가로막는 물건을 치웠다."),
    "block": ("이동로를 막도록 장애물을 설치했다.", "담당자가 통로에 가림막을 세웠다.",
              "실험자가 장애물로 경로를 차단했다.", "직원이 물건을 놓아 통행을 막았다."),
    "advance": ("추가 조작 없이 다음 시간으로 넘어갔다.", "실험자가 한 번 더 시간이 지나게 했다.",
                "잠시 그대로 두어 실험 시간이 한 차례 흘렀다.", "한 단계 기다리고 다음 시점으로 갔다."),
    "owner_grants": ("자원 소유자가 써도 좋다고 답했다.", "물건 주인이 정식으로 이용을 허락했다.",
                     "물품을 가진 당사자가 사용에 동의했다.", "해당 물건의 주인이 승낙했다."),
    "peer_grants": ("주인이 아닌 동료가 써도 좋다고 답했다.", "소유자가 아닌 제삼자가 이용을 허락했다.",
                    "물품을 가지지 않은 다른 사람이 사용에 동의했다.", "해당 물건의 주인이 아닌 지인이 승낙했다."),
    "urgent_now": ("진행자가 현 시점에 비상이라고 공지했다.", "담당자가 바로 대응해야 할 일이 생겼다고 알렸다.",
                   "진행자가 지금 일어난 비상 상황을 알렸다.", "직원이 현재 긴급 대응이 필요하다고 보고했다."),
    "urgent_before": ("진행자가 예전 비상은 이미 해제됐다고 공지했다.", "담당자가 비상 상황은 끝났다고 알렸다.",
                      "진행자가 비상 상태였던 때가 지났다고 밝혔다.", "직원이 과거의 긴급 상황은 종료됐다고 보고했다."),
    "use_in": ("훈련에서 허용한 용도로 자원을 이용하려 했다.", "참가자가 룰에서 허용한 범위 안으로 사용을 요청했다.",
               "훈련생이 합의된 용도대로 물건을 쓰려 했다.", "참가자가 승인 가능한 용도에 자원을 쓰려고 했다."),
    "use_out": ("훈련에서 금지한 용도로 자원을 이용하려 했다.", "참가자가 허용 범위 밖의 사용을 요청했다.",
                "훈련생이 합의된 용도를 벗어나 물건을 쓰려 했다.", "참가자가 금지된 용도에 자원을 쓰려고 했다."),
}
_ORDERS = ((0, 1, 2), (1, 0, 2), (2, 1, 0), (2, 0, 1))

_EXTRA_CLAUSES = {
    "physical": (
        (("물체의 자리가 바뀌지 않고 있다", "물체의 자리가 계속 바뀐다"),
         ("경로 가운데 가로막는 것이 놓여 있다", "경로 가운데 가로막는 것이 없다"),
         ("기준 위치를 지나 본 적이 없다", "기준 위치를 지나 본 적이 있다")),
        (("속도가 없는 상태다", "이동이 진행되고 있다"),
         ("갈 길에 물체가 놓여 있다", "갈 길이 장애물 없이 트였다"),
         ("도착했다고 기록된 일이 없다", "도착했다고 기록된 일이 있다")),
    ),
    "social": (
        (("소유인의 허가가 없다", "소유인의 허가가 있다"),
         ("당장 처리해야 하는 일이 없다", "당장 처리해야 하는 일이 있다"),
         ("과거에 이 자원을 쓴 적이 없다", "과거에 이 자원을 쓴 적이 있다")),
        (("물품의 실제 주인이 허락하지 않았다", "물품의 실제 주인이 허락했다"),
         ("대응할 사건이 지금 벌어지지 않았다", "대응할 사건이 지금 벌어지고 있다"),
         ("아직 한 번도 물건을 사용하지 않았다", "전에 한 번 이상 물건을 사용했다")),
    ),
}


def clause_views():
    clauses = defaultdict(set)
    for world_index, world in enumerate(ACTIONS):
        for view in (*_STATE_VIEWS[world], *_EXTRA_CLAUSES[world]):
            for slot, pair in enumerate(view):
                for value, text in enumerate(pair):
                    clauses[6 * world_index + 2 * slot + value].add(text)
    background = {
        6 * len(ACTIONS): ("트랙 실험 기록", "모의 훈련 자원", "대상 물체의 종류를 기록했다",
                           "훈련실 창문은 닫혀 있다", "물건에 파란 표시를 붙였다",
                           "방금 상황을 관찰하고 정리했다", "이 문장은 제목이다", "모의 상황")
    }
    return {key: tuple(sorted(value)) for key, value in sorted(clauses.items())} | background


def training_views(size=300, seed=42):
    state_contexts = set()
    triplets = []
    for world, item, state, actions, _ in training_specs(size, seed):
        for action in actions:
            after = transition(state, action)
            state_contexts.update(((world, item, state), (world, item, after)))
            triplets.append((STATE_CODES.index((world, state)), ACTION_CODES.index(action), STATE_CODES.index((world, after))))
            state = after

    states = defaultdict(set)
    for world, item, state in sorted(state_contexts):
        label = STATE_CODES.index((world, state))
        states[label].update(describe_state(world, item, state, style) for style in (0, 1))
        for view, order in zip(_STATE_VIEWS[world], _ORDERS, strict=True):
            clauses = [view[i][state[i]] for i in order]
            intro = f"트랙 실험 대상은 {item}이다." if world == "physical" else f"모의 훈련 자원은 {item}이다."
            states[label].add(intro + " " + ". ".join(clauses) + ".")
    events = {
        i: tuple(describe_action(action, style) for style in (0, 1)) + _ACTION_VIEWS[action]
        for i, action in enumerate(ACTION_CODES)
    }
    return {label: tuple(sorted(views)) for label, views in sorted(states.items())}, events, tuple(triplets)
