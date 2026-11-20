"""Keep unseen text and evaluation labels outside model training."""

from dataclasses import replace

from .dataset import Question
from .grounded_data import held_out_exam
from .world import ACTIONS, describe_action, describe_state

ITEMS = {
    "physical": ("칠한 목재 도르래", "도금된 손잡이", "소형 테프론 받침", "철제 굴림통", "누런 사각 기둥", "아크릴 눈금판"),
    "social": ("훈련용 고도계", "분리형 선반", "방수 기록기", "실내용 경광봉", "접이식 깔개", "연습용 청진기"),
}

_ACTIONS = {
    "push": "실험 담당자가 밀어서 움직일 정도로 힘을 보탰다.",
    "no_push": "실험 담당자는 손을 떼고 있었고 밀어 주지 않았다.",
    "unblock": "앞길을 막던 물건을 경로 바깥으로 빼냈다.",
    "block": "물체가 갈 길에 차단물을 배치했다.",
    "advance": "아무 조작도 더 하지 않은 채 한 차례 시간이 흘렀다.",
    "owner_grants": "이 물건의 실제 임자가 이용을 승인했다.",
    "peer_grants": "실제 임자가 아닌 옆 사람이 이용해도 좋다고 응답했다.",
    "urgent_now": "진행자가 지금 당장 대응할 사건이 있다고 통보했다.",
    "urgent_before": "진행자는 대응하던 일이 이미 마무리됐다고 통보했다.",
    "use_in": "참가자는 약속에 들어 있는 목적으로 이 물품을 쓰려고 했다.",
    "use_out": "참가자는 약속에 들어 있지 않은 목적으로 이 물품을 쓰려고 했다.",
}


def _state(world, item, values):
    first, second, third = values
    if world == "physical":
        past = "목표 표시를 지난 전력이 있다" if third else "지금까지 목표 표시를 지난 적은 없다"
        route = "전방에는 진행을 방해하는 것이 없다" if second else "전방을 방해물이 가로지르고 있다"
        motion = "실험 대상의 위치가 계속 달라지고 있다" if first else "실험 대상의 위치가 그대로다"
        return f"{item} 관측. {past}. {route}. {motion}."
    past = "이미 한 번이라도 이 자원을 이용했다" if third else "이 자원을 이용한 일은 아직 없다"
    urgent = "바로 대응해야 할 사태가 진행 중이다" if second else "당장 대응해야 할 사태는 진행되고 있지 않다"
    consent = "임자가 사용을 허용했다" if first else "임자로부터 사용 허가를 얻지 못했다"
    return f"{item} 모의 상황. {past}. {urgent}. {consent}."


def independent_exam():
    original, keys = held_out_exam(ITEMS)
    original_actions = {describe_action(action, 0): action for actions in ACTIONS.values() for action in actions}
    questions, answers = [], []
    for question, key in zip(original, keys, strict=True):
        if key.state_style or key.action_style:
            continue
        item = ITEMS[key.world][int(key.group.rsplit("-", 1)[-1])]
        new_initial = _state(key.world, item, key.initial)
        new_actions = tuple(_ACTIONS[original_actions[text]] for text in question.actions)
        new_choices = tuple(tuple(_state(key.world, item, state) for state in options) for options in key.choices)
        for phase, before, actions, options in (
            ("original", question.before, question.actions, question.options),
            ("initial", new_initial, question.actions, question.options),
            ("action", question.before, new_actions, question.options),
            ("choices", question.before, question.actions, new_choices),
            ("all", new_initial, new_actions, new_choices),
        ):
            identifier = f"{phase}/{question.id}"
            questions.append(Question(identifier, before, actions, options))
            answers.append(replace(key, id=identifier, group=f"{phase}/{key.group}"))
    return tuple(questions), tuple(answers)
