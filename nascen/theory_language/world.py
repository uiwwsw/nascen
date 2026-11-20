"""Closed-world scenarios used only to make supervised examples and answer keys."""

from itertools import product

STATES = tuple(product((False, True), repeat=3))
ACTIONS = {
    "physical": ("push", "no_push", "unblock", "block", "advance"),
    "social": ("owner_grants", "peer_grants", "urgent_now", "urgent_before", "use_in", "use_out"),
}
FAMILY = {
    "push": "force", "no_push": "force", "unblock": "barrier", "block": "barrier", "advance": "advance",
    "owner_grants": "permission", "peer_grants": "permission",
    "urgent_now": "time", "urgent_before": "time", "use_in": "use", "use_out": "use",
}
REFERENCES = {
    "force": "이 실험의 트랙에서 길이 열려 있으면 강하게 밀어 물체를 움직일 수 있다. 밀지 않거나 길이 막혀 있으면 새 움직임은 생기지 않는다.",
    "barrier": "장애물을 치우면 길이 열린다. 반대로 이동 경로에 장애물을 놓으면 길이 막히고 움직이던 물체는 멈춘다.",
    "advance": "시간이 지나면 열린 길에서 움직이던 물체만 목표선에 새로 도달한다. 이미 기록한 도착은 유지되며 이 단계가 끝나면 물체는 멈춘다.",
    "permission": "소유자가 승인하면 물건 사용 허락이 생긴다. 다른 사람의 동의는 기존 허락을 바꾸지 않는다.",
    "time": "지금 일어나고 있는 비상상황만 긴급 상태로 인정한다. 이미 끝난 비상상황은 현재 긴급 상태를 해제한다.",
    "use": "이 모의 훈련에서는 주인의 승인이 있거나 현재 비상상황이면 허용 범위 안에서 자원을 사용할 수 있다. 범위를 벗어나면 새 사용은 일어나지 않는다.",
}


def transition(state, action):
    first, second, third = state
    match action:
        case "push":
            return first or second, second, third
        case "no_push":
            return state
        case "unblock":
            return first, True, third
        case "block":
            return False, False, third
        case "advance":
            return False, second, third or (first and second)
        case "owner_grants":
            return True, second, third
        case "peer_grants":
            return state
        case "urgent_now":
            return first, True, third
        case "urgent_before":
            return first, False, third
        case "use_in":
            return first, second, third or first or second
        case "use_out":
            return state
    raise ValueError(f"Unknown action {action}.")


def describe_state(world, item, state, style):
    a, b, c = state
    if world == "physical":
        if style == 0:
            return f"트랙의 물체는 {item}이다. 물체는 {'이동 중이다' if a else '정지해 있다'}. 길은 {'열려 있다' if b else '막혀 있다'}. 목표선 도착 기록이 {'있다' if c else '없다'}."
        if style == 1:
            return f"{item} 실험 기록. 목표선을 통과한 적이 {'있으며' if c else '없으며'}, 통로에 장애물이 {'없다' if b else '있다'}. 물체는 {'움직이고 있다' if a else '멈춰 있다'}."
        return f"대상 물체는 {item}이다. 지금 {'움직이는' if a else '멈춰 있는'} 상태다. 앞은 {'비어 있다' if b else '장애물로 가로막혀 있다'}. 목표 지점에 닿은 이력이 {'있다' if c else '없다'}."
    if style == 0:
        return f"모의 훈련의 자원은 {item}이다. 소유자 승인이 {'있다' if a else '없다'}. 현재 비상상황이 {'있다' if b else '없다'}. 이 자원을 사용한 기록이 {'있다' if c else '없다'}."
    if style == 1:
        return f"{item} 이용 기록. 해당 물건을 {'쓴 적이 있으며' if c else '쓴 적이 없으며'}, 현재 대응할 비상 상황은 {'있다' if b else '없다'}. 주인은 이용을 {'승인했다' if a else '승인하지 않았다'}."
    return f"훈련용 {item}. 사용 이력은 {'이미 남았다' if c else '아직 없다'}. 지금 긴급 대응은 {'필요하다' if b else '필요하지 않다'}. 소유인의 허락은 {'확인되었다' if a else '확인되지 않았다'}."


def describe_action(action, style):
    return {
        "push": ("실험자가 물체를 세게 밀었다.", "실험자가 물체에 강한 밀기 힘을 가했다.", "담당자가 물체를 힘껏 떠밀었다."),
        "no_push": ("실험자가 물체를 세게 밀지 않았다.", "실험자는 물체에 밀기 힘을 가하지 않았다.", "담당자가 물체를 밀지 않고 지켜봤다."),
        "unblock": ("실험자가 통로에서 장애물을 치웠다.", "실험자가 길을 막는 물건을 꺼냈다.", "담당자가 이동 경로를 비웠다."),
        "block": ("실험자가 통로에 장애물을 놓았다.", "실험자가 물체가 지나는 길을 막았다.", "담당자가 이동 경로를 가로막았다."),
        "advance": ("실험자가 시간을 한 단계 진행했다.", "트랙에서 물체를 그대로 두고 한 단계 기다렸다.", "별도 조작 없이 실험을 다음 시점으로 진행했다."),
        "owner_grants": ("물건 소유자가 사용을 승인했다.", "물건의 주인이 이용해도 좋다고 분명히 말했다.", "해당 물품의 소유인이 사용에 동의했다."),
        "peer_grants": ("물건 소유자가 아닌 동료가 사용을 승인했다.", "주인과 관계없는 다른 사람이 써도 된다고 말했다.", "해당 물품을 소유하지 않은 사람이 사용에 동의했다."),
        "urgent_now": ("담당자가 지금 비상상황이 발생했다고 알렸다.", "담당자가 현재 긴급 대응이 필요하다고 보고했다.", "훈련 진행자가 현시점에 비상이 발생했음을 확인했다."),
        "urgent_before": ("담당자가 예전 비상상황은 이미 끝났다고 알렸다.", "담당자가 긴급 대응이 필요했던 일은 종료됐다고 보고했다.", "훈련 진행자가 비상사태는 과거 일이며 지금은 끝났다고 확인했다."),
        "use_in": ("참가자가 정해진 범위 안에서 자원을 쓰려고 했다.", "참가자가 훈련에서 허용한 용도로 물품을 사용하려 했다.", "훈련 참가자가 사용 범위를 지켜 이 자원을 쓰고자 했다."),
        "use_out": ("참가자가 정해진 범위 밖에서 자원을 쓰려고 했다.", "참가자가 훈련에서 허용하지 않은 용도로 물품을 사용하려 했다.", "훈련 참가자가 허용된 사용 범위를 벗어나 이 자원을 쓰고자 했다."),
    }[action][style]


# Experiment-only object identities. Evaluation names and surface style 2 never enter training.
TRAIN_ITEMS = {
    "physical": (
        "나무 상자", "고무공", "은색 수레", "벽돌", "플라스틱 통", "연습용 원통", "빨간 블록", "종이 상자",
        "작은 바구니", "천 주머니", "유리 구슬", "빈 여행 가방", "녹색 판자", "둥근 돌", "나무 원판", "낮은 의자",
        "실험용 카트", "모형 자동차", "금속 상자", "코르크 마개", "모래 자루", "회색 공", "파란 컵", "긴 막대",
    ),
    "social": (
        "우산", "머그잔", "충전기", "사인펜", "자전거", "태블릿", "사다리", "종이 가위", "손전등", "보호 장갑",
        "삼각대", "타이머", "돋보기", "전기 주전자", "색연필", "노트", "붓", "집게", "마우스", "헤드셋",
        "발판", "연습용 가방", "열쇠고리", "물통",
    ),
}
EXAM_ITEMS = {
    "physical": ("주황색 토막", "강철 공구함", "천을 감은 북", "고무 바퀴", "발포재 조각", "조각상 모형"),
    "social": ("무전기", "빔프로젝터", "마이크", "공기 펌프", "비상용 배터리", "녹음기"),
}
