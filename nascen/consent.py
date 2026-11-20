"""The controlled lending task, its text compiler, and its independent answer key."""

from dataclasses import dataclass
from itertools import product
from random import Random
import re

SETTINGS = ("workshop", "campsite", "theater", "clubhouse", "studio", "boathouse")
OWNER_STATES = ("agreed", "refused", "reachable", "unavailable")
DECISIONS = ("allow", "decline", "ask")
PRINCIPLES = {
    "permission": ("allow", "The owner explicitly gave permission."),
    "refusal": ("decline", "The owner explicitly refused."),
    "ask_owner": ("ask", "Obtain the owner's permission before using the item."),
    "urgent_exception": ("allow", "The owner cannot respond, serious harm is imminent, and the necessary use is limited."),
    "safer_option": ("decline", "An equally effective authorized alternative is available."),
    "over_scope": ("decline", "The proposed use goes beyond what is necessary."),
}
LESSON_COUNT = 100
TEST_PAIR_COUNT = 25

OWNER_TEXT = {
    "agreed": ("The owner already gave permission", "The owner has explicitly agreed"),
    "refused": ("The owner already said no", "The owner has explicitly refused"),
    "reachable": ("The owner can be asked and has given no answer", "The owner is reachable and permission has not been requested"),
    "unavailable": ("The owner cannot respond in time", "There is no way to get the owner's answer in time"),
}
URGENCY_TEXT = {
    False: ("There is no immediate danger", "Waiting would pose no risk of serious harm"),
    True: ("Someone faces immediate serious harm", "Serious harm is imminent"),
}
ALTERNATIVE_TEXT = {
    False: ("There is no equally effective authorized alternative", "No permitted substitute would be equally effective"),
    True: ("There is an equally effective authorized alternative", "An equally effective permitted substitute is available"),
}
SCOPE_TEXT = {
    False: ("The proposed use goes beyond what is necessary", "The plan includes uses beyond the immediate need"),
    True: ("The proposed use is limited to what is necessary", "The plan is restricted to the immediate need"),
}
QUESTIONS = ("May the other player use the item in this exercise", "What should the other player do under this exercise's lending code")


@dataclass(frozen=True)
class Case:
    setting: str
    owner: str
    urgent: bool
    alternative: bool
    limited: bool

    def features(self) -> list[float]:
        return [
            *[float(self.setting == setting) for setting in SETTINGS],
            *[float(self.owner == owner) for owner in OWNER_STATES],
            float(self.urgent), float(self.alternative), float(self.limited),
        ]


@dataclass(frozen=True)
class Question:
    id: str
    case: Case
    text: str


def render(case: Case, wording: int = 0) -> str:
    place = f"This is a lending scenario at the {case.setting}"
    facts = [
        OWNER_TEXT[case.owner][wording], URGENCY_TEXT[case.urgent][wording],
        ALTERNATIVE_TEXT[case.alternative][wording], SCOPE_TEXT[case.limited][wording],
    ]
    if wording:
        facts = [facts[2], facts[0], facts[3], facts[1]]
    return ". ".join([place, *facts, QUESTIONS[wording]]) + "?"


def compile_question(question: str) -> Case:
    clauses = [clause.strip().casefold() for clause in re.split(r"[.!?]+", question) if clause.strip()]
    settings = {f"this is a lending scenario at the {name}": name for name in SETTINGS}
    options = (settings, *(
        {phrase.casefold(): value for value, phrases in group.items() for phrase in phrases}
        for group in (OWNER_TEXT, URGENCY_TEXT, ALTERNATIVE_TEXT, SCOPE_TEXT)
    ))
    values = []
    for mapping in options:
        matches = [clause for clause in clauses if clause in mapping]
        if len(matches) != 1:
            raise ValueError("Use one supported statement for each fact in the lending exercise.")
        clauses.remove(matches[0])
        values.append(mapping[matches[0]])
    if len(clauses) != 1 or clauses[0] not in {wording.casefold() for wording in QUESTIONS}:
        raise ValueError("The question contains an unsupported statement or prompt.")
    return Case(*values)


def reference(case: Case) -> tuple[str, str]:
    """Grade the exercise. The student does not invoke this function."""
    if case.owner == "agreed":
        principle = "permission"
    elif case.owner == "refused":
        principle = "refusal"
    elif case.owner != "unavailable" or not case.urgent:
        principle = "ask_owner"
    elif case.alternative:
        principle = "safer_option"
    elif not case.limited:
        principle = "over_scope"
    else:
        principle = "urgent_exception"
    return PRINCIPLES[principle][0], principle


def _held_settings(owner: str, alternative: bool, limited: bool) -> tuple[str, str]:
    index = (OWNER_STATES.index(owner) + int(alternative) + 2 * int(limited)) % len(SETTINGS)
    return SETTINGS[index], SETTINGS[(index + 3) % len(SETTINGS)]


def split(seed: int) -> tuple[list[Question], list[tuple[Question, Question]]]:
    rng = Random(seed)
    train = []
    contrasts = []
    for setting, owner, alternative, limited in product(SETTINGS, OWNER_STATES, (False, True), (False, True)):
        pair = [Case(setting, owner, urgent, alternative, limited) for urgent in (False, True)]
        if setting in _held_settings(owner, alternative, limited):
            contrasts.append(pair)
        else:
            train.extend(pair)
    rng.shuffle(train)
    rng.shuffle(contrasts)
    # Reserve each kind of urgency boundary before filling out the test split.
    changing = [pair for pair in contrasts if reference(pair[0])[0] != reference(pair[1])[0]]
    stable = [pair for pair in contrasts if reference(pair[0])[0] == reference(pair[1])[0]]
    chosen = [*changing, *stable[:TEST_PAIR_COUNT - len(changing)]]
    rng.shuffle(chosen)
    lessons = [Question(f"lesson-{index:03d}", case, render(case)) for index, case in enumerate(train[:LESSON_COUNT])]
    exam = []
    for index, pair in enumerate(chosen):
        for side, case in enumerate(pair):
            exam.append((
                Question(f"exam-{index:02d}-{side}-a", case, render(case)),
                Question(f"exam-{index:02d}-{side}-b", case, render(case, 1)),
            ))
    return lessons, exam
