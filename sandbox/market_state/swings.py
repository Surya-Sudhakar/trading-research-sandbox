from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class ConfirmedSwing:
    kind: str
    index: int
    confirmation_index: int
    price: float


class PivotLabel(StrEnum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


class StructureBreakKind(StrEnum):
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"


@dataclass(frozen=True, slots=True)
class LabeledSwing:
    swing: ConfirmedSwing
    label: PivotLabel | None


@dataclass(frozen=True, slots=True)
class ConfirmedStructureBreak:
    kind: StructureBreakKind
    confirmation_index: int
    broken_swing: ConfirmedSwing


def confirmed_swings(highs, lows, left: int, right: int) -> list[ConfirmedSwing]:
    result=[]
    for i in range(left, len(highs)-right):
        # Strict inequality avoids ambiguous plateau pivots.
        if all(highs[i] > highs[j] for j in range(i-left, i+right+1) if j != i):
            result.append(ConfirmedSwing("HIGH", i, i+right, float(highs[i])))
        if all(lows[i] < lows[j] for j in range(i-left, i+right+1) if j != i):
            result.append(ConfirmedSwing("LOW", i, i+right, float(lows[i])))
    return result


def structure_counts(swings: list[ConfirmedSwing], count_window: int) -> dict[str, int]:
    result={"higher_high_count":0,"lower_high_count":0,"higher_low_count":0,"lower_low_count":0}
    for kind, higher, lower in (("HIGH","higher_high_count","lower_high_count"),("LOW","higher_low_count","lower_low_count")):
        values=[x.price for x in swings if x.kind==kind][-count_window:]
        for a,b in zip(values,values[1:]):
            if b>a:result[higher]+=1
            elif b<a:result[lower]+=1
    return result


def label_confirmed_swings(swings, as_of_index: int | None = None) -> tuple[LabeledSwing, ...]:
    """Label each visible pivot against the preceding confirmed pivot of its kind."""
    materialized = tuple(swings)
    if as_of_index is not None and (type(as_of_index) is not int or as_of_index < 0):
        raise ValueError("as_of_index must be a nonnegative integer")
    previous = {"HIGH": None, "LOW": None}
    result = []
    for swing in materialized:
        if not isinstance(swing, ConfirmedSwing) or swing.kind not in previous:
            raise ValueError("swings must contain valid ConfirmedSwing values")
        if as_of_index is not None and swing.confirmation_index > as_of_index:
            continue
        prior = previous[swing.kind]
        label = None
        if prior is not None and swing.price != prior.price:
            if swing.kind == "HIGH":
                label = PivotLabel.HH if swing.price > prior.price else PivotLabel.LH
            else:
                label = PivotLabel.HL if swing.price > prior.price else PivotLabel.LL
        result.append(LabeledSwing(swing, label))
        previous[swing.kind] = swing
    return tuple(result)


def confirmed_structure_breaks(closes, swings) -> tuple[ConfirmedStructureBreak, ...]:
    """Emit close-cross breaks only after their referenced pivots are confirmed.

    A bullish/bearish regime is established by an HH+HL/LH+LL pair and is
    subsequently changed by an opposing CHoCH. Crosses with the regime are BOS;
    crosses against it are CHoCH. With no established opposing regime a break is
    BOS. Each confirmed pivot can be broken at most once.
    """
    values = tuple(float(value) for value in closes)
    labeled = label_confirmed_swings(swings)
    by_confirmation = {}
    for item in labeled:
        by_confirmation.setdefault(item.swing.confirmation_index, []).append(item)
    latest = {"HIGH": None, "LOW": None}
    latest_label = {"HIGH": None, "LOW": None}
    broken = set()
    regime = None
    result = []
    for index, close in enumerate(values):
        newly_confirmed = by_confirmation.get(index, ())
        for item in newly_confirmed:
            latest[item.swing.kind] = item.swing
            latest_label[item.swing.kind] = item.label
        if newly_confirmed:
            if latest_label["HIGH"] is PivotLabel.HH and latest_label["LOW"] is PivotLabel.HL:
                regime = "BULLISH"
            elif latest_label["HIGH"] is PivotLabel.LH and latest_label["LOW"] is PivotLabel.LL:
                regime = "BEARISH"
        if index == 0:
            continue
        high = latest["HIGH"]
        if (high is not None and high.confirmation_index <= index
                and ("HIGH", high.index) not in broken
                and values[index - 1] <= high.price < close):
            kind = (StructureBreakKind.CHOCH_BULLISH if regime == "BEARISH"
                    else StructureBreakKind.BOS_BULLISH)
            result.append(ConfirmedStructureBreak(kind, index, high))
            broken.add(("HIGH", high.index))
            regime = "BULLISH"
        low = latest["LOW"]
        if (low is not None and low.confirmation_index <= index
                and ("LOW", low.index) not in broken
                and values[index - 1] >= low.price > close):
            kind = (StructureBreakKind.CHOCH_BEARISH if regime == "BULLISH"
                    else StructureBreakKind.BOS_BEARISH)
            result.append(ConfirmedStructureBreak(kind, index, low))
            broken.add(("LOW", low.index))
            regime = "BEARISH"
    return tuple(result)
