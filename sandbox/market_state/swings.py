from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfirmedSwing:
    kind: str
    index: int
    confirmation_index: int
    price: float


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
