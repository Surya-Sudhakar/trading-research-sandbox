"""Frozen test oracle: original prefix-replay composition."""
from sandbox.market_state.discovery_market_context import build_discovery_market_contexts, with_level_transition
from sandbox.market_state.touch_transition import classify_level_transition
from sandbox.research.discovery_record_builder import (
    _STEP, _anchor_order, aggregate_completed_blocks, build_geometry_instance,
    build_geometry_interaction_histories, build_level_state_sequence,
    build_touch_transition_patterns, build_touch_transition_research_anchor,
    build_untouched_checkpoint_research_anchors, join_research_anchors_multitimeframe,
    join_research_anchors_sessions, assemble_research_record,
    build_research_session_relationships, measure_research_anchor_outcomes,
)

def reference_records_for_segment(bars, config):
    instances = {}
    # count is the number of COMPLETED evidence bars. Future begins at count.
    # Availability is checked only by segment length, never future prices.
    for count in range(1, len(bars) - max(config.horizons) + 1):
        prefix = bars[:count]
        decision = prefix[-1].open_time_utc + _STEP
        for candle in aggregate_completed_blocks(prefix, decision, config.block_config):
            if candle.block_start_utc not in instances:
                instances[candle.block_start_utc] = build_geometry_instance(
                    candle, config.geometry_family_id, config.level_specs)
        anchors = []
        transitions = {}
        for instance in instances.values():
            for history in build_geometry_interaction_histories(instance, prefix, decision):
                sequence = build_level_state_sequence(history)
                candidates = []
                for pattern in build_touch_transition_patterns(sequence):
                    anchor = build_touch_transition_research_anchor(pattern)
                    if anchor is not None:
                        candidates.append(anchor)
                        transitions[_anchor_order(anchor)] = classify_level_transition(pattern)
                candidates.extend(build_untouched_checkpoint_research_anchors(history, config.checkpoints))
                # Past anchors appear in later prefixes; emit exactly at formation.
                anchors.extend(a for a in candidates if a.evidence_end_utc == decision)
        if not anchors:
            continue
        anchors.sort(key=_anchor_order)
        mtf = join_research_anchors_multitimeframe(anchors, prefix)
        sessions = join_research_anchors_sessions(anchors, prefix)
        future = bars[count:count + max(config.horizons)]
        contexts = build_discovery_market_contexts(prefix)
        for anchor, mtf_context, session_context in zip(anchors, mtf, sessions):
            yield assemble_research_record(
                mtf_context, build_research_session_relationships(session_context),
                measure_research_anchor_outcomes(anchor, future, config.horizons),
                with_level_transition(contexts[anchor.evidence_end_utc],
                                      transitions.get(_anchor_order(anchor))))

