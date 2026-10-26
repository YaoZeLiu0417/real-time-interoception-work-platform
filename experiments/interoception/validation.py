"""Outcome-independent test orders and descriptive repeated-block profiles.

Cyclic Latin rows balance serial position across a complete set of N orders.
They do not balance first-order carryover, and a single row is not balanced.
Block means/SDs below are descriptive; shared GLM errors need not be iid.
"""
from collections import Counter
import copy
import random
from statistics import fmean, stdev
from .contracts import finite, fingerprint


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def make_validation_plan(condition_ids, *, repeats=1, n_orders=None, seed=0):
    """Return complete, hashable sequences before any outcome is observed.

    ``repeats`` means complete rounds per order, not additional repetitions.
    Each order repeats its Latin row unchanged. Orders may be assigned to
    separate runs; they are not participants or independent biological samples.
    """
    if not isinstance(condition_ids, (list, tuple)) or not condition_ids:
        raise ValueError('A nonempty ordered condition list is required')
    ids = list(condition_ids)
    if any(not isinstance(c, str) or not c.strip() for c in ids) or len(set(ids)) != len(ids):
        raise ValueError('Condition IDs must be unique nonempty strings')
    repeats = _positive_integer(repeats, 'repeats')
    n = len(ids)
    n_orders = n if n_orders is None else _positive_integer(n_orders, 'n_orders')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    base = ids.copy()
    random.Random(seed).shuffle(base)
    orders = []
    for index in range(n_orders):
        offset = index % n
        row = base[offset:] + base[:offset]
        sequence = row * repeats
        orders.append(dict(order_index=index, latin_row=offset, sequence=sequence,
            n_rounds=repeats, blocks_per_round=n,
            condition_counts=dict(Counter(sequence)),
            adjacent_duplicate_count=sum(a == b for a, b in zip(sequence, sequence[1:]))))
    plan = dict(schema_version='independent-validation-plan-v0.1',
        design='cyclic_latin_square', seed=seed, condition_ids=ids, base_order=base,
        n_conditions=n, n_orders=n_orders, repeats_per_condition_per_order=repeats,
        n_rounds_per_order=repeats, blocks_per_order=n * repeats, orders=orders,
        position_balance=dict(scope='same_serial_position_across_orders',
            complete_cycles=n_orders // n, partial_cycle_orders=n_orders % n,
            exact_across_all_orders=n_orders % n == 0,
            within_single_order_claimed=False),
        carryover_balance_claimed=False,
        adjacent_duplicates_avoidable_within_order=n > 1 or repeats == 1,
        outcome_dependent=False,
        interpretation='Prespecified orders; no participant count or efficacy claim')
    plan['plan_hash'] = fingerprint(plan)
    return plan


def aggregate_test_profile(readouts, *, direction=1):
    """Aggregate one final test-GLM snapshot without collapsing its repeats.

    Keep the complete input separately as the long-form block record. Duplicate
    block IDs are rejected: refits of one block are not independent repeats.
    Pending/unestimable blocks remain in counts, with no invented measurement.
    """
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError('direction must be +1 or -1')
    groups, seen_blocks, run_ids = {}, set(), set()
    for row in readouts:
        block, condition = row.get('block_id'), row.get('condition_id')
        if not isinstance(block, str) or not block or block in seen_blocks:
            raise ValueError('Unique block IDs required in a final snapshot')
        if not isinstance(condition, str) or not condition:
            raise ValueError('Each block requires a condition ID')
        if row.get('role', 'test') != 'test':
            raise ValueError('Only independent test blocks may enter this profile')
        status = row.get('status')
        if status not in ('ok', 'pending', 'unestimable'):
            raise ValueError('Unknown test-readout status')
        if row.get('run_id') is not None:
            run_ids.add(row['run_id'])
        if len(run_ids) > 1:
            raise ValueError('Aggregate separate runs separately')
        seen_blocks.add(block)
        groups.setdefault(condition, []).append(copy.deepcopy(row))
    profile = []
    for condition, rows in groups.items():
        values = [direction * finite(r['beta_difference']) for r in rows if r['status'] == 'ok']
        profile.append(dict(condition_id=condition,
            test_block_ids=[r['block_id'] for r in rows], n_test_blocks=len(rows),
            n_valid_test_blocks=len(values), n_missing_test_blocks=len(rows)-len(values),
            readout_status_counts=dict(Counter(r['status'] for r in rows)),
            independent_test_contrast=fmean(values) if values else None,
            independent_test_sd=stdev(values) if len(values) > 1 else None,
            independent_test_min=min(values) if values else None,
            independent_test_max=max(values) if values else None,
            dispersion='sample_sd_across_test_blocks_descriptive_not_iid_CI'))
    return profile
