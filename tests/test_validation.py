from collections import Counter
import copy
import json
import math
from pathlib import Path
import pytest
from experiments.interoception.contracts import fingerprint
from experiments.interoception.validation import make_validation_plan, aggregate_test_profile
from experiments.interoception.simulate_neuroadaptive import run


def test_complete_cyclic_latin_set_balances_every_serial_position_and_keeps_repeats():
    ids = ['heart', 'breath', 'visual', 'feedback', 'grad', 'reversed']
    plan = make_validation_plan(ids, repeats=3, seed=72)
    assert len(plan['orders']) == len(ids)
    sequences = [r['sequence'] for r in plan['orders']]
    for sequence in sequences:
        assert Counter(sequence) == Counter({c: 3 for c in ids})
        assert all(a != b for a, b in zip(sequence, sequence[1:]))
    for position in range(18):
        assert Counter(s[position] for s in sequences) == Counter(ids)
    assert plan['position_balance']['exact_across_all_orders']
    assert not plan['position_balance']['within_single_order_claimed']
    assert not plan['carryover_balance_claimed']


def test_partial_and_repeated_latin_cycles_are_reported_without_false_balance_claim():
    partial = make_validation_plan(['a', 'b', 'c'], repeats=2, n_orders=2, seed=3)
    assert partial['position_balance'] == dict(scope='same_serial_position_across_orders',
        complete_cycles=0, partial_cycle_orders=2, exact_across_all_orders=False,
        within_single_order_claimed=False)
    complete = make_validation_plan(['a', 'b', 'c'], n_orders=6)
    for position in range(3):
        assert Counter(r['sequence'][position] for r in complete['orders']) == {'a': 2, 'b': 2, 'c': 2}
    singleton = make_validation_plan(['only'], repeats=3)
    assert not singleton['adjacent_duplicates_avoidable_within_order']
    assert singleton['orders'][0]['adjacent_duplicate_count'] == 2


def test_plan_is_seeded_serializable_and_hashes_the_full_prespecified_sequence():
    original = ['a', 'b', 'c', 'd']
    plan = make_validation_plan(original, repeats=2, n_orders=4, seed=17)
    assert plan == make_validation_plan(original, repeats=2, n_orders=4, seed=17)
    assert original == ['a', 'b', 'c', 'd']
    assert json.loads(json.dumps(plan)) == plan
    payload = {k: v for k, v in plan.items() if k != 'plan_hash'}
    assert fingerprint(payload) == plan['plan_hash']
    changed = copy.deepcopy(payload)
    changed['orders'][0]['sequence'].reverse()
    assert fingerprint(changed) != plan['plan_hash']


@pytest.mark.parametrize('ids,options', [
    ([], {}), (['a', 'a'], {}), ('abc', {}), (['a', ' '], {}),
    (['a'], {'repeats': True}), (['a'], {'repeats': 0}),
    (['a'], {'n_orders': 0}), (['a'], {'n_orders': 2.5}),
    (['a'], {'seed': True}), (['a'], {'seed': -1}),
])
def test_invalid_prespecified_inputs_are_rejected(ids, options):
    with pytest.raises(ValueError):
        make_validation_plan(ids, **options)


def test_repeated_condition_means_use_every_block_not_the_last_dict_entry():
    rows = [dict(block_id='a1', condition_id='a', status='ok', beta_difference=0.),
            dict(block_id='b1', condition_id='b', status='ok', beta_difference=4.),
            dict(block_id='a2', condition_id='a', status='ok', beta_difference=10.),
            dict(block_id='b2', condition_id='b', status='pending', beta_difference=None)]
    before = copy.deepcopy(rows)
    a, b = aggregate_test_profile(rows)
    assert a['test_block_ids'] == ['a1', 'a2']
    assert a['independent_test_contrast'] == 5.
    assert a['independent_test_sd'] == pytest.approx(math.sqrt(50))
    assert a['n_test_blocks'] == a['n_valid_test_blocks'] == 2
    assert b['n_test_blocks'] == 2 and b['n_missing_test_blocks'] == 1
    assert b['independent_test_contrast'] == 4. and b['independent_test_sd'] is None
    assert rows == before
    assert aggregate_test_profile(rows, direction=-1)[0]['independent_test_contrast'] == -5.


def test_missing_readouts_are_not_zero_and_training_refits_are_not_test_repeats():
    row = dict(block_id='x', condition_id='a', role='test', status='unestimable', beta_difference=None)
    summary = aggregate_test_profile([row])[0]
    assert summary['independent_test_contrast'] is None and summary['n_valid_test_blocks'] == 0
    with pytest.raises(ValueError, match='Unique block'):
        aggregate_test_profile([row, row])
    with pytest.raises(ValueError, match='Only independent test'):
        aggregate_test_profile([dict(row, role='development')])
    with pytest.raises(ValueError, match='separate runs'):
        aggregate_test_profile([dict(row, run_id='one'), dict(row, block_id='y', run_id='two')])
    for invalid in (True, float('nan'), float('inf'), None):
        with pytest.raises(ValueError):
            aggregate_test_profile([dict(row, status='ok', beta_difference=invalid)])


def test_repeated_simulation_retains_long_form_and_freezes_order_before_development(tmp_path):
    source = Path(__file__).resolve().parents[1]/'experiments/interoception/taskspace.synthetic.json'
    cfg = json.loads(source.read_text(encoding='utf-8'))
    cfg['conditions'] = cfg['conditions'][:3]
    cfg['fixed_test_plan'] = [r['condition_id'] for r in cfg['conditions']]
    cfg['optimizer']['burn_in'] = 1
    path = tmp_path/'small.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    result = run(path, tmp_path/'repeated', budget=1, runs=3,
                 validation_repeats=2, validation_seed=51)
    plan = result['validation_plan']
    assert result['participants'] == 0
    assert plan['position_balance']['exact_across_all_orders']
    for index, r in enumerate(result['runs']):
        assert r['validation_plan_hash'] == plan['plan_hash']
        assert r['test_sequence'] == plan['orders'][index]['sequence']
        assert len(r['test_readouts']) == r['test_blocks'] == 6
        assert len(r['profile']) == 3
        assert len({row['block_id'] for row in r['test_readouts']}) == 6
        assert all(p['n_valid_test_blocks'] == 2 for p in r['profile'])
        assert r['test_glm_separate_from_development'] and r['optimizer_frozen_during_test']
        for p in r['profile']:
            values = [row['measured_contrast'] for row in r['test_readouts']
                      if row['condition_id'] == p['condition_id']]
            assert p['independent_test_contrast'] == pytest.approx(sum(values)/2)
        log = [json.loads(line) for line in
               (tmp_path/f'repeated/synthetic_run_{index+1}.private.jsonl').read_text().splitlines()]
        frozen = next(row['response']['snapshot'] for row in log if row['event']['kind'] == 'freeze')
        assert frozen['test_plan'] == r['test_sequence']
