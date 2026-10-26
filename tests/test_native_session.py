import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from experiments.interoception.contracts import fingerprint
from experiments.interoception.native_session import (
    DEMO_TIMINGS, NativeSessionRunner, SyntheticROIStream, make_demo_space, run,
    write_predevelopment_snapshot,
)
from experiments.interoception.psychopy_bridge import NamedTaskClock


class FakeClock:
    def __init__(self): self.now = 0.
    def getTime(self): return self.now
    def reset(self): pass


class FakeWindow:
    def __init__(self, clock):
        self.clock, self.callbacks = clock, []
        self.inside_callback = False
        self.frames = 0
    def callOnFlip(self, function, *args, **kwargs): self.callbacks.append((function, args, kwargs))
    def flip(self):
        self.clock.now += .25
        self.frames += 1
        callbacks, self.callbacks = self.callbacks, []
        self.inside_callback = True
        try:
            for function,args,kwargs in callbacks: function(*args,**kwargs)
        finally: self.inside_callback = False


class FakeKeyboard:
    def __init__(self): self.clock = FakeClock()
    def getKeys(self, **kwargs): return []
    def clearEvents(self): pass


class FakeStim:
    def __init__(self, *args, **kwargs): self.__dict__.update(kwargs)
    def draw(self): pass


def install_fake_psychopy(monkeypatch):
    module = ModuleType('psychopy')
    module.visual = SimpleNamespace(ImageStim=FakeStim, ShapeStim=FakeStim, Circle=FakeStim)
    monkeypatch.setitem(sys.modules, 'psychopy', module)


def setup_runner(*, repeats=1, budget=6, source=None):
    space, plan = make_demo_space(seed=11, test_repeats=repeats)
    native = FakeClock()
    clock = NamedTaskClock(space['clock_id'], native)
    events = []
    runner = NativeSessionRunner(space,'fake_software_run',clock,validation_plan=plan,
        budget=budget,seed=11,event_sink=events.append,roi_source=source,
        clock_context='deterministic_fake_clock')
    return runner, native, events


def test_demo_space_has_six_explicit_presentable_families_and_predeclared_test_order():
    space, plan = make_demo_space(seed=7,test_repeats=2,order_index=2)
    assert len(space['conditions']) == 6
    assert len({r['family'] for r in space['conditions']}) == 6
    assert all(r['parameter_values']['parameter_status'] == 'explicit_toy_demo_only' for r in space['conditions'])
    assert all('synthetic_preset' not in r['parameter_values'] for r in space['conditions'])
    assert len(space['fixed_test_plan']) == 12
    assert space['fixed_test_plan'] == plan['orders'][2]['sequence']
    assert space['definition_sha256'] == fingerprint({k:v for k,v in space.items() if k != 'definition_sha256'})
    assert space['validation_plan_hash'] == plan['plan_hash']
    assert plan['position_balance']['within_single_order_claimed'] is False


def test_full_native_presenter_fake_window_flow_reaches_gp_freeze_and_separate_test(monkeypatch):
    install_fake_psychopy(monkeypatch)
    runner, native, events = setup_runner()
    win, kb = FakeWindow(native), FakeKeyboard()
    old_poll, old_fit = runner.source.poll, runner.session.optimizer.fit
    updates_during_test = []

    def checked_poll(now):
        assert not win.inside_callback
        return old_poll(now)

    def checked_fit(observations):
        assert not win.inside_callback
        if runner.session.phase == 'test': updates_during_test.append(True)
        return old_fit(observations)

    monkeypatch.setattr(runner.source,'poll',checked_poll)
    monkeypatch.setattr(runner.session.optimizer,'fit',checked_fit)
    native.now = 16.5  # explicitly virtual acquisition/delivery in this fake-window check
    for _ in range(400):
        response = runner.request_next()
        if response['status'] == 'recommendation':
            result = runner.present_command(win,kb,response['command'],autoplay=True)
            assert result['status'] == 'completed'
        elif response['status'] == 'test_complete':
            break
        elif response['status'] != 'frozen':
            native.now += 2.
    else:
        pytest.fail('The causal fake-clock integration never completed')
    report = runner.summary()
    assert report['status'] == 'completed'
    assert report['uses_virtual_clock'] is True
    assert report['development_blocks'] == 6 and report['test_blocks'] == 6
    assert report['actual_software_start_receipts'] == report['actual_software_end_receipts'] == 12
    assert report['command_and_display_matched'] is True
    assert report['hrf_tail_s'] == 32.
    assert report['frozen_snapshot_unchanged_after_test'] is True
    assert report['optimizer_posterior_unchanged_after_test'] is True
    assert report['independent_test_glm'] is True and not updates_during_test
    assert report['development_volumes_at_freeze'] == report['development_volumes_after_test']
    assert len(report['final_test_readouts']) == 6
    assert all(r['status'] == 'ok' and r['latest_sample_time_s'] >= r['block_offset_s']+32.
               for r in report['final_test_readouts'])
    requests = [r['response'] for r in events if r['kind'] == 'controller_request_next']
    assert any(r.get('optimization',{}).get('reason') == 'maximum_ei' for r in requests if r.get('optimization'))
    assert events[0]['kind'] == 'validation_plan_frozen_before_development'
    assert all(r['physical_trigger_time_s'] is None and r['measured_output_time_s'] is None for r in events)
    assert all(r['clock_context'] == 'deterministic_fake_clock' for r in events)
    assert report['live_acquisition_ready'] is False and report['participants'] == 0
    ratings = [r['event'] for r in events if r['kind'] == 'presentation_event'
               and r['event'].get('event') == 'rating_recorded']
    assert len(ratings) >= 2  # the fixed test includes the body-feedback probe
    assert all(r['response_contract'] == 'two_button_seven_point_v1' and r['scale'] == 'seven_point_v1'
               and 1 <= r['final_cursor'] <= 7 for r in ratings)
    assert all(r['value'] == r['final_cursor'] if r['interaction_count'] else r['value'] is None for r in ratings)


def test_source_uses_actual_delivery_time_separate_from_synthetic_due_time():
    space, _ = make_demo_space()
    source = SyntheticROIStream(space,origin_s=0.,tr_s=2.,latency_s=.2,seed=4)
    assert source.poll(.19) == []
    rows = source.poll(5.5)
    assert [r['event']['sample_time_s'] for r in rows] == [0.,2.,4.]
    assert all(r['event']['available_time_s'] == 5.5 for r in rows)
    assert rows[0]['simulation']['scheduled_available_time_s'] == .2
    assert source.poll(5.5) == []
    with pytest.raises(ValueError,match='backwards'): source.poll(5.4)


def test_source_cannot_use_proposed_block_or_access_future_samples():
    runner,native,_ = setup_runner()
    proposal = runner.request_next()
    assert proposal['status'] == 'recommendation'
    assert runner.source._blocks == {}
    assert not runner.session.blocks
    assert all(row['event']['sample_time_s'] <= 0. for row in runner.source.poll(0.))
    assert runner.session.optimizer.posterior()[proposal['command']['condition_id']]['mean'] == 0.


class SilentSource:
    is_synthetic = True
    def __init__(self,space):
        self.clock_id, self.roi_definition_id = space['clock_id'], space['roi_definition_id']
    def poll(self,now): return []
    def block_started(self,event): pass
    def block_ended(self,event): pass


def test_missing_roi_after_actual_end_remains_waiting_and_does_not_invent_a_score():
    space,plan = make_demo_space()
    native = FakeClock();clock = NamedTaskClock(space['clock_id'],native)
    runner = NativeSessionRunner(space,'no_input',clock,validation_plan=plan,budget=1,
        roi_source=SilentSource(space),clock_context='deterministic_fake_clock')
    win = FakeWindow(native)
    command = runner.request_next()['command']
    runner.bridge.queue_start_on_flip(win,command,clock,2.,runner.accept_presenter_event)
    win.flip();native.now += 2.
    runner.bridge.queue_end_on_flip(win,command['block_id'],clock,runner.accept_presenter_event)
    win.flip();native.now += 34.
    response = runner.request_next()
    assert response['status'] == 'waiting_for_eligible_readout'
    assert response['readout']['beta_difference'] is None
    assert runner.session.pending is None and len(runner.session.blocks) == 1
    assert runner.summary()['frozen_snapshot_sha256'] is None
    native.now += 100.
    timed_out = runner.request_next()
    assert timed_out['status'] == 'stalled_missing_readout' and timed_out['next_condition'] is None
    assert timed_out['last_readout']['beta_difference'] is None
    assert runner.session.pending is None and runner.request_next()['status'] == 'stalled_missing_readout'


def test_wrong_pending_command_is_refused_before_gui_import():
    runner,_,_ = setup_runner()
    command = runner.request_next()['command']
    changed = dict(command,condition_id='another')
    with pytest.raises(ValueError,match='outstanding'):
        runner.present_command(None,None,changed)
    assert runner.presentations == []


def test_clock_provenance_and_frozen_plan_tampering_are_rejected():
    space, plan = make_demo_space()
    native = FakeClock()
    with pytest.raises(ValueError,match='NamedTaskClock'):
        NativeSessionRunner(space,'wrong',NamedTaskClock('other',native),validation_plan=plan)
    changed = copy.deepcopy(plan);changed['orders'][0]['sequence'].reverse()
    with pytest.raises(ValueError,match='Validation plan'):
        NativeSessionRunner(space,'wrong',NamedTaskClock(space['clock_id'],native),validation_plan=changed)
    changed_space = dict(space,is_synthetic=False)
    with pytest.raises(ValueError,match='synthetic'):
        NativeSessionRunner(changed_space,'wrong',NamedTaskClock(space['clock_id'],native),validation_plan=plan)


def test_future_declared_arrival_and_unexecuted_flip_receipt_are_not_accepted():
    runner,native,_ = setup_runner()
    with pytest.raises(ValueError,match='before.*availability'):
        runner.dispatch(dict(kind='roi_volume',is_synthetic=True,clock_id=runner.clock.clock_id,
            available_time_s=1.))
    command = runner.request_next()['command']
    with pytest.raises(ValueError,match='executed software'):
        runner.dispatch(dict(kind='block_started',command=command,onset_s=0.,duration_s=2.,
            is_synthetic=True,clock_id=runner.clock.clock_id))
    assert runner.session.pending == command and not runner.session.blocks


def test_cli_real_mode_is_refused_before_importing_psychopy_or_creating_outputs(tmp_path):
    with pytest.raises(ValueError,match='--simulate'):
        run(SimpleNamespace(simulate=False,out=tmp_path/'never_created'))
    assert not (tmp_path/'never_created').exists()


def test_predevelopment_declaration_persists_plan_space_and_source_without_overwrite(tmp_path):
    space,plan = make_demo_space(seed=8)
    declaration = write_predevelopment_snapshot(tmp_path,space,plan,run_configuration={'seed':8})
    stored = json.loads((tmp_path/'validation_plan.synthetic.json').read_text(encoding='utf-8'))
    assert stored == declaration and stored['validation_plan'] == plan and stored['task_space'] == space
    assert stored['response_contract'] == 'two_button_seven_point_v1'
    assert stored['snapshot_sha256'] == fingerprint({k:v for k,v in stored.items() if k != 'snapshot_sha256'})
    folder = Path(__file__).parents[1]/'experiments/interoception'
    for name in ('native_session.py','presentation.py','run_presentation.py','ui.py','session.py'):
        assert stored['source_sha256'][name] == hashlib.sha256((folder/name).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_predevelopment_snapshot(tmp_path,space,plan)
