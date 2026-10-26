"""Native PsychoPy + real controller integration with synthetic inputs only.

The clock advances in native wall time; the CLI never fast-forwards the
32-second HRF window. Simulated ROI acquisitions are delivered on that same
software clock. No scanner, physiology, author masks or human observations are
used. A native-window smoke run checks software integration, not physical timing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .contracts import finite, fingerprint, private_output
from .presentation import FAMILIES, RESPONSE_CONTRACT, TaskPresentation
from .psychopy_bridge import NamedTaskClock, PsychoPyBlockBridge
from .roi_glm import block_regressor
from .run_presentation import load_button_mapping, present_block, _texture
from .session import NeuroadaptiveSession
from .taskspace import load_taskspace
from .ui import FIELD, participant_card


DEMO_TIMINGS = dict(instruction_s=1.5, task_s=2., synchrony_s=2., confidence_s=2., fixation_s=1.)


def write_predevelopment_snapshot(out, space, plan, *, run_configuration=None):
    """Exclusively persist the declared plan, space and source before a window.

    Hashes identify software files, not device calibration or human validity.
    Refuse overwriting a prior declaration, including a partially written one.
    """
    folder = Path(__file__).resolve().parent
    names = ('native_session.py','presentation.py','run_presentation.py','ui.py',
             'session.py','roi_glm.py','neuroadaptive.py','psychopy_bridge.py',
             'validation.py','taskspace.py','contracts.py','taskspace.synthetic.json')
    declaration = dict(scope='Synthetic software declaration before the native window and development',
        response_contract=RESPONSE_CONTRACT, validation_plan=copy.deepcopy(plan),
        task_space=copy.deepcopy(space), run_configuration=copy.deepcopy(run_configuration or {}),
        source_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names},
        is_synthetic=True, participants=0, physical_timing_validated=False)
    declaration['snapshot_sha256'] = fingerprint(declaration)
    with (Path(out)/'validation_plan.synthetic.json').open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(declaration,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return declaration


def make_demo_space(*, seed=0, test_repeats=1, order_index=0):
    """Six distinct demonstration families; no claim of 18 human task presets.

    The validation order is chosen and hashed before development. Position
    balance applies across the complete Latin-square set, not this one run.
    """
    from .validation import make_validation_plan
    original = load_taskspace(Path(__file__).with_name('taskspace.synthetic.json'))
    source_hash = original.pop('definition_sha256')
    params = {
        'heartbeat_attention': dict(instruction_target='own_heartbeat'),
        'respiration_attention': dict(instruction_target='natural_respiration'),
        'visual_rhythm': dict(interval_s=.8, pulse_width_s=.12, target_probability=.25),
        'body_feedback_synchrony': dict(attention='heartbeat', linkage='current', pulse_width_s=.12),
        'gradcpt_standard': dict(period_s=.8, mountain_probability=.1),
        'gradcpt_reversed': dict(period_s=.8, mountain_probability=.1),
    }
    rows = []
    for family in FAMILIES:
        row = next(r for r in original['conditions'] if r['family'] == family and r['preset_id'] == 'p1')
        row['parameter_values'] = dict(params[family], parameter_status='explicit_toy_demo_only')
        row['implementation_status'] = 'synthetic_native_controller_demonstration_only'
        rows.append(row)
    plan = make_validation_plan([r['condition_id'] for r in rows], repeats=test_repeats, seed=seed)
    if isinstance(order_index, bool) or not isinstance(order_index, int) or not 0 <= order_index < len(plan['orders']):
        raise ValueError('Choose an existing predeclared validation order index')
    original.update(conditions=rows, fixed_test_plan=list(plan['orders'][order_index]['sequence']),
        clock_id='native_session_synthetic_task_clock', source_taskspace_sha256=source_hash,
        validation_plan_hash=plan['plan_hash'], validation_order_index=order_index,
        native_demo_scope='Six demonstration families, synthetic geometry and ROI values; not a validated human library')
    original['definition_sha256'] = fingerprint(original)
    return original, plan


class SyntheticROIStream:
    """Causal software delivery of artificial paired PSC values.

    Only this generator holds the artificial response function. It learns block
    schedules from accepted actual flip receipts, never from proposed starts.
    available_time_s is the time the outer poll actually delivers the sample;
    nominal delivery latency is retained separately as simulation metadata.
    """
    is_synthetic = True

    def __init__(self, space, *, origin_s, tr_s=2., latency_s=.2, seed=0):
        if space.get('is_synthetic') is not True:
            raise ValueError('Synthetic ROI source refuses a real task space')
        self.clock_id, self.roi_definition_id = space['clock_id'], space['roi_definition_id']
        self.origin_s, self.tr_s, self.latency_s = map(finite, (origin_s, tr_s, latency_s))
        if self.origin_s < 0 or not 0 < self.tr_s <= 4 or self.latency_s < 0:
            raise ValueError('Nonnegative origin/latency and TR in (0, 4] required')
        self._rng = np.random.default_rng(seed)
        self._noise = np.zeros(2)
        self._next_id = 0
        self._last_poll = self.origin_s
        self._blocks = {}
        self._truth = {}
        for row in space['conditions']:
            x, y = row['coordinates']
            self._truth[row['condition_id']] = np.array([
                .4+.8*np.exp(-((x-.25)**2+(y-.6)**2)/.1),
                .3+.7*np.exp(-((x-.8)**2+(y-.4)**2)/.1)])

    def block_started(self, receipt):
        command = receipt['command']
        key = command['block_id']
        if key in self._blocks or command['condition_id'] not in self._truth:
            raise ValueError('Unknown or duplicate generator block')
        self._blocks[key] = dict(condition_id=command['condition_id'],
            onset_s=finite(receipt['onset_s']), offset_s=None)

    def block_ended(self, receipt):
        block = self._blocks[receipt['block_id']]
        offset = finite(receipt['offset_s'])
        if block['offset_s'] is not None or offset <= block['onset_s']:
            raise ValueError('Invalid generator actual offset')
        block['offset_s'] = offset

    def poll(self, now_s):
        now = finite(now_s)
        if now < self._last_poll:
            raise ValueError('Synthetic source clock moved backwards')
        self._last_poll = now
        deliveries = []
        while self.origin_s+self._next_id*self.tr_s+self.latency_s <= now:
            sample = self.origin_s+self._next_id*self.tr_s
            value = np.zeros(2)
            for block in self._blocks.values():
                # Until the actual end is known, the input remains on. Future
                # or proposed offsets cannot change an already acquired sample.
                stop = block['offset_s'] if block['offset_s'] is not None else sample+1.
                if sample >= block['onset_s'] and stop > block['onset_s']:
                    regressor = block_regressor(np.array([sample]), block['onset_s'], stop-block['onset_s'])[0]
                    value += regressor*self._truth[block['condition_id']]
            self._noise = .35*self._noise+self._rng.normal(0.,.02,2)
            value += self._noise+np.array([.0001,-.00015])*(sample-self.origin_s)
            event = dict(kind='roi_volume', volume_id=self._next_id, sample_time_s=sample,
                roi_values=value.tolist(), available_time_s=now, clock_id=self.clock_id,
                roi_definition_id=self.roi_definition_id, is_synthetic=True, valid=True)
            deliveries.append(dict(event=event, simulation=dict(
                scheduled_available_time_s=sample+self.latency_s,
                delivery_basis='outer_poll_task_clock', artificial_roi_values=True)))
            self._next_id += 1
        return deliveries


class NativeSessionRunner:
    """Small serial handoff around the existing session and flip bridge.

    poll_inputs only delivers ROI samples. request_next/freeze may fit models
    and are called between blocks. The native presenter remains responsible
    for its flip loop. Supply clock_context explicitly in non-native tests.
    """
    def __init__(self, space, run_id, clock, *, validation_plan, budget=6, seed=0,
                 event_sink=None, roi_source=None, clock_context='native_psychopy_software_clock',
                 max_readout_wait_s=90.):
        if space.get('is_synthetic') is not True:
            raise ValueError('Native integration currently requires a synthetic task space')
        if not isinstance(clock, NamedTaskClock) or clock.clock_id != space['clock_id']:
            raise ValueError('One matching NamedTaskClock is required across the session')
        if clock_context not in ('native_psychopy_software_clock', 'deterministic_fake_clock'):
            raise ValueError('Explicit native-software or deterministic-fake clock context required')
        if (validation_plan.get('plan_hash') != space.get('validation_plan_hash')
                or fingerprint({k:v for k,v in validation_plan.items() if k != 'plan_hash'}) != validation_plan.get('plan_hash')
                or validation_plan['orders'][space['validation_order_index']]['sequence'] != space['fixed_test_plan']):
            raise ValueError('Validation plan must match the predeclared task space')
        self.space, self.clock, self.clock_context = copy.deepcopy(space), clock, clock_context
        self.run_id, self._sink = run_id, event_sink or (lambda event: None)
        self.session = NeuroadaptiveSession(space, run_id, seed=seed, budget=budget)
        self.max_readout_wait_s = finite(max_readout_wait_s)
        if self.max_readout_wait_s <= self.session.glm.hrf_tail_s:
            raise ValueError('Readout timeout must allow the complete 32-second HRF tail')
        self.bridge = PsychoPyBlockBridge(space, run_id, self.session.definition_hash)
        self.origin_s = finite(clock.getTime())
        self.source = roi_source or SyntheticROIStream(space, origin_s=self.origin_s, seed=seed)
        if (self.source.is_synthetic is not True or self.source.clock_id != space['clock_id']
                or self.source.roi_definition_id != space['roi_definition_id']):
            raise ValueError('ROI source provenance and definitions must match the session')
        self.volume_count = 0
        self.presentations = []
        self.receipts = []
        self._frozen_digest = None
        self._development_volume_count_at_freeze = None
        self._final_readouts = []
        self._status = 'running'
        self._sequence = 0
        self._record('validation_plan_frozen_before_development', validation_plan=validation_plan,
                     selected_order_index=space['validation_order_index'])

    def _record(self, kind, **fields):
        row = dict(kind=kind, sequence=self._sequence, run_id=self.run_id,
            task_time_s=finite(self.clock.getTime()), clock_id=self.clock.clock_id,
            clock_context=self.clock_context, timestamp_basis='software_task_clock',
            is_synthetic=True, physical_trigger_time_s=None, measured_output_time_s=None,
            response_contract=RESPONSE_CONTRACT, **fields)
        self._sequence += 1
        self._sink(copy.deepcopy(row))

    def dispatch(self, event, *, simulation=None):
        """Consume matching receipts or ROI data serially, outside callbacks."""
        now = finite(self.clock.getTime())
        kind = event.get('kind')
        if kind not in ('block_started', 'block_ended', 'roi_volume'):
            raise ValueError('Only actual software receipts and ROI inputs enter this handoff')
        if event.get('is_synthetic') is not True or event.get('clock_id') != self.clock.clock_id:
            raise ValueError('Native session input clock or provenance mismatch')
        if kind == 'roi_volume':
            if finite(event['available_time_s']) > now:
                raise ValueError('An input cannot be delivered before its declared availability')
        else:
            stamp = finite(event['onset_s'] if kind == 'block_started' else event['offset_s'])
            basis = 'onset_basis' if kind == 'block_started' else 'offset_basis'
            if (event.get(basis) != 'software_flip_callback' or stamp > now
                    or event.get('actual_flip_callback_time_s') != stamp
                    or event.get('physical_trigger_time_s') is not None
                    or event.get('measured_output_time_s') is not None):
                raise ValueError('A matching executed software flip receipt is required')
        response = self.session.dispatch(event)
        if event['kind'] == 'block_started':
            self.source.block_started(event)
            self.receipts.append(copy.deepcopy(event))
        elif event['kind'] == 'block_ended':
            self.source.block_ended(event)
            self.receipts.append(copy.deepcopy(event))
        elif event['kind'] == 'roi_volume':
            self.volume_count += 1
        self._record('controller_event', event=event, response=response, simulation=simulation)
        return response

    def accept_presenter_event(self, event):
        if event.get('kind') in ('block_started', 'block_ended'):
            return self.dispatch(event)
        self._record('presentation_event', event=event)

    def poll_inputs(self):
        # No GLM scoring or GP fitting is performed by this per-frame hook.
        deliveries = self.source.poll(self.clock.getTime())
        for delivery in deliveries:
            event = delivery['event']
            if self.session.phase == 'test' and event['sample_time_s'] < self.session.freeze_time:
                self._record('roi_sample_excluded', event=event,
                    reason='development_acquisition_delivered_after_freeze')
                continue
            self.dispatch(event, simulation=delivery.get('simulation'))
        return len(deliveries)

    def request_next(self):
        if self._status != 'running':
            return dict(status=self._status, reason='session_is_not_running')
        self.poll_inputs()
        requested = finite(self.clock.getTime())
        response = self.session.propose(requested)
        if response['status'] == 'waiting_for_eligible_readout':
            records = self.session.blocks if self.session.phase == 'development' else self.session.test_blocks
            latest = records[-1]
            if requested-latest['actual_offset_s'] > self.max_readout_wait_s:
                response = dict(status='stalled_missing_readout',block_id=latest['block_id'],
                    reason='no_eligible_readout_before_explicit_timeout',
                    last_readout=response['readout'], next_condition=None,
                    max_readout_wait_s=self.max_readout_wait_s)
                self._status = 'stalled_missing_readout'
        if response['status'] == 'ready_to_freeze':
            response = self.session.freeze(self.clock.getTime())
            self._frozen_digest = response['snapshot_sha256']
            self._development_volume_count_at_freeze = len(self.session.development_glm.volumes)
            self.bridge.set_phase('test', not_before_s=response['snapshot']['freeze_time_s'], freeze_receipt=response)
        elif response['status'] == 'test_complete':
            self._final_readouts = self.session.readouts(self.clock.getTime())
            self._status = 'completed'
        self._record('controller_request_next', response=response, request_time_s=requested,
                     controller_completed_time_s=finite(self.clock.getTime()))
        return response

    def present_command(self, win, kb, command, *, autoplay=False, timings=None,
                        key_aliases=None, status_sink=None):
        if command != self.session.pending:
            raise ValueError('Presentation must use the actual outstanding session command')
        row = next(r for r in self.space['conditions'] if r['condition_id'] == command['condition_id'])
        flow = TaskPresentation(row, mode='demo', clock_id=self.clock.clock_id,
            timings=timings or DEMO_TIMINGS, seed=len(self.presentations))
        record = dict(block_id=command['block_id'], command_condition_id=command['condition_id'],
                      display_condition_id=flow.condition_id, role=command['role'],
                      planned_task_duration_s=flow.timings['task_s'])
        self.presentations.append(record)
        self._record('presentation_requested', **record)
        result = present_block(win, kb, self.clock, flow, self.bridge, command,
            event_sink=self.accept_presenter_event, autoplay=autoplay, key_aliases=key_aliases,
            status_sink=status_sink, poll_inputs=self.poll_inputs)
        if result['status'] == 'aborted':
            self._status = 'aborted'
        return result

    def summary(self):
        starts = [r for r in self.receipts if r['kind'] == 'block_started']
        ends = [r for r in self.receipts if r['kind'] == 'block_ended']
        by_block = {r['command']['block_id']:r for r in starts}
        end_by_block = {r['block_id']:r for r in ends}
        matched = len(by_block) == len(self.presentations) and all(
            r['block_id'] in by_block and r['display_condition_id'] == r['command_condition_id']
            == by_block[r['block_id']]['condition_id'] for r in self.presentations)
        frozen_unchanged = self._frozen_digest is not None and fingerprint(self.session.frozen) == self._frozen_digest
        posterior_unchanged = self.session.frozen is not None and (
            fingerprint(self.session.optimizer.posterior()) == fingerprint(self.session.frozen['prediction']))
        training_volumes = len(self.session.development_glm.volumes) if self.session.phase == 'test' else None
        return dict(status=self._status, scope='Synthetic native-controller integration, not a neuroscience or hardware result',
            is_synthetic=True, participants=0, live_devices=0, run_id=self.run_id,
            response_contract=RESPONSE_CONTRACT,
            clock_context=self.clock_context, uses_virtual_clock=self.clock_context == 'deterministic_fake_clock',
            elapsed_software_clock_s=finite(self.clock.getTime())-self.origin_s,
            conditions=len(self.space['conditions']), development_blocks=len(self.session.blocks),
            test_blocks=len(self.session.test_blocks), roi_volume_count=self.volume_count,
            actual_software_start_receipts=len(starts), actual_software_end_receipts=len(ends),
            command_and_display_matched=matched, presentations=copy.deepcopy(self.presentations),
            block_timing=[dict(block_id=key, planned_duration_s=start['planned_duration_s'],
                onset_s=start['onset_s'], offset_s=end_by_block[key]['offset_s'],
                actual_software_duration_s=end_by_block[key]['software_duration_s'])
                for key,start in by_block.items() if key in end_by_block],
            hrf_tail_s=self.session.glm.hrf_tail_s, max_readout_wait_s=self.max_readout_wait_s,
            validation_plan_hash=self.space['validation_plan_hash'],
            selected_validation_order_index=self.space['validation_order_index'],
            selected_validation_sequence=list(self.space['fixed_test_plan']),
            frozen_snapshot_sha256=self._frozen_digest, frozen_snapshot_unchanged_after_test=frozen_unchanged,
            optimizer_posterior_unchanged_after_test=posterior_unchanged,
            development_volumes_at_freeze=self._development_volume_count_at_freeze,
            development_volumes_after_test=training_volumes,
            independent_test_glm=self.session.phase == 'test' and self.session.glm is not self.session.development_glm,
            final_test_readouts=copy.deepcopy(self._final_readouts),
            task_geometry='explicitly artificial coordinates', masks_used=False,
            physical_timing_validated=False, live_acquisition_ready=False)


def run(args):
    if not args.simulate:
        raise ValueError('Native session requires --simulate; real acquisition is not implemented')
    if finite(args.baseline_s) <= 0:
        raise ValueError('A positive synthetic baseline interval is required')
    from psychopy import visual, core
    from psychopy.hardware import keyboard
    import psychopy
    space, plan = make_demo_space(seed=args.seed, test_repeats=args.test_repeats, order_index=args.order_index)
    mapping, aliases = load_button_mapping(args.button_map)
    out = private_output(args.out)
    out.mkdir(parents=True)
    declaration = write_predevelopment_snapshot(out,space,plan,run_configuration=dict(
        seed=args.seed,budget=args.budget,test_repeats=args.test_repeats,order_index=args.order_index,
        baseline_s=args.baseline_s,max_readout_wait_s=args.max_readout_wait_s,
        autoplay=args.autoplay,button_mapping_sha256=fingerprint(mapping)))
    rows = []
    clock = NamedTaskClock(space['clock_id'], core.MonotonicClock())
    runner = NativeSessionRunner(space, args.run_id, clock, validation_plan=plan,
        budget=args.budget, seed=args.seed, event_sink=rows.append,
        max_readout_wait_s=args.max_readout_wait_s)
    win = None
    error = None
    try:
        win = visual.Window(size=(1280,800),fullscr=args.fullscreen,screen=args.screen,
            units='pix',color=FIELD,colorSpace='hex',checkTiming=False,autoLog=False,
            title='Interoception · SYNTHETIC NATIVE SESSION',allowGUI=not args.fullscreen)
        kb = keyboard.Keyboard()
        hold = visual.ImageStim(win, image=_texture(participant_card(dict(phase='fixation',is_synthetic=True))),
                                size=(1280,800), units='pix', autoLog=False)
        win.recordFrameIntervals = True
        runner._record('native_session_start', psychopy_version=psychopy.__version__,
            autoplay=args.autoplay, button_mapping=mapping, definition_hash=runner.session.definition_hash,
            predevelopment_snapshot_sha256=declaration['snapshot_sha256'],
            baseline_s=args.baseline_s, baseline_scope='Artificial zero-centered PSC, not a human baseline estimate')

        def wait_frame():
            for key in kb.getKeys(keyList=list(aliases),waitRelease=False,clear=True):
                if aliases[key.name] == 'escape':
                    runner._status = 'aborted'
                    return False
            hold.draw()
            win.flip()
            runner.poll_inputs()
            return True

        while clock.getTime() < runner.origin_s+args.baseline_s:
            if not wait_frame():
                break
        last_checked_count = -1
        last_request_time = float('-inf')
        while runner._status == 'running':
            if runner.volume_count != last_checked_count or clock.getTime()-last_request_time >= 1.:
                last_checked_count = runner.volume_count
                last_request_time = clock.getTime()
                response = runner.request_next()
                if response['status'] == 'recommendation':
                    runner.present_command(win,kb,response['command'],autoplay=args.autoplay,key_aliases=aliases)
                    last_checked_count = -1
                    continue
                if response['status'] == 'frozen':
                    last_checked_count = -1
                    continue
                if response['status'] == 'test_complete':
                    break
                if response['status'] == 'stalled_missing_readout':
                    break
            if not wait_frame():
                break
    except Exception as exc:
        runner._status = 'error'
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        frames = list(win.frameIntervals) if win else []
        if win:
            win.close()
        summary = dict(runner.summary(), error=error, psychopy_version=psychopy.__version__,
                       frame_count=len(frames), frame_intervals_s=frames, button_mapping_sha256=fingerprint(mapping),
                       predevelopment_snapshot_sha256=declaration['snapshot_sha256'])
        with (out/'native_session.synthetic.jsonl').open('x',encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        (out/'summary.synthetic.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return 0 if runner._status == 'completed' else 130 if runner._status == 'aborted' else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulate',action='store_true')
    p.add_argument('--autoplay',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-id',default='native_synthetic_run_1')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--budget',type=int,default=6)
    p.add_argument('--test-repeats',type=int,default=1)
    p.add_argument('--order-index',type=int,default=0)
    p.add_argument('--baseline-s',type=float,default=16.)
    p.add_argument('--max-readout-wait-s',type=float,default=90.)
    p.add_argument('--fullscreen',action='store_true')
    p.add_argument('--screen',type=int,default=0)
    p.add_argument('--button-map',type=Path,default=Path(__file__).with_name('button_mapping.template.json'))
    return p


if __name__ == '__main__':
    sys.exit(run(parser().parse_args()))
