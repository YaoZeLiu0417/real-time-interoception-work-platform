"""Run synthetic ROI volumes through GLM, GP and independent profile validation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .contracts import private_output, fingerprint
from .session import NeuroadaptiveSession
from .roi_glm import block_regressor
from .taskspace import load_taskspace
from .validation import make_validation_plan, aggregate_test_profile


def synthetic_truth(coordinates):
    """An arbitrary verification function, never an insular biological result."""
    x,y=coordinates
    return np.array([.25+1.1*np.exp(-((x-.24)**2+(y-.34)**2)/.055),
                     .2+.9*np.exp(-((x-.72)**2+(y-.76)**2)/.065)])


def _spearman(first,second):
    """A constant/single-condition profile has no defined rank correlation."""
    if len(first)<2 or len(set(first))<2 or len(set(second))<2:return None
    value=float(spearmanr(first,second).statistic)
    return value if np.isfinite(value) else None


def run(space_path,output,*,seed=17,budget=16,runs=2,direction=1,acquisition='ei',
        validation_repeats=None,validation_seed=0):
    base_space=load_taskspace(space_path)
    if not base_space['is_synthetic']:raise ValueError('The simulator accepts synthetic spaces only')
    if isinstance(runs,bool) or not isinstance(runs,int) or runs<1:raise ValueError('Positive run count required')
    if validation_repeats is not None:
        validation_plan=make_validation_plan(list(dict.fromkeys(base_space['fixed_test_plan'])),
            repeats=validation_repeats,n_orders=runs,seed=validation_seed)
    else:
        # Preserve the registry's exact legacy sequence, including any repeats.
        validation_plan=dict(schema_version='independent-validation-plan-v0.1',
            design='registry_fixed_order',seed=None,n_orders=runs,
            n_rounds_per_order=None,repeats_per_condition_per_order=None,
            orders=[dict(order_index=i,sequence=list(base_space['fixed_test_plan'])) for i in range(runs)],
            position_balance=dict(scope='not_designed',exact_across_all_orders=False,
                within_single_order_claimed=False),carryover_balance_claimed=False,
            outcome_dependent=False)
        validation_plan['plan_hash']=fingerprint(validation_plan)
    out=private_output(output);out.mkdir(parents=True)
    # Persist before development, rather than reconstructing a plan after scores.
    (out/'validation_plan.prespecified.json').write_text(
        json.dumps(validation_plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    coordinates={r['condition_id']:r['coordinates'] for r in base_space['conditions']}
    allruns=[]
    for index in range(runs):
        space=copy.deepcopy(base_space)
        sequence=validation_plan['orders'][index]['sequence']
        if validation_repeats is not None:
            space.update(fixed_test_plan=list(sequence),validation_plan_hash=validation_plan['plan_hash'],
                         validation_order_index=index)
            space['definition_sha256']=fingerprint({k:v for k,v in space.items() if k!='definition_sha256'})
        rng=np.random.default_rng(seed+index)
        session=NeuroadaptiveSession(space,f'synthetic_run_{index+1}',seed=seed+index,
            budget=budget,direction=direction,acquisition=acquisition)
        events=[];volume_id=0;last_t=-2.;now=0.;noise=np.zeros(2)

        def dispatch(event):
            response=session.dispatch(event)
            events.append(dict(event=copy.deepcopy(event),response=copy.deepcopy(response),is_synthetic=True))
            return response

        def volumes_through(stop):
            nonlocal volume_id,last_t,now,noise
            times=np.arange(last_t+2.,stop+.001,2.)
            y=np.zeros((len(times),2))
            for block in session._records():
                h=block_regressor(times,block['onset_s'],block['duration_s'])
                y+=h[:,None]*synthetic_truth(coordinates[block['condition_id']])[None,:]
            for t,value in zip(times,y):
                noise=.35*noise+rng.normal(0,.035,2)
                signal=value+noise+np.array([.0001,-.00015])*t
                now=float(t+.25)
                dispatch(dict(kind='roi_volume',volume_id=volume_id,sample_time_s=float(t),
                    roi_values=signal.tolist(),available_time_s=float(t+.2),clock_id=space['clock_id'],
                    roi_definition_id=space['roi_definition_id'],is_synthetic=True,valid=True))
                volume_id+=1;last_t=float(t)

        volumes_through(18.)
        for _ in range(budget):
            decision=dispatch(dict(kind='request_next',now_s=now))
            if decision['status']!='recommendation':raise RuntimeError(decision['status'])
            onset=float(np.ceil(now/2)*2)
            dispatch(dict(kind='block_started',command=decision['command'],onset_s=onset,duration_s=30.))
            volumes_through(onset+28.)
            now=onset+30.01
            dispatch(dict(kind='block_ended',block_id=decision['command']['block_id'],offset_s=onset+30.))
            volumes_through(onset+30.+32.)
        frozen=dispatch(dict(kind='freeze',now_s=now))
        train_prediction=frozen['snapshot']['prediction']
        frozen_digest=frozen['snapshot_sha256']
        train_rows=frozen['snapshot']['development_readouts']
        training_volume_count=len(session.development_glm.volumes)
        # A new GLM and an outcome-independent test plan form a separate stage.
        volumes_through(last_t+20.)
        for _ in space['fixed_test_plan']:
            decision=dispatch(dict(kind='request_next',now_s=now))
            onset=float(np.ceil(now/2)*2)
            dispatch(dict(kind='block_started',command=decision['command'],onset_s=onset,duration_s=30.))
            volumes_through(onset+28.)
            now=onset+30.01
            dispatch(dict(kind='block_ended',block_id=decision['command']['block_id'],offset_s=onset+30.))
            volumes_through(onset+30.+32.)
        tests=session.readouts(now)
        if any(r['status']!='ok' for r in tests):raise RuntimeError('Incomplete independent test profile')
        # Long form is authoritative. A dict keyed by condition would silently
        # keep only its last block when a fixed test sequence contains repeats.
        test_rows=[dict(copy.deepcopy(r),sequence_position=i+1,
             round_index=(i//len(set(sequence))+1) if validation_repeats is not None else None,
             measured_contrast=float(direction*r['beta_difference'])) for i,r in enumerate(tests)]
        profile=aggregate_test_profile(tests,direction=direction)
        for p in profile:
            c=p['condition_id']
            p.update(coordinates=coordinates[c],predicted_mean=train_prediction[c]['mean'],
                predicted_sd=train_prediction[c]['sd'],
                known_synthetic_truth=float(direction*(synthetic_truth(coordinates[c])[0]-synthetic_truth(coordinates[c])[1])))
        pred=[p['predicted_mean'] for p in profile];obs=[p['independent_test_contrast'] for p in profile]
        rho=_spearman(pred,obs)
        result=dict(run_id=session.run_id,is_synthetic=True,direction=direction,
            development_blocks=len(session.blocks),test_blocks=len(tests),volume_count=volume_id,
            virtual_run_seconds=now,
            frozen_snapshot_sha256=frozen_digest,profile_spearman=rho,
            profile_rmse=float(np.sqrt(np.mean((np.array(pred)-obs)**2))),
            profile_metric_scope='frozen_GP_predictions_vs_unique_condition_test_means_descriptive',
            development_condition_ids=[b['condition_id'] for b in session.blocks],
            development_readouts=train_rows,profile=profile,
            test_readouts=test_rows,test_sequence=list(sequence),
            validation_plan_hash=validation_plan['plan_hash'],validation_order_index=index,
            validation_plan_frozen_before_development=True,
            run_taskspace_sha256=space['definition_sha256'],
            test_glm_separate_from_development=session.glm is not session.development_glm,
            training_volume_count_at_freeze=training_volume_count,
            training_volume_count_after_test=len(session.development_glm.volumes),
            frozen_prediction_unchanged_during_test=session.optimizer.posterior()==train_prediction,
            optimizer_frozen_during_test=True,
            limitations=['Repeated test-block betas can share GLM error and temporal noise.',
                'Repeat SD is descriptive, not an iid standard error or confidence interval.',
                'Synthetic restart runs are not participants; no policy superiority is established.'])
        allruns.append(result)
        with (out/f'{session.run_id}.private.jsonl').open('x',encoding='utf-8') as stream:
            for row in events:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print('Completed',session.run_id,flush=True)
    summary=dict(schema_version='neuroadaptive-simulation-v0.2',is_synthetic=True,participants=0,
        nature_masks_used=False,live_hardware_connected=False,psychopy_gui_executed=False,
        scope='Software integration using arbitrary ROI response functions, not human neuroscience results',
        taskspace_sha256=base_space['definition_sha256'],acquisition=acquisition,
        validation_plan=validation_plan,runs=allruns)
    if len(allruns)>=2:
        first,second=({p['condition_id']:p for p in r['profile']} for r in allruns[:2])
        common=[c for c in first if c in second]
        summary['independent_run_profile_spearman']=_spearman(
            [first[c]['predicted_mean'] for c in common],[second[c]['predicted_mean'] for c in common])
        summary['independent_run_test_profile_spearman']=_spearman(
            [first[c]['independent_test_contrast'] for c in common],
            [second[c]['independent_test_contrast'] for c in common])
        summary['independent_run_comparison']=dict(run_ids=[r['run_id'] for r in allruns[:2]],
            condition_ids=common,scope='first_two_synthetic_restart_runs_not_participants')
    (out/'summary.synthetic.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--space',type=Path,default=Path(__file__).with_name('taskspace.synthetic.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--budget',type=int,default=16);p.add_argument('--runs',type=int,default=2)
    p.add_argument('--direction',type=int,choices=[-1,1],default=1);p.add_argument('--acquisition',choices=['ei','gp_ucb'],default='ei')
    p.add_argument('--validation-repeats',type=int,default=None,
        help='Complete fixed-test rounds per order; omitted preserves the registry sequence')
    p.add_argument('--validation-seed',type=int,default=0,
        help='Independent prespecified Latin-order seed; exact position balance needs N run orders')
    a=p.parse_args();result=run(a.space,a.out,seed=a.seed,budget=a.budget,runs=a.runs,direction=a.direction,
        acquisition=a.acquisition,validation_repeats=a.validation_repeats,validation_seed=a.validation_seed)
    print(json.dumps(dict(is_synthetic=True,runs=len(result['runs']),participants=0)))
