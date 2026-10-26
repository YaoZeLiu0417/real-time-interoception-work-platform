"""Audit whether a batch detector revises old decisions after more input.

This tests a necessary property of an irrevocable causal decision stream.
It does not validate physiological fiducials, ECG accuracy, physical latency,
or causality under all possible signals. No samples or event arrays are
included in the returned report.
"""
import math
import numpy as np


def _indices(detector, samples, fs):
    events=np.asarray(detector(samples.copy(),fs))
    if (events.ndim!=1 or (events.size and events.dtype.kind not in 'iu')
            or np.any(events<0) or np.any(events>=len(samples))
            or np.any(np.diff(events.astype(float))<=0)):
        raise ValueError('Detector must return strictly increasing integer decision indices inside its input')
    return events.astype(np.int64)


def audit_prefixes(samples, fs, detector, prefix_seconds, *, guard_s=5., start_s=5.):
    """Compare each shorter prefix with the longest requested prefix.

    Matching is exact sample-index equality, not a physiological event matcher.
    The terminal ``guard_s`` of each prefix is excluded from both event sets.
    Each row is a nested comparison on the same recording, not an independent
    trial or participant. Caller freezes sampling-rate and detector definitions.
    """
    x=np.asarray(samples,dtype=float)
    if x.ndim!=1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Finite nonempty one-dimensional samples required')
    for value in (fs,guard_s,start_s):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Finite sampling frequency and time limits required')
    if fs<=0 or guard_s<0 or start_s<0:raise ValueError('Invalid frequency or history limits')
    seconds=list(prefix_seconds)
    if (not seconds or any(isinstance(t,bool) or not isinstance(t,(int,float))
            or not math.isfinite(t) or t<=start_s+guard_s for t in seconds)
            or any(b<=a for a,b in zip(seconds,seconds[1:]))):
        raise ValueError('Strictly increasing prefixes with a nonempty stable history required')
    stops=[int(round(t*fs)) for t in seconds]
    if any(b<=a for a,b in zip(stops,stops[1:])) or stops[-1]>len(x):
        raise ValueError('Prefixes must be distinct sample lengths inside the recording')
    if stops[0]-int(math.ceil(guard_s*fs))<=int(math.ceil(start_s*fs)):
        raise ValueError('Stable history must contain at least one sample')
    future=_indices(detector,x[:stops[-1]],fs)
    rows=[]
    for requested,stop in zip(seconds,stops):
        early=future if stop==stops[-1] else _indices(detector,x[:stop],fs)
        lo=int(math.ceil(start_s*fs));hi=stop-int(math.ceil(guard_s*fs))
        a=early[(early>=lo)&(early<hi)];b=future[(future>=lo)&(future<hi)]
        removed=np.setdiff1d(a,b,assume_unique=True)
        added=np.setdiff1d(b,a,assume_unique=True)
        union=np.union1d(a,b)
        rows.append(dict(prefix_requested_s=float(requested),prefix_samples=stop,
            reference_prefix_samples=stops[-1],history_start_sample=lo,history_stop_sample=hi,
            earlier_history_decisions=len(a),later_history_decisions=len(b),
            historical_decisions_removed_after_append=len(removed),
            historical_decisions_added_after_append=len(added),
            changed_decision_memberships=len(removed)+len(added),union_decisions=len(union),
            changed_membership_fraction=(len(removed)+len(added))/len(union) if len(union) else None,
            future_samples_appended=stops[-1]-stop))
    tested=[r for r in rows if r['future_samples_appended']>0]
    return dict(schema_version='decision-prefix-audit-v1',fs=float(fs),start_s=float(start_s),
        guard_s=float(guard_s),matching='exact_sample_decision_indices',
        reference_prefix_s=stops[-1]/fs,comparisons=rows,
        informative_comparisons=len(tested),comparisons_with_revised_history=sum(r['changed_decision_memberships']>0 for r in tested),
        scope='Future-context sensitivity of batch decision indices; not ECG accuracy or physical latency',
        independent_physiological_reference_used=False,causality_proven_for_all_inputs=False)
