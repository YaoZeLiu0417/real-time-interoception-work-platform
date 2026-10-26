# Software verification · 26 October 2026

These records concern software behavior with synthetic signals, simulated
keys and zero participants. They do not validate an MRI installation,
physiology accuracy, physical display timing or response-device latency.

| Check | Observed scope | Record |
|---|---|---|
| Software suite | 213 tests passed; seven-point adjustment, endpoints, missingness, controller, GLM, freeze and validation plans | [Test record](../results/seven_point_software_validation_20261026.synthetic.json) |
| Native controller | PsychoPy 2025.1.1; actual software clock, 472.24 s, 6 development + 6 test blocks, 12 matching start/end receipts | [Integration record](../results/native_controller_validation_20261026.synthetic.json) |
| Final participant window | Three synchrony adjustments saved 5; two confidence adjustments returned to and saved 4; fixed-window end | [Presentation record](../results/seven_point_presentation_validation_20261026.synthetic.json) |
| Repeated independent profiles | 6 software runs × 6 artificial conditions × 2 test blocks; long-form blocks retained; separate test GLMs | [Profile report](../results/independent_profile_validation_20261026.synthetic.json) |

The full native run saved its validation plan, task space and source hashes
before development. Its optimizer and frozen snapshot stayed unchanged
during testing, and the development-volume count stayed at 123. It used one
prespecified test order, rather than claiming balance within a single run.
The repeated headless check covers all six Latin orders; its ROI function is
different from the native generator, so those response profiles are not
pooled.

Final instruction text and artwork were simplified after the full native
integration run. The separate final-window report records the resulting
sources and captures. That check also uses each rating page's own duration
for synthetic autoplay. Historical source hashes remain attached to the
run that actually executed them.

The response contract is `two_button_seven_point_v1`. The initial cursor 4
is not an answer until the participant interacts. Tests separately cover
untouched missing responses, returns to 4, boundary clamping and deadline
inputs. A no-response event is not interpreted as inability to feel a
heartbeat. The rating is ordinal; the III–IV MRI contrast remains separate.

Native captures show established pages after flip and interrupt rendering
performance. Software frame-interval warnings occurred on this workstation;
the run is an integration check, not a stimulus-timing acceptance test.
Reported rating RTs use the caller's task-clock polling time relative to
the model page start; they are not calibrated hardware reaction times.

![Synthetic independent-profile verification](assets/validation/independent-validation.synthetic.png)

Rebuild this figure after installing `requirements-figures.txt`:

```bash
python scripts/plot_independent_validation.py --summary results/independent_profile_validation_20261026.synthetic.json --out docs/assets/validation
```

The generic `prefix_audit.py` module checks whether a deterministic detector
revises historical sample indices when given longer prefixes at a fixed
sample rate. It is a necessary stability probe, not an ECG accuracy test or
an end-to-end causality certificate. It contains no human recordings or
clinical-derived outputs.

Earlier binary-presentation records retain their original contract and
are not relabeled as evidence for the seven-point endpoint.
