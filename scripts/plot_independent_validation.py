"""Plot a synthetic frozen-profile check, never human or accuracy evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


def plot(source, out):
    source, out = Path(source), Path(out)
    data = json.loads(source.read_text(encoding='utf-8'))
    if data.get('is_synthetic') is not True or data.get('participants') != 0:
        raise ValueError('This plot accepts zero-participant synthetic reports only')
    plan, runs = data['validation_plan'], data['runs']
    if not plan or not runs:
        raise ValueError('A prespecified validation plan and completed runs are required')
    flags = ('validation_plan_frozen_before_development',
             'test_glm_separate_from_development',
             'frozen_prediction_unchanged_during_test', 'optimizer_frozen_during_test')
    for run in runs:
        if any(run.get(flag) is not True for flag in flags):
            raise ValueError('A run lacks the required freeze/isolation checks')
        if run['training_volume_count_at_freeze'] != run['training_volume_count_after_test']:
            raise ValueError('Development volumes changed after freeze')
    ids = plan['condition_ids']
    matrix = np.array([[ids.index(cid)+1 for cid in order['sequence'][:len(ids)]]
                       for order in plan['orders']])
    sns.set_theme(style='ticks', context='paper', font='DejaVu Sans', font_scale=1.1,
                  rc={'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .8})
    colors = sns.color_palette('colorblind', len(ids))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.3, 4.9),
                                    gridspec_kw={'width_ratios': [1.05, 1]})
    sns.heatmap(matrix, annot=True, fmt='d', cmap=ListedColormap(colors), cbar=False,
                linewidths=2, linecolor='white', square=True, ax=left,
                vmin=.5, vmax=len(ids)+.5,
                annot_kws={'color': 'black', 'fontsize': 12})
    left.set(xlabel='Position in a complete round', ylabel='Software order',
             title='A   Fixed orders before search')
    left.set_xticklabels(range(1,len(ids)+1)); left.set_yticklabels(range(1,len(matrix)+1), rotation=0)
    points = 0
    for index, cid in enumerate(ids):
        x, y = [], []
        for run in runs:
            row = next(row for row in run['profile'] if row['condition_id'] == cid)
            if row['independent_test_contrast'] is not None:
                x.append(row['predicted_mean']); y.append(row['independent_test_contrast'])
        right.scatter(x, y, s=33, color=colors[index], edgecolor='white', linewidth=.5,
                      label=str(index+1), zorder=3)
        points += len(x)
    limits = np.array([min(right.get_xlim()[0],right.get_ylim()[0]),
                       max(right.get_xlim()[1],right.get_ylim()[1])])
    right.plot(limits,limits,color='#858585',ls='--',lw=1,zorder=1)
    right.set(xlim=limits,ylim=limits,xlabel='Frozen model prediction (synthetic PSC contrast)',
              ylabel='Independent test mean (synthetic PSC contrast)',
              title='B   Predictions stay frozen during testing')
    right.set_aspect('equal',adjustable='box'); sns.despine(ax=right)
    right.legend(title='Condition ID',frameon=False,ncol=3,loc='upper left',fontsize=8,title_fontsize=8)
    fig.subplots_adjust(left=.075,right=.98,bottom=.28,top=.88,wspace=.40)
    repeats=plan['repeats_per_condition_per_order']
    fig.text(.075,.14,f'{len(runs)} software runs · {len(ids)} artificial conditions · {repeats} test blocks per condition · 0 participants',
             fontsize=10,weight='bold')
    fig.text(.075,.09,'Order balance applies across the full set. Dots are run–condition means, not independent people.',fontsize=9,color='#555555')
    fig.text(.075,.045,'Arbitrary ROI functions and task geometry; no human task ranking, algorithm superiority or physical timing claim.',
             fontsize=9,color='#555555')
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/f'independent-validation.synthetic.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    report=dict(is_synthetic=True,participants=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                runs=len(runs),conditions=len(ids),points=points,test_repeats=repeats,
                freeze_isolation_checks_passed=True,
                plotted_dispersion=None,scope='Descriptive software verification, no biological inference')
    (out/'independent-validation.synthetic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(plot(args.summary,args.out)))
