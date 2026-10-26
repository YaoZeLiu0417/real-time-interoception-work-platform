# 独立测试顺序与重复响应谱

2026年10月26日。本文规定软件原型如何预先生成测试顺序、保留重复区组，并描述冻结后获得的响应谱。它没有确定人体重复次数、功效或扫描预算，也不提供优化策略优于随机采样的结论。

## 在看到结果前固定顺序

`validation.make_validation_plan(condition_ids, repeats=1, n_orders=None, seed=0)`只接受条件列表与设计参数，不接受响应结果。返回的JSON包含种子、条件表、完整顺序、轮数、每条件次数及`plan_hash`。`repeats`表示总轮数，因此`repeats=2`是每条件出现两次。

计划采用循环Latin square。先按种子打乱基础条件表，再依次循环移位，生成不同order；每个order按同一顺序重复完整轮。N个条件的完整N个orders，在每个相同序列位置上各出现一次每种条件。完整循环的整数倍保持这一性质。

这个位置平衡只成立于完整的N个orders集合。单个order、少于N个orders或非整倍数集合，不能宣称整体位置已平衡；JSON明确记录完整循环数和不完整循环的order数。每条件出现同样次数，也不等于它们在单个run内占据不同位置。循环设计没有声称平衡相邻条件的携带效应。

每个order包含`N × repeats`个区组，各条件恰有`repeats`次。N大于1时，同一order内包括轮次接缝都没有相邻重复；只有一个条件且需重复时，这一约束不可满足，计划会明确标记。不同order可分配给不同run，不能把这些软件顺序或合成run计为参与者。

例如，六个条件、两轮、六个orders意味着每个order有十二个测试区组，全部集合有七十二个区组。这个例子验证顺序属性，不是推荐七十二个真人扫描区组；正式重复次数、run分配和休息安排须由独立先导及预算确定。

```python
from experiments.interoception.validation import make_validation_plan

plan = make_validation_plan(condition_ids, repeats=2, n_orders=6, seed=51)
fixed_sequence = plan['orders'][0]['sequence']
```

现场或原生呈现程序须在开发阶段开始前保存计划及hash，再选定order。把`fixed_sequence`写入任务空间的`fixed_test_plan`后，要以新配置重算`definition_sha256`，再创建Session。这是明确更换实验定义，不能等看到开发结果后再修改顺序或假装沿用旧定义。

## 合成运行的入口与兼容性

原默认路径继续使用注册表已有的`fixed_test_plan`，保持其确切顺序和次数，不自动变成Latin设计。默认一次覆盖也不声称已经完成反平衡或重复性检验。

显式增加`--validation-repeats`会在任何开发数据产生前，生成覆盖全部合成run的循环计划，并先写出`validation_plan.prespecified.json`。`--validation-seed`只控制测试顺序；原`--seed`控制合成噪声与搜索种子。若条件有N个而只执行两个run，报告会标记位置平衡不完整。

```text
python -m experiments.interoception.simulate_neuroadaptive \
  --space /path/to/taskspace.synthetic.json \
  --out /private/new_validation_run \
  --runs 6 --budget 16 --validation-repeats 2 --validation-seed 51
```

输出目录须在公开仓库外且尚不存在。命令中的run与时长是软件验证设置，不是人体样本量；合成时间线也不证明现场扫描效率。

## 先保留逐区组记录，再汇总条件

每run的`test_readouts`是完整长表，保存每个block ID、条件、测试位置、GLM估计、可用时间和有方向的`measured_contrast`。重复条件不能先转换成以condition ID为键的字典，否则后一次会覆盖前一次。

`aggregate_test_profile(readouts, direction=1)`再将同一条件的所有有效区组汇总为一个响应谱点。输出包括：

* `independent_test_contrast`：有方向的区组beta差均值，兼容原响应谱绘图字段。
* `independent_test_sd`：两个及以上有效区组的样本标准差；只有一次时保持空值。
* `n_test_blocks`、`n_valid_test_blocks`、`n_missing_test_blocks`与全部block ID。
* 原始readout状态计数，以及有效估计的最小值和最大值。

未完成或不可估计区组不当作0。重复block ID会被拒绝，避免把同一区组的多次重估当成新重复。不同run须分别聚合。模拟器目前要求所有预定测试区组得到可估计读出后才完成；聚合函数本身仍支持明确记录缺失状态。

## 冻结与解释边界

开发完成后冻结GP及开发快照，使用新的测试GLM。测试volume不进入开发GLM，也不更新GP。报告记录冻结hash、训练volume数在测试前后的数值、测试GLM是否独立，以及冻结预测是否保持不变。

保留的`profile_spearman`和`profile_rmse`现在比较唯一条件的测试均值与冻结GP预测，不对重复条件重复加权。跨run比较先按condition ID对齐；旧`independent_run_profile_spearman`仍比较前两个run的GP预测，新增`independent_run_test_profile_spearman`比较前两个run的测试均值。它们都是合成验证的描述指标。

同一GLM中的区组beta可以相关，BOLD误差也可能有时间自相关。因此区组间SD不是独立同分布假设下的标准误或置信区间；当前报告没有计算显著性、组水平可靠性或真实内感受效应。重复合成run检验软件重启与数据隔离，不代表参与者间或同一参与者跨访视重复性。评估GP策略是否更有效，还需要另行预先规定对照策略、相同预算、噪声条件与评价端点。
