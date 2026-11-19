# 代码与复现

## 一分钟本地检查

Python 3.12；下面各步不需要第三方库、GPU、数据或API：

```bash
python3 examples/local_contract_demo.py
python3 tools/show_results.py
python3 tools/verify_release.py
python3 tests/test_ust_task_contract.py
python3 tests/test_ust_task_contract_independent.py
```

演示使用人为定义的排序数值；测试还覆盖完全可见的有限决策树。它们检查任务约束和答案逻辑，不复现论文分数。

## 原研究代码

`src/privacy_synth/` 保存可复用组件，`scripts/` 保存主入口及静态识别的本地Python依赖闭包。`archive/source-manifest.json`记录逐文件字节与SHA-256，`archive/entrypoints.json`标明主入口。历史支线依赖也保留，避免只留下不能理解来源的片段。

这些脚本包含固定日期目录、冻结清单、环境约束和输出存在性检查。它们是研究快照，不是对任意机器一键可运行的产品。不要批量运行 `scripts/`，尤其不要将旧授权、历史价格或历史预算当作当前授权。

## 完整实验所需本地资产

1. 合法取得数据与模型，保留原始版本与哈希。USTBench作者代码需按原研究版本恢复，第三方代码不随本仓库复制。
2. 恢复自己的本地拟合、开发、评测划分；缺失值、并列与标签语义必须验证。原Final材料不在公开仓库。
3. 恢复原运行清单与检查点，或在新的运行目录建立新实验配置。脚本中的 `runs/...` 路径指原研究归档，不是下载地址。
4. 生成与规则监督：从 `ust_full_schema_generator_20261104.py`、`ust_joint_generator_repair_20261107.py` 和缺失值样本入口阅读。现有源代码的输入/输出资格检查应保留。
5. 训练：`ust_scale_20261108.py` 固定来源题数、初始化、旋转视图、顺序和更新。`n272`历史命名不代表真实侧有272个独立题，真实侧仍为68；主展示使用匹配的n68。
6. 教师：`ust_scale_teacher_20261108.py`和补调用入口需独立准备合法的供应商、凭证、外发白名单与费用预留。此仓库不会自动授权外部调用。
7. 评测：`ust_final_matched68_20261109.py`为特定历史补测入口，不能直接用来重新访问封存集。新实验要先冻结协议与合法划分。
8. 部署：量化与vLLM入口以固定模型、题面和解码设置进行配对比较，分别保留导出/重载、准确率和性能记录。

## 环境

`environment/research-pyproject.toml`和`research-uv.lock`是历史主环境，含CTGAN本地vendor依赖，不能作为新环境直接安装的承诺。`research.gitmodules`记录相关第三方地址，恢复时遵循各自版本与许可。部署实验还使用独立的bitsandbytes、vLLM和量化工具环境；实际版本与参数以已归档代码、结果和本地运行协议为准，不要在同一个环境中盲目混装。

## 执行纪律

每个新实验使用独立目录，保存输入/代码哈希、配置、费用、日志、原始回答、检查点和完成状态。网络失败与格式失败分别记录；未完成不报告为完成。真实/合成按同预算比较，所有顺序都报告。导出的汇总必须保持数据集范围与示例条件。
