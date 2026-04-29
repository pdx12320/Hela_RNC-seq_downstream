# rnc_isoform_feature_analysis

用于 RNC-seq / total RNA 转录本层面的结构与翻译调控特征分析项目。

## 1. 安装

```bash
cd rnc_isoform_feature_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Python 3.10+。

## 2. 输入文件

必需：
- `results.csv`: 至少包含列
  - `gene_name, transcript_id, transcript_id_base, M_mean, R_mean, log2FC, IF_Total, IF_Ribo, Delta_IF`
- `annotation.gtf`: Ensembl/GENCODE GTF。
- `genome.fa`: 对应基因组 FASTA。

可选：
- `canonical_transcripts.tsv`: 每个 gene 的 canonical/MANE/APPRIS transcript。
- `interproscan.tsv` 或 `pfam.tsv`（在 `config.yaml` 里填 `domain_tsv`）。
- `miranda_output.tsv` 或 `targetscan_output.tsv`（在 `config.yaml` 里填 `mirna_tsv`）。

## 3. 运行

### 3.1 配置
编辑 `config.yaml`：
- 输入路径：`input.*`
- 输出路径：`output.*`
- 参数：`params.*`

### 3.2 一键运行

```bash
python run_pipeline.py --config config.yaml
```

支持按候选 isoform pair 的 `|ΔΔIF|` 阈值过滤：

```bash
python run_pipeline.py --config config.yaml --ddif-threshold 0.2
python run_pipeline.py --config config.yaml --ddif-threshold 0.3
```

### 3.3 分步运行

```bash
python scripts/00_prepare_annotation.py --config config.yaml
python scripts/01_extract_transcript_features.py --config config.yaml
python scripts/02_scan_sequence_motifs.py --config config.yaml
python scripts/03_predict_orf_uorf_nmd.py --config config.yaml
python scripts/04_run_rnafold.py --config config.yaml
python scripts/05_compare_isoforms_within_gene.py --config config.yaml --ddif-threshold 0.2
python scripts/06_plot_results.py --config config.yaml --ddif-threshold 0.2
```

## 4. 输出说明

### 序列文件（`output/sequences/`）
- `transcript.fa`
- `five_utr.fa`
- `cds.fa`
- `three_utr.fa`
- `protein.fa`

### 表格（`output/tables/`）
- `missing_transcripts.tsv`: 在 GTF 中找不到的 transcript。
- `transcript_features.tsv`: 最终主表（每行一个 transcript）。
- `isoform_pairwise_comparison_all.tsv`: 全量 pairwise（包含 reference vs self）。
- `isoform_pairwise_comparison.tsv`: 候选 pairwise（按 `abs(delta_Delta_IF) >= ddif_threshold` 过滤）。
- `isoform_pairwise_comparison.filtered.ddif_ge_{threshold}.tsv`: 带阈值标签的候选 pair 备份文件。
- `statistics_summary_all.tsv`: transcript-level 全量统计结果。
- `statistics_summary.tsv`: 候选 pair-level 统计结果（基于阈值过滤后）。
- `*.log`: 每一步处理日志。

### 图（`output/figures/`）
- `DeltaIF_vs_5UTR_length.png`
- `DeltaIF_vs_CDS_length.png`
- `DeltaIF_vs_3UTR_length.png`
- `DeltaIF_vs_uORF_count.png`
- `DeltaIF_vs_Kozak_strength.png`
- `DeltaIF_vs_*_MFE.png`
- `within_gene_isoform_deltaIF_dotplot.png`

## 5. 已实现特征

- 结构特征：`transcript/exon/CDS/UTR` 长度、`protein_length`、coding/noncoding。
- ORF 比较：与 gene 内 reference transcript 比较 ORF/CDS/protein/start/stop/N端/C端/frame/premature stop。
- uORF：5'UTR 内 ATG-STOP 扫描、数量、最长长度、强 Kozak uORF 数量。
- Kozak：主 ORF 起始上下文（-6 到 +4）与强弱评分。
- miRNA：输出 `three_utr.fa`；若提供结果表则汇总位点数/miRNA数/保守位点/最强结合能。
- polyA signal：canonical + variant motif 扫描与距 3' 端最近信号距离。
- RNAfold ΔG：5'UTR / CDS start window / 3'UTR（可选）。
- NMD likelihood：最后 EJC 50-55 nt 规则估计 high/low/NA。

## 6. 可选外部工具

### RNAfold (ViennaRNA)
- 若系统检测到 `RNAfold`，自动计算 MFE。
- 若未安装，不中断流程，MFE 填 `NA` 并在日志提示。

### InterProScan / PfamScan
- 项目默认生成 `protein.fa`。
- 可自行运行外部结构域注释后，将结果路径写入 `config.yaml -> input.domain_tsv`。

### miRanda / TargetScan / RNAhybrid
- 项目默认生成 `three_utr.fa`。
- 外部运行后，把结果路径写入 `config.yaml -> input.mirna_tsv`。

## 7. 稳健性与容错

- transcript ID 自动兼容带版本号与去版本号。
- 缺失 CDS/UTR 不会导致流程中断。
- 外部依赖缺失时采用 NA 回填，不终止主流程。

## 8. Candidate isoform filtering by |ΔΔIF|

- 定义：`ΔΔIF = delta_Delta_IF`（来自 `isoform_pairwise_comparison`）。
- 候选筛选条件：`abs(delta_Delta_IF) >= ddif_threshold`。
- 参数来源优先级：
  1. 命令行 `--ddif-threshold`
  2. `config.yaml -> params.ddif_threshold`
  3. 默认 `0.0`
- 筛选时会排除 `ref_transcript == query_transcript` 的 self-comparison。
- 若候选 pair 为 0，不报错：保留空候选表并在日志 warning，候选统计输出 NA/说明信息，候选绘图自动跳过。
