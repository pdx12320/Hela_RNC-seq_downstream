# RNC Isoform Feature Analysis

用于 RNC-seq / total RNA 比较的转录本层面翻译调控分析管线。该项目读取 `results.csv + GTF + genome FASTA`，提取结构特征、调控特征，并进行同一基因内 isoform 对比分析。

## 1. 安装

```bash
cd rnc_isoform_feature_analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 可选外部工具：
> - RNA 二级结构：`RNAfold`（ViennaRNA）
> - 蛋白结构域：InterProScan / PfamScan（输出 TSV 后填入 `config.yaml`）
> - miRNA 位点：miRanda / TargetScan / RNAhybrid（输出 TSV 后填入 `config.yaml`）

## 2. 输入文件

### 必需
- `results.csv`
  - 必需列：
    - `gene_name, transcript_id, transcript_id_base, M_mean, R_mean, log2FC, IF_Total, IF_Ribo, Delta_IF`
  - 若无 `transcript_id_base`，程序会自动从 `transcript_id` 去版本号生成。
- `annotation.gtf`
  - 建议 Ensembl/GENCODE GTF。
- `genome.fa`
  - 与注释版本匹配。

### 可选
- `canonical_transcripts.tsv`
  - 列：`gene_name`, `transcript_id`
- `interproscan.tsv` 或 `pfam.tsv`
  - 结构域注释 TSV，自动识别常见列。
- `miranda_output.tsv` 或 `targetscan_output.tsv`
  - miRNA 预测结果 TSV，自动识别常见列。

## 3. 配置

编辑 `config.yaml`：

- `inputs.*`：输入路径
- `outputs.*`：输出目录
- `params.rnafold_bin`：RNAfold 命令名
- `params.rnafold_max_len`：超长序列截断长度
- `params.cds_start_flank`：CDS start 区域窗口半宽（默认 70nt）

## 4. 运行

### 一键运行

```bash
python run_pipeline.py --config config.yaml
```

### 分步运行

```bash
python scripts/00_prepare_annotation.py --config config.yaml
python scripts/01_extract_transcript_features.py --config config.yaml
python scripts/02_scan_sequence_motifs.py --config config.yaml
python scripts/03_predict_orf_uorf_nmd.py --config config.yaml
python scripts/04_run_rnafold.py --config config.yaml
python scripts/05_compare_isoforms_within_gene.py --config config.yaml
python scripts/06_plot_results.py --config config.yaml
```

## 5. 输出说明

### 序列文件（`output/sequences/`）
- `transcript.fa`：拼接转录本序列
- `five_utr.fa`：5'UTR 序列
- `cds.fa`：CDS 序列
- `three_utr.fa`：3'UTR 序列（可直接喂给 miRNA 工具）
- `protein.fa`：CDS 翻译蛋白序列（可直接喂给 InterProScan/Pfam）

### 表格（`output/tables/`）
- `missing_transcripts.tsv`：结果表中在 GTF 里找不到的 transcript
- `transcript_basic_features.tsv`：结构坐标、长度、表达信息
- `transcript_features_motifs.tsv`：Kozak/uORF/polyA/miRNA 汇总
- `transcript_features_orf_nmd.tsv`：ORF 对比、domain 变化、NMD 风险
- `transcript_features.tsv`：最终总表
- `isoform_pairwise_comparison.tsv`：同一基因内相对 reference 的差异
- `statistics_summary.tsv`：统计检验结果

### 图（`output/figures/`）
- Delta_IF 与 5'UTR/CDS/3'UTR 长度关系
- Delta_IF 与 uORF 数量、Kozak 强度、RNAfold MFE 关系
- 基因内 isoform Delta_IF 差异 dotplot

## 6. reference transcript 选择规则

同一 gene 内优先级：
1. `canonical_transcripts.tsv` 指定
2. CDS 最长
3. `M_mean` 最高

## 7. 特征计算摘要

- ORF 差异：CDS/蛋白是否变化、N/C 端变化、潜在移码或提前终止
- Domain 差异：可选输入的结构域 gain/loss
- uORF：5'UTR 中 ATG→同 frame stop 的完整 uORF
- Kozak：主 ORF start 周边 `-3` 和 `+4` 位点打分
- polyA signal：canonical + variant motif 扫描
- RNAfold：5'UTR / CDS 起始窗口 / 3'UTR MFE
- NMD：按 stop 到最后 exon junction 距离（>55 nt 判 high）

## 8. 稳健性与容错

- 缺少 CDS/UTR 的转录本不会中断流程，相关字段填 `NA`。
- 未安装 RNAfold：自动跳过并填 `NA`。
- 未提供 domain / miRNA 文件：继续运行并输出可供外部工具使用的 FASTA。

