# Pruning, Distillation and Low-Precision Deployment in Industrial Surface Inspection: A Critical Review of Selected Studies

## Abstract

Industrial surface inspection reports combine graph reduction, transfer training and low-precision execution, but their claimed benefits depend on the comparison baseline and measurement conditions. This purposive critical review, supplemented by documented bounded searches and citation tracking, examines selected available studies through reference-explicit contrasts and a mechanism-based synthesis. Structural budgets and protected interfaces, teacher/student recovery targets, and numerical/runtime outcomes are analysed separately. Teacher, native-model and immediately reduced-student references can produce different conclusions from the same reported results. Primary surface-box evidence and separate segmentation/classification or assembly cases retain their different tasks; named deployments include GPU and RKNN-NPU implementations; failed engine construction is recorded separately from unreported or inapplicable performance. Module-power observations remain tied to device modes and sampling scope, rather than being pooled into a global efficiency ranking. The contribution is a focused interpretation of existing industrial evidence, not a new detector, validated coding scale or claim of first deployment-aware comparison. Accessibility-based sampling, targeted but incomplete coverage checks and pending independent adjudication limit generalisation. Future studies should preserve graph budgets, transfer interfaces, calibration, explicit references and complete execution boundaries.

# 1. Introduction

Industrial surface inspection requires a detector to retain spatial evidence for small or weakly contrasted defects while meeting the resource constraints of the inspection system. These requirements extend beyond aggregate detection accuracy: image acquisition, prediction, postprocessing and communication may occupy different parts of the available response time. Industrial reviews already distinguish recognition tasks and discuss the implications of evaluation protocols and deployment conditions \cite{R03,R21}. Model compression is therefore useful only in relation to the task, baseline and executable configuration for which its outcome is measured.

Pruning, distillation and precision reduction intervene at different points of the model-development process. Pruning changes a graph's retained structure; distillation changes how a student is trained and may recover accuracy without changing that student's inference graph; reduced-precision execution changes a numerical configuration and its runtime requirements. Industrial reports combine these decisions in different orders \cite{P01,P03,P04,P10,N0295}. A final-model comparison alone can conceal whether resource reduction came from the architecture, pruning or teacher selection, and whether the reported accuracy change came from recovery training.

Existing reviews already provide deployment-aware and protocol-aware perspectives. Steel surveys discuss model efficiency and industrial evaluation settings \cite{R20,R21}; Noé and colleagues use case-level matrices, conservative missing-data handling and a traceable framework \cite{R22}; the PCB computing review covers algorithms, platforms and accelerators \cite{R23}. The present article adopts those concerns rather than claiming to originate traceability or deployment-aware comparison. Its contribution is a focused critical synthesis of how reported pruning, transfer training and low-precision execution should be interpreted in selected industrial studies.

The analysis makes three reference questions explicit: comparison with a native architecture, comparison with a larger teacher, and comparison with the immediately reduced or unguided student. Table 1 reconstructs a prior-informed case summary and then requires these references separately. This reconstruction is an authors illustration, not a reimplementation of the R22 figure-of-merit calculation or an allegation that that review misinterpreted a study. Prior templates can accommodate the same information; making the reference compulsory improves the precision of the present synthesis without establishing a new theoretical framework.

Four questions guide the analysis. Which stages reduce inference resources, relative to which baseline? What evidence supports accuracy recovery independently of structural reduction? Which reported outcomes demonstrate execution on a named edge device, rather than workstation feasibility or future intention? Finally, which task, data, training and timing conditions permit a numerical contrast, and which require qualitative treatment or temporary withholding? Supervised bounding-box detection forms the primary track. Segmentation supplies a separate task track, while architecture-only, same-architecture training and classification reports provide explicitly labelled context. The review answers these questions through reference-explicit contrasts, mechanism comparisons and field-level evidence limits. It does not claim that its selected studies represent the full research population.

### Table 1. Case summaries and explicit reference-dependent questions

| Documented prior-review lens | Worked information retained here | What can be concluded |
|---|---|---|
| R22 case-level task/dataset/model/metric/hardware summary \cite{R22} | N0295 final student:17.56 M, AP50 82.1, FPS57.3, RTX3090 \cite{N0295} | Describes a configuration; an improvement still requires an explicit reference |
| Prior-informed summary with teacher reference | Teacher 67.64 M/86.4 AP50/34.7 FPS; student 17.56 M/82.1/57.3 | Smaller and faster than teacher; reported AP50 lower |
| Explicit native-reference contrast | R18 19.88 M/75.8 AP50/66.2 FPS versus final student | Smaller and higher AP50, but slower than native |
| Explicit fixed-student training contrast | Student pre/post KD80.3/82.1 AP50;17.56 M/57.3 FPS in both | Reported accuracy recovery; no KD-only removal or speed change |

Source and limits: Authors worked reconstruction informed by R22 methods/Section 6 and N0295 Table 5, p. 15. This is not an exact reimplementation of the R22 FoM or a claim that R22 evaluated N0295 incorrectly. Prior frameworks can accommodate these references; the distinction is an explicitly required contrast record, not proof of a new theory or global novelty. AP/FPS are source reports with split/timing/dispersion limits retained.

# 2. Review Approach, Sampling and Evidence Coding

## 2.1. Purpose and search history

This is a purposive critical review of reported pruning, distillation and low-precision deployment, not a completed systematic review. The sampling purpose is to contrast structural budgets, recovery targets, numerical/runtime changes and differing deployment evidence. Complete current-project texts form the initial accessible pool. The initial analysis uses all 23 original reports in that pool for task, mechanism, publication and outcome assessment; it does not choose only studies with favorable accuracy or timing. This purpose was formalised during the present source-based reanalysis, not prospectively preregistered before the original retrieval.

The recorded public-channel search round took place on 13 October 2026, using a 2021–2026 date window ending on that date. OpenAlex, EuropePMC and arXiv queries, earlier seeds and supplemental discoveries have separate records; Crossref supports metadata checking. Prepared WoS, Scopus and IEEE Xplore searches were not completed. Exact queries and partial/failed states remain in Appendix A. These sources and their search semantics are not treated as equivalent, and query hit counts are not converted into a PRISMA selection flow.

## 2.2. Selection purpose and stopping boundary

Cases are sought for mechanism variation, manufacturing-material variation, native/teacher/student reference differences, and measured or failed device execution. The available pool supplies BN-/filter-based pruning, masked or task-specific transfer, TensorRT/RKNN precision configurations, steel/metal/PCB/fabric/belt/magnetic-tile tasks, and both favorable and unfavorable outcomes. The initial analysis froze after source-located assessment of the available pool and a targeted access check. A supplementary search and source-appraisal round was subsequently completed on 14 October 2026. It is a dated analysis freeze, not evidence of theoretical saturation or adequate population coverage.

The targeted check prioritised seven unresolved leads for adaptive compression, wood/embedded execution, group-convolution pruning, anchor-free or distribution-corrected KD, FPGA and hybrid precision. They were selected from recorded titles/metadata and publicly accessible source links. No new eligible full text was obtained in that initial seven-lead access check; failed requests and landing-only pages remain pending. Their abstracts supply no numerical or method conclusions in this synthesis. The coverage log makes these blind spots visible and constrains the article to selected-study interpretation.

## 2.3. Independent study descriptors and outcome permissions

The initial 23 original reports comprise 19 box-detection,2 segmentation and 2 classification sources. Task, reported mechanism, publication/version status and outcome permission are independent descriptors. Sixteen box-detection reports describe a compression operation under the present rule; this count includes uncertain source outcomes rather than certifying sixteen valid deployment studies. N0251 and N0300 retain teacher/student mechanism eligibility despite their different publication or measurement limitations. N0249/N0268 are registered as method reports, while their conflicting numeric outcomes are withheld rather than making uncertainty a prediction-task category.

Segmentation outcomes remain separate from box AP, and classification is contextual. Native architecture-only or same-size training results are not converted into evidence of structural compression. Unavailable fulltexts and unresolved candidates are not academically excluded because access failed. Table 2 and Appendix B disclose descriptor and field permissions for every available source; the counts belong to this accessible pool, not final systematic-search inclusion.

## 2.4. Data model and contrast definitions

A configuration identifies a named graph/training/numerical/runtime/device state. A configuration group is an organisational view, not an independent experiment. Relationships record documented structural transitions or same-graph training contrasts. A teacher is a supervisory reference and not automatically a checkpoint predecessor of the student. Root references, graph/training predecessors and checkpoint continuity have different fields; unknown checkpoint continuity stays unknown.

A contrast is the only object assigned comparison eligibility. It records from/to configurations, metric, reference type, changed factors, source locator and restrictions. REC denotes a documented fixed-student/task training or recovery contrast; it does not certify causal control or low bias. CFG denotes a full-configuration or resource contrast. A configuration without an eligible counterpart remains an OBS observation; conflicting fields are HELD rather than silently selecting one reported row. No comparison grade is assigned to an entire configuration or publication.

Configuration, relationship, measurement and contrast records are cross-linked in Appendix C. Mechanical checks validate unique IDs, valid endpoints, same-graph/reference logic and registered metric labels. A separate constraint check verifies fixed-graph recovery categories against documented source pairs and graph identifiers. Both checks share the source annotations. This is a software consistency check, not independent human recoding or validation of source annotations.

## 2.5. Quality, measurement status and provenance

Quality domains cover data independence, metric definition, baseline and training attribution, timing, repeatability and internal consistency without a summed score. P01's test-assisted tuning limits independent accuracy interpretation; P09's AP formula and joint device/runtime/precision transition limit its standard AP and isolated quantization claims, while its RK3568 path remains informative \cite{P01,P09}. Quality therefore acts on claims, not as an interchangeable technical-role label.

Measurements distinguish reported, unreported-after-check, unverified, not-applicable, build-failed and held-conflict states. Missing power or memory extraction does not automatically become NR. Failed-engine throughput is unavailable, not zero. N0251's native/teacher values come from Table  1/2 and its distilled values from Table  2; N0126 Tables 4/6 explicitly label mAP@0.5 \cite{N0251,N0126}. Metric-specific locations, unit conversions and author/reproduced/quoted provenance are retained.

Checkpoint storage and measured module power are synthesised only under named conditions. Total runtime-peak memory is not quantitatively synthesised:allocator-specific reported values do not establish total engine memory. P11's energy data retain total module rail, device modes and sampling context rather than being interpreted as GPU-only or per-item energy \cite{P11}.

The extraction is an AI-assisted source-based workflow and has not undergone independent human adjudication or full current-publication-notice checking. No detector experiments are reproduced. Coverage and annotation limits prevent prevalence estimates, universal rankings and claims of an optimal compression sequence.

### Table 2. Independent study and outcome descriptors

| Axis | Coding rule | Cases showing why axes differ |
|---|---|---|
| Prediction task | box detection / segmentation / classification | P11/P12 segmentation; N0573/N0717 classification; other reports box detection |
| Reported technical operation | pruning/teacher-student or recovery/integerexecution; architecture-only context | N0251/N0300 retain reported teacher/student eligibility; native baseline choice still matters |
| Publication/identity | formal publication identity/preprint; review/currentnotice status separately | N0246 preprint; N0251 peer-review status unverified; DOI identity does not validate experiments |
| Outcome permission | specific metric/contrast may be reported, unverified or held | P09 AP held but NPU path retained; N0249/N0268 method reports registered with numeric outcomes held |

Source and limits: Authors inventory of 23 available original reports. Tasks:19 box-detection,2 segmentation,2 classification. Sixteen box-detection reports describe a compression operation under the present rule; this includes supplementary/uncertain outcomes, not 16 validated deployment studies. No whole-study role is an evidence grade; no counts imply systematic-search inclusion or population prevalence.

## 2.6. Supplementary discovery, tracking and scope control

A supplementary round on 14 October 2026 used 16 OpenAlex title-and-abstract queries covering pruning, distillation, low precision, device families and reviews. Eleven Crossref relevance-ranked discovery queries were attempted; four returned HTTP 429. Each query retrieved a bounded first page, rather than every indexed match. Eight previously appraised sources seeded backward reference-list and forward citation checks, yielding 168 deposited references and 26 indexed citing records. These are occurrence/relationship counts, not included-study counts; deposited lists and indexed citing records can be incomplete. Exact queries, times, returned counts, errors and snapshots are retained in Supplement E.

The combined API snapshot contains 1052 distinct discovery records after DOI/title-based deduplication. It remains a discovery pool: rule-assisted triage is not a claim that every abstract was individually reviewed. Twenty-six priority leads were appraised for their ability to fill mechanism, material, device or task-boundary gaps. Five new openly licensed PDFs and two online full-text bodies supported selective methods/results appraisal. Recovered copies of already appraised reports were not counted as new studies. Full-text access failure remains a retrieval state, not an academic exclusion.

Two additional sources inform the primary surface-box compression analysis: CTL-YOLO and DCKD \cite{SX07,N0144}. Solder-Yolo is a separate segmentation source \cite{S0189}. Missing-component FPGA inspection, cigarette-filter classification, descriptor-based classification and reverse-distillation anomaly detection are explicitly adjacent or excluded-core cases \cite{S0472,S0885,S0531,S0117}. They clarify scope and execution interpretation without being pooled with box AP or counted as four additional compressed surface-box detectors. The priority-screening ledger retains source locations and reasons. This bounded supplement does not establish field-level coverage, saturation, independent dual screening or a completed PRISMA process.

# 3. A Layered Taxonomy of Compression Decisions

## 3.1. Structural transformation and defect information

Structural decisions specify which channels, filters, layers or architectural paths remain in the inference graph. The inspected pruning pipelines differ in their importance signals and protected components. P01 sparsifies batch-normalization scaling factors before channel/layer removal; P03 ranks filters by weight L1 norm and retains prediction and feature-reconstruction components; N0064 uses an L2-based channel criterion; N0141 combines an enhanced architecture with Slim pruning \cite{P01,P03,N0064,N0141}. These operations should not be grouped as an identical L1 sparse-training procedure. In particular, L1-norm ranking and regularising BN scales intervene at different points of the pruning process.

The protected structure matters for defect localisation. P03 retains terminal prediction, DFL/FASFF and selected reconstruction components because the authors aim to preserve output dimensions and tiny-defect representations \cite{P03}. This rationale describes the implemented pruning constraints; the review does not assume that each protected component's individual benefit has been independently isolated. The geometric-median segmentation report concerns a different task and counts operations using a small dummy input, which cannot substitute for the computation of a full-resolution segmentation deployment \cite{P12}. Table 3 separates the decision layers; Table 7 compares the structural and recovery choices that cannot be represented by a common pruning-rate label.

Pruning before target-task training changes the baseline chain. RDDPA uses ImageNet-pretrained BN information to obtain a compressed structure, adjusts the neck and trains with a strategy that varies across compression settings \cite{P04}. Thus, its target-task from-scratch procedure does not imply the absence of pretraining information or a constant training budget. Conversely, P01 and N0126 apply pruning to a task model and then recover accuracy through teacher guidance \cite{P01,N0126}. These are different pipeline orders, not interchangeable demonstrations that one pruning sequence is universally best.

## 3.2. Training transfer and accuracy recovery

The transfer layer describes how a student or reduced graph is trained. P03 combines intermediate-feature matching with classification and localisation distillation adapted to its detector outputs; N0295 uses teacher-driven dynamic masks with local feature and global channel-related supervision \cite{P03,N0295}. N0545 separates foreground/background and scale-related feature guidance with attention and global terms \cite{N0545}. These mechanisms target different representations of defect evidence. Their comparison requires the student graph, feature alignment, loss weights and recovery schedule, not just the label knowledge distillation.

Compression, recovery and same-architecture supervision remain separate roles. A student can be smaller than a teacher before distillation starts, and distillation can improve its accuracy without deleting additional parameters. N0295 reports identical student parameter and FPS values before and after KD, while N0545 compares a compact student's accuracy with and without guidance \cite{N0295,N0545}. SD-GASNet's pretrained teacher has the same architecture as its student, making the transfer a training enhancement rather than a structural-reduction stage \cite{P08}. This separation prevents teacher-to-student size differences from being attributed to the KD loss alone.

Different transfer objectives do not improve every metric. MBDANet's reported PCB ablation includes distillation losses below the no-distillation result, and SD-GASNet's best overall AP objective is not its best small-object AP objective \cite{P06,P08}. A transfer analysis must therefore retain unsuccessful objectives and class/scale-level tradeoffs. Anomaly teacher–student discrepancy belongs to a different prediction mechanism and cannot automatically be treated as compression into a student-only inference graph \cite{R07}.

## 3.3. Numerical representation

Numerical representation concerns the precision and calibration of the executable graph. P10 evaluates the same final detector through TensorRT FP32, FP16 and INT8 configurations, but leaves the calibration method and some runtime conditions unreported \cite{P10}. P09 describes RKNN asymmetric INT8 conversion using channel/MMSE settings, although calibration-set independence and tool versions are not established \cite{P09}. Both contribute integer-execution information; neither permits an evidence-based ranking of PTQ against QAT. QAT is therefore retained as a reporting distinction and research question rather than represented as a mechanism comprehensively compared by this corpus.

The segmentation parallel track illustrates why calibration and target execution require separate checks. P11 describes calibration/evaluation separation, reports substantial mask-AP loss for one INT8 configuration and observes Jetson engine-build failure under TensorRT 10.3 \cite{P11}. Its suggested FP16 mask-prototype branch is not an evaluated mixed-precision solution. FP16 classification checkpoint reduction in N0717 also cannot establish integer calibration quality or detector accuracy preservation \cite{N0717}. The layer is consequently indexed by actual numerical representation, calibration evidence and prediction task.

## 3.4. Execution implementation and contextual architecture

The execution layer joins the graph to a compiler/runtime and device. P01 provides AGX Xavier throughput; P02 reports TensorRT-FP16 execution on TX2; P09 supplies an RKNN/RKNPU2 path on RK3568; P11 evaluates TensorRT engines on two Orin devices \cite{P01,P02,P09,P11}. These are distinct implementation configurations. A valid deployment comparison must preserve which graph was built, whether execution succeeded and what part of processing was measured. Runtime deployment is an evidence axis, not another form of structural pruning.

Native lightweight designs are retained as context for baseline selection. P02 modifies blocks, fusion and attention without an evaluated pruning/KD stage; P05 and P08 enhance accuracy with larger resources relative to selected native or small-ablation baselines \cite{P02,P05,P08}. Their inclusion does not make every lightweight-labelled architecture a compression study. The layered taxonomy assigns structural transformation, transfer, numerical representation and execution separate fields, with architecture, task and data forming their context. Figure 1 visualises this organisation; the configuration/measurement/contrast records apply these distinctions without assigning a contrast grade to a stage.

## 3.5. Mechanism-level implications across the selected reports

The pruning families require different budget interpretations. A BN-sparsity workflow first changes the learned channel-scale distribution, whereas weight-norm filtering ranks a graph's filters without implying the same sparse-training step \cite{P01,P03}. N0064's layer-wise 75-percent channel target, P03's FLOPs-based target and RDDPA's reported compression ratios do not denote the same deleted structure \cite{N0064,P03,P04}. Consequently, a common percentage cannot by itself establish equivalent student capacity or recovery difficulty. Table 7 retains each source's unit of reduction and graph constraints.

Protected prediction interfaces and recovery budgets are coupled decisions. P03 preserves terminal prediction/DFL/FASFF and selected reconstruction components rather than pruning every convolution; RDDPA adjusts its neck and channel shape before training, and changes training strategy with compression \cite{P03,P04}. These designs preserve different task interfaces or alter different learning budgets. Their source results cannot identify a universally best importance criterion. In particular, better recovered AP after a combinedpipeline does not prove that the pruning ranking alone was superior.

Transfer targets differ in the information they select. P03 combines compatible-layer feature matching with classification/localisation guidance; N0126 uses masked feature alignment; N0295 uses teacher activation to weight local/global guidance; N0545 separates foreground/background and attention/scale terms \cite{P03,N0126,N0295,N0545}. Table 8 makes their alignment and teacher conditions explicit. These differences explain which evidence is needed to interpret a loss, not which loss must win across materials. A mask's construction and a teacher's defect representation are part of the transfer setting; neither improved AP nor a heatmap proves that an alternative teacher or feature interface will behave similarly.

Recovery also has an operational limit. Fixed-student transfer can change reported accuracy while leaving graph size and tabled FPS unchanged, as in N0295; defect-sensitivity weighting can change the balance between defects and false calls, as in N0064 \cite{N0295,N0064}. Thus, the relevant conclusion is conditional accuracy recovery under a budget and risk criterion, not KD-induced parameter deletion or universally preserved defect recall. Inference-time teacher/adaptor removal must be established by the evaluated graph rather than assumed from the method name.

Low precision adds two distinct failure surfaces: altered task outputs and inability to construct or execute an engine. P10 supplies workstation precision outcomes; P09 changes representation together with RKNN/NPU hardware; P11 has a workstation mask-accuracy loss and a failedJetsonINT8 build \cite{P10,P09,P11}. Calibration and graph/runtime evidence are therefore needed separately. The proposedFP16 mask branch remains untested, and incomplete calibration descriptions do not allow attribution of the observed loss to a particular quantization cause or a ranking of QAT against PTQ.

### Table 3. Compression decisions and the evidence needed to interpret them

| Decision layer | Implemented choice | Inference-relevant boundary |
|---|---|---|
| Structure \cite{P01,P03,P04,N0064,N0141} | BNsparsity, filter-norm/channel pruning, protected outputs, pretraining-guided structure | Recordremovalunit, graphdependencies and protectedcomponents; cannotcompare identical percentages with different definitions |
| Transfer \cite{P03,N0295,N0545,N0126} | feature/head losses, foreground/background or teacher-activation masks | Separatestudentgraph reduction from recoverytraining; alignment and teacher supervision may not transfer unchanged |
| Representation \cite{P10,P09,P11} | TRT/RKNN lowprecision, calibration orper-channel settings | Numericalaccuracy and engineconstruction are separateoutcomes; no PTQ/QAT ranking inferred |
| Execution \cite{P01,P02,P09,P11} | Xavier/TX2/RKNN-NPU/Orin stacks | Successful run, unsupported build and measurement scope retained; not another structuralpruning criterion |

Source and limits: Authors synthesis of cited full-text methods. Native architecture/task/data are context. Decision layers are neither independent causal factors in every study nor a validated selection algorithm.

### Table 7. Structural and recovery differences within the selected reports

| Source | Importance/ordering | Budget meaning | Protected graph or recovery | Transfer limitation |
|---|---|---|---|---|
| \cite{P01} | BN-L1 sparsification; channel+layer removal | 90%channels then 2 layers; not parameter fraction | Channel/layer widths; KD from original model | Temperature and complete recovery budget U |
| \cite{P03} | Global filter weight L1 ranking | Reported 50%FLOPs pruning target | Terminal head/DFL/FASFF/reconstruction protected; BFCD-KD | Alignment-layer list/split incomplete |
| \cite{P04} | ImageNet BN-ranked compressed structure | Reportedcompression ratio; channels 16 multiples | Neck adjusted; training budget scales with pruning | Cannot isolate criterion from budget; timing scope U |
| \cite{N0064} | L2 channel importance after transfer | 75%channels; input/output excluded | Original transfer-trained model teacher; KL/positive weights | Prune choice heuristic combines AD and MB; not universal optimum |
| \cite{N0141} | BN-L1 sparse→Slim/retrain | Reported 1.5 xsetting; not measured speed | Enhanced head/backbone then channel pruning | Accuracy definition and hardware conflict; FPS NR |
| \cite{N0246} | Author-implemented LAMP channel pruning | LAMP1.5 x label; kept as reported | Compact backbone/neck/head before pruning | Strict AP column conflict; batch 1/32 separate; preprint |
| \cite{N0126} | BN slimming/retrain | 70%reported pruning rate | YOLOv5m teacher; masked feature KD | Source grouping and full recovery budget U |

Source and limits: Original author methods at the locators retained in the pruning synthesis CSV. No method is ranked; budget labels are not interchangeable. A protected component is the authors design choice, not independently proven necessary on every material. Reported graph/training changes do not identify an operator-level runtime bottleneck.

### Table 8. What the transfer loss changes and what it does not establish

| Source | Student state | Transfer object | Stage interpretation | Condition limiting reuse |
|---|---|---|---|---|
| \cite{P01} | Reduced YOLOv4 | Soft class/box teacher supervision after pruning | Recovery remains on selected graph | Test used for tuning; temperature U |
| \cite{P03} | 50%pruned SAF | Mimic feature+Sigmoid classification/CIoU localisation | Feature matching requires selected compatible layers | Layer list U; no assumed extra inference adapter |
| \cite{P06} | Compact MBDANet | BCKD classification/localisation coupling | Teacher YOLOv8s; chosen loss outperforms some alternatives | Other losses below no-KD; metric IoU U |
| \cite{N0064} | Pruned transfer-trained YOLOv5s | KL soft targets plus positive confidence weighting | Defect-sensitivity depends on false calls as well as recall | AD, not AP; heuristic budget not general risk criterion |
| \cite{N0295} | Fixed compact FFD | Teacher activation mask; local/global channels | Retains structure;5 seed AP means without dispersion | Teacher ability/alignment/weights limit transfer |
| \cite{N0545} | CompactS-model | Foreground/background/scale/attention/global terms | Separate teacher/student graph from student KD contrast | Strict metric/size units ambiguous; no KD timing pair |
| \cite{N0126} | Slim YOLOv5n | Masked feature matching with alignment | Transfers from YOLOv5m without claiming further parameter deletion | Mask/alignment and recovery budget need explicit records |

Source and limits: Authors analysis of cited method/loss descriptions, not performance ranking. Losses act on different teacher/student interfaces and tasks; positive-source results do not prove transfer to another graph/material. A train-time adapter is not presumed absent from the deployed graph without an export description.

## 3.6. Transfer supervision and the meaning of a compressed student

DCKD corrects erroneous teacher outputs using labelled targets and reuses hard samples, providing a different recovery mechanism from feature imitation. Its methods specify longer distillation training than the undistilled baseline, so improved AP cannot be attributed solely to the correction loss \cite{N0144}. Solder-Yolo uses an intermediate teacher and additional attention within a separately evaluated segmentation pipeline; its ablations contain different directions for precision and stricter mask AP \cite{S0189}. These reports motivate explicit recording of supervision, training budget and metric choice rather than a universal preference for a larger teacher.

Reverse distillation has a different inference purpose: the appraised two-teacher anomaly method calculates teacher--student discrepancies at test time \cite{S0117}. A smaller student therefore does not establish teacher-free inference or a smaller deployed system. Similarly, selecting handcrafted descriptors for an SVM on Raspberry Pi is not pruning a deep detector \cite{S0531}. Both cases are retained only to clarify scope.

# 4. Stage-Level and Deployment Analysis

## 4.1. Baseline-sensitive attribution: a complete pruning–recovery case

P03 provides a complete illustration of baseline-sensitive conclusions. On the reported GC10-DET evaluation, native YOLOv8n has 3.0 million parameters and AP50 of 67.2 percent; the enhanced SAF teacher has 4.1 million parameters and AP50 of 70.8 percent. The selected pruned model has 2.4 million parameters and AP50 of 67.5 percent, and BFCD-KD subsequently reports 70.6 percent at the same 2.4 million parameters \cite{P03}. Table 4 retains these separate stages rather than collapsing them into a final-versus-baseline gain.

Three questions now receive different answers. Against the enhanced teacher, the final student reduces resources while retaining slightly lower AP50. Against native YOLOv8n, the final student is smaller and has higher reported AP50, but this contrast includes the architectural, pruning and training changes. Against the immediately pruned student, KD improves reported AP50 while leaving the tabulated parameters and operation count unchanged \cite{P03}. The review's configuration/contrast register makes these attribution boundaries explicit. GC10 train/test partition details and layer-alignment implementation remain insufficiently specified, and no edge latency is available; consequently, the case supports reported resource/accuracy recovery rather than a demonstrated hardware speedup.

## 4.2. Teacher versus native baseline in compact-student distillation

N0295 shows a different baseline dependence in printed-fabric detection. Its teacher reports 67.64 million parameters and 34.7 FPS; the compact student reports 17.56 million parameters and 57.3 FPS. The native R18 baseline, however, reports 19.88 million parameters and 66.2 FPS \cite{N0295}. The student is faster than the teacher but slower than the native baseline. Calling that result simply a distillation speedup would conceal the reference model.

The fixed student reports AP50 of 80.3 percent before KD and 82.1 percent after KD, with AP50:95 changing from 57.6 to 60.3 percent; its parameter count and FPS remain unchanged \cite{N0295}. This is eligible as an REC training/recovery contrast, not evidence that KD alone removed parameters or accelerated inference. The paper reports AP means over five random-seed runs but not dispersion; the FPS aggregation is not established, and its augmentation-source isolation and timing boundary remain unclear. The reported pattern can be retained without a statistical-significance claim or a production-line generalisation.

N0545 adds foreground/background/attention transfer evidence for a compact industrial student. The aluminium student has higher reported AP with KD than without it, while the source's teacher/student FPS comparison uses different graphs \cite{N0545}. It does not supply a before/after-KD timing contrast. Its model-size unit and strict-IoU step description are unresolved, and the bottle-bottom coding experiment is not a surface-defect evaluation. Only task-labelled method and outcome fields are therefore retained; neither ambiguous size values nor bottle-code scores enter defect-compression comparisons.

## 4.3. Resource reduction, recovery and defect-risk tradeoffs

The conveyor-belt study supplies explicit mAP50 stage labels: the unpruned model reports 95.57 percent, the 70-percent slim model 90.74 percent, and the distilled slim model 94.83 percent at 0.49 million parameters \cite{N0126}. This supports partial accuracy recovery relative to the pruning stage. It does not show that the final student exceeds the unpruned distilled model, and neither data-source isolation nor full latency boundaries are established. Confirming the metric label resolves one extraction uncertainty without resolving those experimental limitations.

For magnetic tiles, N0064 reports a final 0.460-million-parameter model against a 7.082-million-parameter YOLOv5s comparator, but average inference time changes only from 10.857 to 9.46 ms \cite{N0064}. These are CFG combined-pipeline observations; timing scope and the reported GPU designation remain unresolved. The same report's defect-sensitivity objective also produces false calls among defect-free images. Its aggregate AD measure and defect-class recall must not be translated into a guarantee of zero missed defects and zero false alarms.

P04 reports increasing GTX960M throughput across compression settings, but varies its training strategy and does not define a complete timing scope \cite{P04}. N0246 separately reports a slight batch-one throughput decrease and a batch-thirty-two increase after its selected LAMP stage \cite{N0246}. P06 reports lower parameter/operation counts with lower final throughput under its stated batch-four setting \cite{P06}. Together these configuration contrasts show why resource measures and measured speed remain distinct outcomes. They do not identify the responsible operator or estimate how prevalent the discrepancy is across the field.

N0141 is retained for its pruning mechanism and resource fields: its enhanced NEU model reports 6.87 million parameters and 14.9 GFLOPs before Slim pruning, and 3.51 million parameters and 9.95 GFLOPs afterward \cite{N0141}. Its accuracy definition contains an unrelated single-category meter statement, and no actual FPS values substantiate the stated acceleration setting. The resource transformation can be described, but its reported mAP must not silently become standard AP50 and its 1.5-fold setting must not become a measured device speedup.

## 4.4. Actual edge configurations and failed execution

Table 5 separates observations, full-runtime contrasts and failed construction. The power values are module measurements under the stated conditions, not an additional model-compression experiment. P01 reports 47.8 FPS for the distilled detector on AGX Xavier, but does not supply a matched native-model Xavier result or complete timing boundary \cite{P01}. Moreover, its test set is explicitly described as serving hyperparameter tuning, so independent accuracy validation is limited. P02 reports 10.1 and 18.9 FPS for its baseline and MicroDETR on TX2 FP16; its workstation comparison has a different speed direction, and other timing statements remain inconsistent \cite{P02}. These observations support particular deployments rather than a device-independent lightweight advantage.

P09 documents an actual RK3568 NPU path using RKNN conversion and runtime components \cite{P09}. This widens the execution evidence beyond TensorRT/GPU cases. Its desktop-FP32 and RKNN-INT8 configurations simultaneously change hardware, runtime and numerical representation, while accuracy, size and input/count statements require reconciliation. The present synthesis therefore retains successful named-device execution as OBS evidence and withholds a quantization-only speedup or standard mAP preservation claim. The absence of such a matched contrast is not the absence of an NPU implementation.

In the segmentation parallel track, P11 reports pretrained nano-model TensorRT-FP16 throughput of 146.7 FPS on AGX Orin and 89.9 FPS on Orin Nano under specified device modes, at batch one and input 640 \cite{P11}. The measured processing scope includes pre/forward/post processing and host transfers but excludes camera acquisition, buffering and controller communication. Jetson INT8 engine construction fails under TensorRT 10.3, and its FP16 mask-branch workaround is untested. These entries represent actual processing configurations and a version-specific failed build, not a full acquisition-to-action performance claim.

## 4.5. Field-specific quality and comparison boundaries

Table 6 attaches quality limitations to the conclusions they affect. Test-assisted tuning in P01 limits independent accuracy claims without erasing the pruning graph; metric inconsistency in P09 and N0141 limits standard-AP comparisons without erasing their named operations; missing device timing in P03, N0141 and N0295 limits deployment conclusions \cite{P01,P09,N0141,P03,N0295}. This claim-specific treatment is preferable to either accepting every field because a DOI exists or discarding an entire report because one outcome is unresolved.

The tasks and evaluation populations also remain attached to the configuration. P11 evaluates a cleaned 146-image steel subset with calibration separation; P10 uses a public/private mixed PCB dataset with a reported same-PCB grouping policy but no independent test set; N0717 changes the original MVTec AD partitions into supervised classification splits \cite{P11,P10,N0717}. Their mask AP, box AP and classification F1 answer different questions. An unclear augmented-source split is a reporting risk rather than proof of leakage, and a stated isolation rule is not independent verification of its implementation.

P10's FP32/FP16/INT8 table retains a reported accuracy–latency tradeoff, while P11's workstation INT8 mask-AP loss and failed Jetson build demonstrate different numerical and execution outcomes \cite{P10,P11}. The workstation accuracy/latency comparisons are CFG contrasts; the failed build is a separate execution observation. Runtime and calibration restrictions remain attached to each outcome. N0251 is used only as a flagged supplementary case: its distilled student matches the reported native student's AP50, parameters and forward-only latency, rather than exceeding them \cite{N0251}. The baseline strict-IoU metric is unreported, so equality of AP50 does not establish equality of every accuracy metric. Its independent test and peer-review status remain unresolved; the result remains supplementary.

Classification NAS/KD latency in N0573, FP16 checkpoint reduction in N0717, native-architecture enhancements in P05/P08 and budget-sensitive transfer in N0300 remain contextual tracks \cite{N0573,N0717,P05,P08,N0300}. They help identify a changing baseline or resource definition but are not pooled into box-detector performance. In particular, N0300's contradictory native-resolution/640 descriptions and timing attribution are held outside numerical cost synthesis. Across all tracks, the review compares stage roles and evidence eligibility; it does not normalize heterogeneous FPS by FLOPs, model names or nominal device throughput.

### Table 4. Baseline-sensitive stage examples

| Source | Model/stage | Parameters(M) | Box AP50(percent) | FPS | Locator |
|---|---|---|---|---|---|
| \cite{P03} | YOLOv8n | 3.0 | 67.2 | NR | Tables 2/3 pp. 8/10 |
| \cite{P03} | SAF-YOLOv8n | 4.1 | 70.8 | NR | Tables 2/3 pp. 8/10 |
| \cite{P03} | pruned SAF | 2.4 | 67.5 | NR | Tables 2/3 pp. 8/10 |
| \cite{P03} | BFCD-KD | 2.4 | 70.6 | NR | Tables 2/3 pp. 8/10 |
| \cite{N0295} | RT-DETR-R18 | 19.88 | 75.8 | 66.2 | Table 5 p. 15; seed procedure p. 11 |
| \cite{N0295} | RT-DETR-X teacher | 67.64 | 86.4 | 34.7 | Table 5 p. 15; seed procedure p. 11 |
| \cite{N0295} | FFD pre-KD | 17.56 | 80.3 | 57.3 | Table 5 p. 15; seed procedure p. 11 |
| \cite{N0295} | FFD post-KD | 17.56 | 82.1 | 57.3 | Table 5 p. 15; seed procedure p. 11 |

Source and limits: Original author reports: \cite{P03}, Tables 2–3, pp. 8/10; \cite{N0295}, Table 5, p. 15. P03 uses GC10-DET/640/RTX4090; split details and inference timing NR. N0295 uses textile 2318→3528/640/RTX3090; AP means over five seeds; dispersion NR; FPS aggregation U; inference batch/timing scope NR. Rows compare stages within a source, not across the two studies. Resources unchanged in a KD contrast are not KD-induced removal. NR is checked unreported information; no significance or universal speed ranking is inferred.

### Table 5. Device observations, runtime contrasts and measured power scope

| Source/configuration | Reported FPS | Module power(W) | Interpretation scope |
|---|---|---|---|
| P01 AGXXavier distilled \cite{P01} | 47.8 | NR | OBS; native-device counterpart absent; test tuningrisk |
| P02 TX2 baseline / MicroDETR FP16 \cite{P02} | 10.1 / 18.9 | U | CFG; architecture/configuration contrast; timingscopeunclear |
| P09 RK3568/RKNN INT8 \cite{P09} | notinferred | U | OBSexecution; APconflict; no isolated quantizationgain |
| P11 AGXOrin PyTorch FP32 / TRT FP16 \cite{P11} | 37.0 / 146.7 | 5.21 / 5.03 | CFG; pretrained nano, batch 1,640; MAXN |
| P11 OrinNano PyTorch FP32 / TRT FP16 \cite{P11} | 26.5 / 89.9 | 8.48 / 7.87 | CFG; pretrained nano, batch 1,640; MAXN_SUPER |
| P11 bothJetsons Segment 26 INT8 \cite{P11} | unavailable | notassigned | BUILD_FAILEDunderTRT10.3; not 0 FPS; workarounduntested |

Source and limits: Original reports: P01 Table 6 p.23; P02 Table 11 p. 15; P09§3.3/Table 8; P11 Table 10/§4.4/§6.2. P11 power is total module power sampled at 500 ms under stated modes, not GPU-only power or per-item energy. PyTorch→TensorRT also changesprecision; no isolatedprecisioneffect. Processing excludescamera/buffering/control. No rowwise boardranking, no FPS-to-latency inference. Runtime-peak memory is not synthesised here: allocator-only values are not total engine memory. U=notextracted/unclear, notNR.

### Table 6. How source-quality limitations change permitted claims

| Source | Observed limitation | Retain | Withhold or qualify |
|---|---|---|---|
| \cite{P01} | Test set explicitly used for hyperparameter tuning | Pruning/recovery method and author-reported device execution | Independent-test accuracy validation |
| \cite{P09} | Nonstandard AP formula; inconsistent size/accuracy; joint transition | RKNN/RK3568 execution path | Standard detectorAP or quantization-only speedup |
| \cite{N0141} | Unrelated meter metric definition; no measured FPS | Implemented Slim/resource reduction | StandardAP50 conversion or 1.5 x measured acceleration |
| \cite{P03} | Split/timing absent; layer alignment U | Labelled resource/recovery stages | Embedded speed or robust cross-material benefit |
| \cite{N0295} | 5-seed means without dispersion; timing/group isolation unclear | Fixed-student reported AP changes | Statistical significance or KD-only FPS improvement |
| \cite{N0545} | Size unit/strict-IoU step unresolved; extra coding task | Task-labelled KD methods and source results | AmbiguousM as parameter count; bottle-code defect performance |
| \cite{N0126} | mAP50 label confirmed; grouping/budget/timing unclear | Stage mAP50/resources | Declare metric unspecified or infer independent production validity |

Source and limits: Authors field-specific appraisal against full-text locators in phase 8_frozen_study_roles_and_quality.csv; provisional human-review status. No summed risk score. Unclear reporting is not proof of leakage, fabrication or invalidity of every field. Full corpus quality domains, including contextual/held reports, are in Appendix B.

## 4.6. Additional counterexamples and execution boundaries

CTL-YOLO's reported pruning-budget series preserves AP50 at its selected 4.0-times setting but drops to 70.2 and 66.1 percent at the more aggressive settings. Its separate ablation table retains AP50 while reducing stricter AP and FPS after pruning. The pre-pruning parameter counts conflict between Tables 4 and 7; no compression ratio is reconstructed by choosing one value \cite{SX07}. These source-specific observations support a conditional budget trade-off, not a hardware-independent pruning threshold.

The FPGA missing-component study documents a PYNQ-Z2 execution chain but reports slower processing than its workstation. Its approximately 10-W figure is explicitly drawn from board documentation, not a measured model power trace \cite{S0472}. The cigarette-filter study explicitly performs classification without boxes. Its DPU inference time and file-to-result pipeline delay are different quantities, and dual buffering changes the latter without isolating quantization's effect \cite{S0885}. These adjacent cases strengthen timing interpretation without expanding the primary prediction task.

Solder-Yolo's inference platform is a GPU-free Windows industrial PC; its report describes earlier embedded attempts as insufficient in speed and stability. Workstation FPS and industrial-PC timing are kept separate, while inconsistencies in the printed FLOPs-reduction percentage and AT/FPS remain unresolved \cite{S0189}. Its claimed production-line adoption is an author statement, not independent validation. Table 9 summarises the new evidence and permitted interpretations.

### Table 9. Supplementary source-appraisal roles and limitations

|Source|Task/role|Evidence added|Limit on interpretation|
|---|---|---|---|
|\cite{SX07}|Primary surface-box pruning|Budget/AP/FPS trade-off;Tables4/7|Pre-pruning parameter conflict;workstation only|
|\cite{N0144}|Primary surface-box distillation|Teacher-output correction;hard-sample reuse;§III|Different training budgets;edge timing not extracted|
|\cite{S0189}|Segmentation parallel|Intermediate teacher;Tables7/9;§4.5|IndustrialPC,not embedded-board validation;metric conflicts|
|\cite{S0472}|Adjacent assembly box|PYNQ-Z2;Tables2/3|Different defect task;10W rated;bit width U|
|\cite{S0885}|Classification context|VitisAI INT8;Tables6/8|No boxes;kernel and pipeline timing differ|
|\cite{S0531}|Excluded-core boundary|Descriptors/SVM;§3.2|Not neural-network compression|
|\cite{S0117}|Excluded-core boundary|Teacher-student test discrepancy;§3.4|Not supervised compressed-box detection|

Source and limits: Original source locations are retained in Supplement E and the supplementary evidence matrix. These seven appraisals are not seven additional core detectors. They do not justify a cross-task performance ranking or field-level coverage claim.

# 5. Research Questions, Limitations and Future Directions

## 5.1. What the stage analysis establishes

For the first question, structural reduction and the compact student graph account for the documented resource changes; the reference model determines their interpretation. P03's teacher, native model and pruned student yield different comparisons, and N0295's teacher/native distinction reverses the direction of its speed advantage \cite{P03,N0295}. For the second question, fixed-student KD can recover reported accuracy without reducing parameters or changing the tabulated FPS. The inspected recovery results support this conditional role rather than a general claim that KD itself compresses or accelerates a graph.

For the third question, named-device execution is present in the Xavier, TX2, RK3568 and Orin reports, but matched compression benefits and full production boundaries are unevenly documented \cite{P01,P02,P09,P11}. For the fourth question, the M/C/D/H register separates interpretable stage/task contrasts from complete-configuration outcomes and held fields. The remaining uncertainty is therefore structured: the corpus can identify implemented procedures and author-reported tradeoffs, while it cannot select a universal compression order, a device-independent speed champion or an optimal quantization method.

## 5.2. Testable priorities grounded in the audited cases

A first priority is baseline-aware recovery evaluation. A study should compare the fixed reduced student with and without recovery under a documented training budget, separately from native-model and teacher comparisons. P03, N0295 and N0126 provide useful stage structures but leave different split, budget or uncertainty fields incomplete \cite{P03,N0295,N0126}. The testable question is whether a recovery objective retains its reported advantage when student structure, model-selection data and training budget are held fixed. This is a proposed future evaluation, not an experiment performed for this review.

A second priority is separating representation effects from execution effects. The RKNN transition changes device/runtime/precision jointly, and the segmentation INT8 case has both workstation accuracy degradation and Jetson construction failure \cite{P09,P11}. Matched same-device, same-runtime tests and explicit calibration/branch-precision configurations would identify which transition changes accuracy or speed. The suggested FP16 mask branch should be tested as a specified graph configuration rather than treated as an established remedy. The available corpus does not settle PTQ versus QAT superiority.

A third priority connects batch-dependent computation with defect-risk criteria. The N0246 throughput directions differ at batch one and thirty-two, while the magnetic-tile sensitivity setting trades defect recall against false calls \cite{N0246,N0064}. Evaluations should report the production deadline, batch availability, class/scale outcomes and acceptable false-call/missed-defect operating point. Operator profiling may explain a particular bottleneck, but the current reports do not provide enough evidence to identify one. A lower operation count alone is not a diagnosis of lower acquisition-to-action latency.

## 5.3. Reporting and scope limitations

The reporting checklist in Appendix D follows the configuration/measurement/contrast definitions: task/data independence, named baseline, retained graph, recovery budget, calibration, export success, runtime/device, measurement boundary and result provenance. It is a proposed audit aid, not a validated industrial standard. Its value here is demonstrated by changed interpretation of documented cases, not by a new detector benchmark. Current source quality, inaccessible documents, incomplete retrieval screening and pending human author review limit the reach of those interpretations.

The comparison with prior reviews remains bounded to the inspected passages and available full texts \cite{R20,R21,R22,R23}. The review makes references and field permissions explicit for the selected corpus; it does not prove that previous reviews were unable to do so. The held reports and unresolved publication-status checks must remain visible, and future retrieval updates may alter the corpus. The appropriate contribution claim is therefore a traceable, baseline-sensitive critical synthesis under stated limits, whose usefulness is interpretation of the selected reports rather than a new validated theory.

## 5.3. Coverage-dependent interpretation

The supplement adds pruning-budget counterexamples, corrected distillation supervision and differentiated execution evidence. However, adaptive working-condition compression, wood/CPU detectors, non-YOLO transfer and matched low-precision surface-box deployment remain incompletely appraised. Indexed leads without accessible full text identify retrieval priorities rather than demonstrate their methods or quantitative outcomes. The article consequently remains a critical review of selected studies; a larger reference list alone would not establish population-level conclusions.

# 6. Conclusion

This review connects implemented compression decisions to their baselines, recovery procedures and evaluated configurations. Using reference-explicit contrasts makes several distinctions explicit: compact design and pruning reduce selected resource fields; KD can recover accuracy within a fixed student without independently reducing its graph; numerical precision and execution support determine whether a target configuration succeeds. The P03 and N0295 cases show why teacher, native-model and immediately reduced-student baselines cannot be substituted for one another \cite{P03,N0295}.

Measured edge execution is documented for several named platforms, but its support varies from an isolated device result to a version-specific failure or a confounded desktop-to-NPU transition \cite{P01,P02,P09,P11}. The review consequently retains field-specific quality limitations and compares stage roles rather than ranking heterogeneous accuracy and FPS. Its purposive accessible corpus and incomplete retrieval/adjudication prevent exhaustive or field-wide gap claims. Future evidence should preserve baseline and training budgets, independent evaluation data, calibration and complete timing boundaries. These are literature-grounded recommendations, not new experimental findings.


The supplementary search strengthens selected mechanism and counterexample coverage but retains incomplete full-text appraisal and bounded retrieval. Its findings therefore support source-specific interpretation of compression and deployment trade-offs, while field-wide prevalence, completeness and optimal-method claims remain outside the article's scope.



## Figures

![Typed evidence records](figures/figure01_typed_records.png)

**Figure 1.** Authors coding aid. Only a contrast receives REC/CFG eligibility; the diagram is not a tested selection algorithm.

![P03 reference-dependent comparisons](figures/figure02_P03_baselines.png)

**Figure 2.** P03 author-reported values, Tables 2–3, pp. 8/10 \cite{P03}; parameters in M and AP50 in percent. GC10-DET, input 640, RTX4090; split/alignment incomplete and timing NR. These values do not establish measured edge acceleration or isolated causal effects.
