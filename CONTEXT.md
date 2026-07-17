# ETP-R1 Navigation Models

This context describes the navigation-model variants and map concepts used in this repo. It exists so engineering work uses consistent language when discussing VLN-CE model candidates.

## LLM finetuning operation

LLM-Boxes and LLM-Grid finetuning use one fixed corpus: R2R `train` plus
English RxR `train`. Dataset provenance remains available for metrics and
artifact auditing, but prompts are tag-free. `--limit-per-dataset` is a
debugging control applied independently to each source; it does not select a
dataset.

The default rendered-token budgets are 1,152 prompt tokens and 4,096 completion
tokens. The RxR target analysis covered 19,954 examples per target. LLM-Boxes
targets had mean/P95/P99/max lengths of 1,234/2,818/3,958/7,173 tokens, with
0.81% over 4,096. LLM-Grid scale-2 targets had
1,388/2,595/3,219/4,969, with 0.15% over 4,096. The 4,096 completion budget was
chosen to retain more than 99% of each measured RxR target set while rejecting
oversized examples explicitly; 1,152 is the measured prompt budget used by
training and analysis.

Production finetuning is one Slurm node and one task with eight visible GPUs.
The launchers start one `torchrun` rank per visible GPU, using per-device batch
size 1, gradient accumulation 1, gradient checkpointing, ten epochs, and LoRA
rank/alpha/dropout 32/64/0.05. Only rank zero writes metrics and checkpoints.
An epoch checkpoint appears only after that epoch completes; cancelling
mid-epoch does not create an interruption checkpoint or preserve optimizer
state. Wait for the required `checkpoints/epoch-N` directory before cancelling.

Submit the maintained training commands with:

```text
sbatch scripts/submit/llm-boxes-train-r1p5.sh
sbatch scripts/submit/llm-boxes-train-r2p5.sh
sbatch scripts/submit/llm-grid-train-r1p5.sh
```

Generate and consume the mixed, tag-free navigation artifacts with:

```text
sbatch scripts/submit/llm-boxes-nav-cache-r1p5.sh
sbatch scripts/submit/llm-grid-nav-cache-r1p5.sh
sbatch scripts/submit/llm-boxes-current-pretrain.sh
sbatch scripts/submit/llm-grid-try5-pretrain.sh
```

## Language

**ETP-R1**:
The author's graph-based VLN-CE model used as the baseline system in this repo.
_Avoid_: author model, base model

**Training pipeline**:
The full model-development flow consisting of pretraining followed by finetuning.
_Avoid_: training, training stack

**Finetuning**:
The post-pretraining stage composed of DAgger and GRPO.
_Avoid_: tuning

**DAgger**:
The supervised online finetuning stage before GRPO.
_Avoid_: SFT when the distinction from GRPO matters

**GRPO**:
The reinforcement finetuning stage after DAgger.
_Avoid_: RFT when the specific optimizer matters

**Cognitive map**:
A spatial semantic prior over the environment that describes likely objects, regions, and navigation-relevant layout information.
_Avoid_: floor plan, occupancy map when object and region semantics matter

**Updated cognitive map**:
A full cognitive map revised during navigation using current navigation state, rather than a delta or patch to a previous cognitive map.
_Avoid_: map delta, map patch

**Map encoder**:
A model component that converts a cognitive map into map tokens for navigation.
_Avoid_: map embedder

**Cognitive map decoder**:
A model component that converts updated map tokens into an updated cognitive map for dense supervision or inspection.
_Avoid_: box decoder when the component predicts dense map logits

**Navigation architecture**:
The model-side contract for consuming cognitive-map tokens, independent of where the map came from. The supported architectures are `current` and `try5`.
_Avoid_: model candidate when only the token interaction is meant

**Current architecture**:
The navigation architecture with map-first bidirectional map-token fusion and cognitive-map box reconstruction during pretraining.
_Avoid_: default architecture, new model

**Try5 architecture**:
The navigation architecture with one-way graph queries over fixed map tokens, `direction5` metadata, and no cognitive-map decoder or box loss.
_Avoid_: Try5-like when the implementation satisfies this complete contract

**Cognitive-map source**:
The producer or cache family supplying a cognitive map. Supported sources are `prior_gt`, `imagined`, `llm_boxes`, and `llm_grid`.
_Avoid_: navigation architecture, model architecture

**Cognitive-map candidate**:
A supported pair of one navigation architecture and one cognitive-map source.
_Avoid_: cache namespace when identifying model behavior

**Bidirectional map-token fusion**:
A navigation-model interaction in which cognitive-map tokens and global navigation graph representations remain distinct, but each representation is updated using information from the other.
_Avoid_: map concatenation, pooled map fusion, single fused map

**Map predictor**:
A model component that predicts a cognitive map or cognitive-map-like representation from instruction metadata.
_Avoid_: instruction mapper

**PriorGT**:
An upper-bound candidate that uses ground-truth cognitive maps.
_Avoid_: GT model, oracle model

**Imagined**:
A candidate model that predicts cognitive maps from instructions and start metadata before navigation.
_Avoid_: OccWorld model, predicted-map model

**LLM candidate**:
A candidate model that uses an instruction-tuned language model to produce a structured cognitive-map representation from instructions and metadata.
_Avoid_: T5 candidate, Llama candidate, Tell2Design model

**LLM-Boxes**:
The predictor-only milestone for evaluating whether an LLM can generate useful object and region boxes.
_Avoid_: LLM candidate when navigation integration is not included, T5-Boxes

**LLM-Grid**:
The non-cheating LLM candidate for generating a grid-native cognitive map compatible with Try5-style map consumption, including the non-observation map metadata needed by that consumption path.
_Avoid_: LLM-Boxes when the target is not object and region boxes

**LLM-Grid-Probe**:
The earlier predictor-only milestone for evaluating whether an LLM can generate only the grid portion of a grid-native cognitive map while non-grid metadata is held outside the candidate comparison.
_Avoid_: LLM-Grid when `direction_vectors` and other required candidate map inputs are predicted by the model

**Mentioned-only LLM-Boxes target**:
The LLM-Boxes training and evaluation target restricted to relevant semantic entities whose category is mentioned by the instruction.
_Avoid_: full relevant boxes when unmentioned context entities are excluded

**LLM-Navigation**:
The navigation-integrated milestone for evaluating an LLM-derived cognitive map inside ETP-R1 finetuning and evaluation.
_Avoid_: LLM-Boxes when policy integration is included, T5-Navigation

**LLM policy**:
The separate navigation policy variant that consumes LLM-derived cognitive maps.
_Avoid_: Imagined policy when the map path uses text generation and JSON parsing

**Precomputed LLM-derived cognitive map**:
A cognitive map generated from LLM-Boxes output before navigation rollout and consumed as a fixed map input by LLM-Navigation.
_Avoid_: online LLM map when generation is not part of rollout

**LLM-Navigation cache miss**:
The absence of the expected navigation-ready `.npz` cache file for a language-filtered LLM-Navigation episode.
_Avoid_: navigation failure when the policy never receives a map input

**LLM generation failure**:
An LLM-Navigation evaluation outcome where a cache miss is counted in the evaluation denominator with zero navigation metrics and reported separately from rollout-completed episodes.
_Avoid_: fallback navigation, zero-map rollout, skipped evaluation episode

**Structured cognitive-map specification**:
A text-generated representation of predicted objects, regions, and spatial metadata that can be converted into relevant semantic boxes before rasterization.
_Avoid_: raw JSON when discussing model semantics

**Semantic entity**:
A predicted object or region entry inside a structured cognitive-map specification.
_Avoid_: room when the entry may be an object

**Canonical category name**:
The exact object or region category string used by the mapped category vocabulary.
_Avoid_: category index when discussing LLM-Boxes output

**Relevant semantic boxes**:
The instruction- and path-relevant object and region boxes selected for one scene level, carrying the ground-truth trajectory, trajectory keypoints, and start direction needed to build a cognitive map.
_Avoid_: semantic boxes when relevance filtering or predicted relevance matters

**Ground-truth trajectory**:
The dense episode trajectory from VLN-CE ground-truth files, used to select path-relevant semantic boxes for the bbox-based cognitive-map pipeline.
_Avoid_: reference path, sparse path, GT path

**Trajectory keypoints**:
A compact ordered set of selected-level path waypoints derived from the ground-truth trajectory by keeping the start and abrupt-turn waypoints, used as `(5, 2)` map-encoder metadata and as the path target for LLM-Boxes.
_Avoid_: direction vectors, reference path when discussing model metadata

**Full-level semantic boxes**:
All object and region boxes available on a selected MP3D semantic level, before relevance filtering.
_Avoid_: relevant semantic boxes

**Category-aware IoU**:
An overlap metric that only matches predictions and targets within the same canonical category and semantic entity type.
_Avoid_: IoU when category mistakes should be penalized

**Level-local coordinates**:
Metric coordinates expressed within the selected MP3D semantic level.
_Avoid_: absolute coordinates when scene-level origin matters

**Start-relative coordinates**:
Metric coordinates expressed relative to the agent's start pose, with locations interpreted from the start orientation.
_Avoid_: egocentric coordinates when discussing possible LLM-Boxes output targets

## Example Dialogue

Developer: "Should the LLM candidate replace Imagined?"

Domain expert: "No. Imagined predicts dense cognitive maps directly. The LLM candidate should be compared as another candidate map predictor."

Developer: "Can PriorGT use predicted maps?"

Domain expert: "No. PriorGT is the upper-bound candidate because it uses ground-truth cognitive maps."

Developer: "What should the LLM candidate output feed into?"

Domain expert: "It should produce a structured cognitive-map specification, convert that to relevant semantic boxes, and then rasterize to a cognitive map."

Developer: "Should LLM output use another coordinate frame?"

Domain expert: "No. The current LLM candidate should output level-local coordinates, while start-relative coordinates remain a future research option."

Developer: "Should generated JSON identify categories by list position?"

Domain expert: "No. LLM-Boxes should emit canonical category names, which are validated and mapped to category indices deterministically."

Developer: "Should objects and regions share one generated array?"

Domain expert: "No. The structured cognitive-map specification should keep separate object and region arrays."

Developer: "Should LLM-Boxes generate whether an entity was mentioned?"

Domain expert: "No. Mention status should be derived after generation from the instruction and generated categories."

Developer: "Should LLM-Grid reuse the LLM-Boxes target format?"

Domain expert: "No. LLM-Grid predicts a grid-native cognitive-map target, while LLM-Boxes predicts structured object and region boxes before rasterization."

Developer: "Can we call a grid-only prediction experiment LLM-Grid?"

Domain expert: "No. A grid-only predictor is LLM-Grid-Probe; LLM-Grid is reserved for the non-cheating candidate that predicts the needed map inputs rather than borrowing ground-truth metadata."

Developer: "Can a predictor path become LLM-Grid once it predicts `direction_vectors`?"

Domain expert: "Yes. Predicting sparse grid anchors plus Try5-style `direction_vectors` removes the main ground-truth metadata leak, so the experiment should be reported as LLM-Grid."

Developer: "Should LLM-Boxes generate confidence scores?"

Domain expert: "No. Confidence should be assigned during conversion or rasterization rather than generated by LLM-Boxes in the first version."

Developer: "Should LLM-Boxes generate the reference path?"

Domain expert: "No. LLM-Boxes should generate trajectory keypoints using a `keypoints` entity, not a reference path."

Developer: "Should LLM-Boxes predict all boxes on the level?"

Domain expert: "No. LLM-Boxes should target relevant semantic boxes, not full-level semantic boxes."

Developer: "Should LLM-Boxes train on unmentioned relevant context entities?"

Domain expert: "No. The current LLM-Boxes target is mentioned-only so the generated text stays focused on categories named by the instruction."

Developer: "Should LLM-Boxes include the instruction source?"

Domain expert: "No. Dataset provenance is retained for metrics and artifacts, but it is excluded from the model prompt so the predictor cannot specialize on a dataset identity side channel."

Developer: "Should LLM-Boxes include scene identity?"

Domain expert: "No. Scene id and scan id should be excluded from the LLM-Boxes prompt."

Developer: "Should LLM-Boxes include start metadata?"

Domain expert: "Yes. The prompt should include level-local start position and start direction."

Developer: "How precise should LLM-Boxes numbers be?"

Domain expert: "Coordinates and extents should use one decimal place in meters, and rotations should use two decimals in radians."

Developer: "Should invalid LLM-Boxes output be repaired by another model decode?"

Domain expert: "No. Unless a repair objective is trained, invalid output should be handled by deterministic validation, entity dropping, clipping, or empty fallback."

Developer: "Should LLM-Boxes use rotation augmentation immediately?"

Domain expert: "No. Establish the no-augmentation baseline first, then add right-angle rotation augmentation and report it separately."

Developer: "How should LLM-Boxes fine-tune its language-model backend?"

Domain expert: "Start with end-to-end fine-tuning, while keeping the language-model backend configurable."

Developer: "What is the primary LLM-Boxes metric?"

Domain expert: "Use category-aware IoU, especially category-aware raster IoU, with JSON validity and entity validity as diagnostics."

Developer: "Should LLM-Navigation be folded into Imagined?"

Domain expert: "No. LLM-Navigation should use separate policy and trainer names while reusing shared map helpers."

Developer: "Should LLM-Navigation run the LLM online during rollout?"

Domain expert: "No. LLM-Navigation should consume precomputed LLM-derived cognitive maps so rollout uses deterministic map tensors and generation failures can be measured before navigation."

Developer: "How should LLM-Navigation evaluation handle a missing cache file?"

Domain expert: "Treat the LLM-Navigation cache miss as an LLM generation failure: keep the episode in the denominator, assign zero navigation metrics, and report cache-missing count and rate."

Developer: "Should generated entities be ordered along the path?"

Domain expert: "No. LLM-Boxes should use deterministic category and geometry ordering rather than path-order serialization."

Developer: "Should LLM-Boxes save JSON prediction artifacts?"

Domain expert: "No. Save plain LLM-Boxes text artifacts so artifacts mirror the model-facing compact text format."

Developer: "Does bidirectional map-token fusion mean merging the cognitive map and topological map into one representation?"

Domain expert: "No. The cognitive-map tokens and global navigation graph representations remain distinct; fusion updates both representations without collapsing them into a single map."

Developer: "Should cognitive-map tokens attend to the STOP action when they are updated from the global navigation graph?"

Domain expert: "No. STOP is an action pseudo-node rather than a spatial graph node, so bidirectional map-token fusion should update cognitive-map tokens from real valid graph nodes."
