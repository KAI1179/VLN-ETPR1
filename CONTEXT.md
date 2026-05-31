# ETP-R1 Navigation Models

This context describes the navigation-model variants and map concepts used in this repo. It exists so engineering work uses consistent language when discussing VLN-CE model candidates.

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

**Map encoder**:
A model component that converts a cognitive map into map tokens for navigation.
_Avoid_: map embedder

**Map predictor**:
A model component that predicts a cognitive map or cognitive-map-like representation from instruction metadata.
_Avoid_: instruction mapper

**PriorGT**:
An upper-bound candidate that uses ground-truth cognitive maps.
_Avoid_: GT model, oracle model

**Imagined**:
A candidate model that predicts cognitive maps from instructions and start metadata before navigation.
_Avoid_: OccWorld model, predicted-map model

**T5 candidate**:
A candidate model that uses a text-to-text transformer to produce a structured cognitive-map representation from instructions and metadata.
_Avoid_: Tell2Design model

**T5-Boxes**:
The predictor-only milestone for evaluating whether T5 can generate useful object and region boxes.
_Avoid_: T5 candidate when navigation integration is not included

**Mentioned-only T5-Boxes target**:
The T5-Boxes training and evaluation target restricted to relevant semantic entities whose category is mentioned by the instruction.
_Avoid_: full relevant boxes when unmentioned context entities are excluded

**Dataset tag**:
A prompt-side label identifying the instruction source, such as R2R, RxR, or Gemini-augmented Prevalent data.
_Avoid_: task type when referring to generated text prompts

**T5-Navigation**:
The navigation-integrated milestone for evaluating a T5-derived cognitive map inside ETP-R1 finetuning and evaluation.
_Avoid_: T5-Boxes when policy integration is included

**T5 policy**:
The separate navigation policy variant that consumes T5-derived cognitive maps.
_Avoid_: Imagined policy when the map path uses text generation and JSON parsing

**Structured cognitive-map specification**:
A text-generated representation of predicted objects, regions, and spatial metadata that can be converted into relevant semantic boxes before rasterization.
_Avoid_: raw JSON when discussing model semantics

**Semantic entity**:
A predicted object or region entry inside a structured cognitive-map specification.
_Avoid_: room when the entry may be an object

**Canonical category name**:
The exact object or region category string used by the mapped category vocabulary.
_Avoid_: category index when discussing T5 output

**Relevant semantic boxes**:
The instruction- and path-relevant object and region boxes selected for one scene level, carrying the reference path and start direction needed to build a cognitive map.
_Avoid_: semantic boxes when relevance filtering or predicted relevance matters

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
_Avoid_: egocentric coordinates when discussing possible T5 output targets

## Example Dialogue

Developer: "Should the T5 candidate replace Imagined?"

Domain expert: "No. Imagined predicts dense cognitive maps directly. The T5 candidate should be compared as another candidate map predictor."

Developer: "Can PriorGT use predicted maps?"

Domain expert: "No. PriorGT is the upper-bound candidate because it uses ground-truth cognitive maps."

Developer: "What should the T5 candidate output feed into?"

Domain expert: "It should produce a structured cognitive-map specification, convert that to relevant semantic boxes, and then rasterize to a cognitive map."

Developer: "Should T5 output use another coordinate frame?"

Domain expert: "No. The current T5 candidate should output level-local coordinates, while start-relative coordinates remain a future research option."

Developer: "Should generated JSON identify categories by list position?"

Domain expert: "No. T5 should emit canonical category names, which are validated and mapped to category indices deterministically."

Developer: "Should objects and regions share one generated array?"

Domain expert: "No. The structured cognitive-map specification should keep separate object and region arrays."

Developer: "Should T5 generate whether an entity was mentioned?"

Domain expert: "No. Mention status should be derived after generation from the instruction and generated categories."

Developer: "Should T5 generate confidence scores?"

Domain expert: "No. Confidence should be assigned during conversion or rasterization rather than generated by T5 in the first version."

Developer: "Should T5-Boxes generate the reference path?"

Domain expert: "No. T5-Boxes should focus on object and region boxes. Reference-path handling belongs to the T5-Navigation milestone."

Developer: "Should T5-Boxes predict all boxes on the level?"

Domain expert: "No. T5-Boxes should target relevant semantic boxes, not full-level semantic boxes."

Developer: "Should T5-Boxes train on unmentioned relevant context entities?"

Domain expert: "No. The current T5-Boxes target is mentioned-only so the generated text stays focused on categories named by the instruction."

Developer: "Should T5-Boxes include the instruction source?"

Domain expert: "Yes. T5-Boxes should include a dataset tag in the prompt, mirroring the existing task-type encoding side channel."

Developer: "Should T5-Boxes include scene identity?"

Domain expert: "No. Scene id and scan id should be excluded from the T5-Boxes prompt."

Developer: "Should T5-Boxes include start metadata?"

Domain expert: "Yes. The prompt should include level-local start position and start direction."

Developer: "How precise should T5-Boxes numbers be?"

Domain expert: "Coordinates and extents should use one decimal place in meters, and rotations should use two decimals in radians."

Developer: "Should invalid T5 output be repaired by another T5 decode?"

Domain expert: "No. Unless a repair objective is trained, invalid output should be handled by deterministic validation, entity dropping, clipping, or empty fallback."

Developer: "Should T5-Boxes use rotation augmentation immediately?"

Domain expert: "No. Establish the no-augmentation baseline first, then add right-angle rotation augmentation and report it separately."

Developer: "How should T5-Boxes fine-tune T5?"

Domain expert: "Start with end-to-end T5-large fine-tuning, while keeping the T5 model size configurable."

Developer: "What is the primary T5-Boxes metric?"

Domain expert: "Use category-aware IoU, especially category-aware raster IoU, with JSON validity and entity validity as diagnostics."

Developer: "Should T5-Navigation be folded into Imagined?"

Domain expert: "No. T5-Navigation should use separate policy and trainer names while reusing shared map helpers."

Developer: "Should generated entities be ordered along the path?"

Domain expert: "No. T5-Boxes should use deterministic category and geometry ordering rather than path-order serialization."

Developer: "Should T5-Boxes save JSON prediction artifacts?"

Domain expert: "No. Save plain T5-Boxes text artifacts so artifacts mirror the model-facing compact text format."
