# Defense notes

Generated from the implemented protocol and `results/summary.json` by `python scripts/document_defense.py`.
These notes explain the implemented protocol and its limitations.

Current questions: 67. Empirical mitigation results and later modules require additional questions when delivered.

**What question do the completed learning measurements answer?**

Whether the cosine reward's self-inclusion effect reproduces under the declared federated protocol. Implementing a proposed mitigation does not establish its effectiveness; pilot selection and evaluation must supply that evidence.

**What is the experimental unit?**

A seeded complete training run. Client participations within a run share model history and are dependent; treating them as independent runs would overstate statistical evidence.

**What is a model update?**

The flattened local model after training minus its starting global snapshot. Both parameter arrays are converted to float64 before subtraction; training parameters themselves use float32.

**Why this architecture?**

The unpadded convolutional model in models.py has 44,426 trainable parameters. It fixes the source's approximate model description, so noise-vector length and update norms are reproducible.

**What changes if convolution padding is added?**

The flattened feature dimension and parameter count change. That changes both optimization and the total magnitude of coordinatewise Gaussian noise; it is a protocol change, not a harmless refactor.

**Why does the classifier return logits?**

Cross entropy applies the appropriate log-softmax internally. Adding a softmax layer before that loss would change the objective and gradients.

**How are training and test data separated?**

Only the training split is partitioned among clients. Central test evaluation runs after aggregation and does not provide validation scores to the reward rules.

**What does the Dirichlet parameter control?**

The protocol uses alpha=0.5. It controls concentration of per-class client proportions; smaller values tend to yield more unequal class allocations. It is not an attacker parameter.

**Why redraw the entire partition when a client is too small?**

A minimum sample count avoids empty local training. Deterministic full redraw preserves disjoint coverage and avoids an undocumented redistribution heuristic.

**How is sampling independent of message arrival?**

data.py maps the run seed and round to a sampling RNG stream. The server orders received updates by sampled scientific client id, not by the order replies arrive.

**Why use coordinate-derived local seeds?**

A local training stream depends on run seed, source round and client id. Scheduling the same client on a different worker cannot consume a different shared RNG sequence.

**What is the local optimizer?**

Fresh SGD each round with learning rate 0.01, momentum 0.9, batch size 32 and 5 local epochs. Carrying momentum between rounds would define a different experiment.

**Why explicitly shuffle with a generator?**

Minibatch order affects optimization. An explicit generator makes the order part of the seeded protocol instead of an accidental consequence of worker process state.

**What does Flower actually execute?**

The simulation runtime schedules ClientApp train messages. Each client loads its own indexed partition, trains the transmitted snapshot and replies with a delta; ServerApp performs aggregation and logging.

**Why does the runner use a deprecated entry point?**

The pinned Flower version still routes that entry point to its official Message API simulation runtime. This makes the documented Python command self-contained; future upgrades require deliberate migration and repeatability checks.

**What is the difference between a node id and a client id?**

A node id is a runtime transport address. A scientific client id selects a data partition and RNG stream. Keeping them separate prevents new transport addresses from changing the experiment.

**Why check duplicate replies and sample counts?**

A missing or duplicated update would alter both training and payouts. The server rejects unexpected identities and partition-size mismatches instead of silently computing a different round.

**What is the training aggregator?**

FedAvg weighted by local sample counts, including adversarial reports. Removing adversaries here would confound the reward comparison with a defended training trajectory.

**Why is the scoring reference unweighted?**

The reproduced cosine definition uses the unweighted mean of sampled updates. The training aggregator and the reward reference are deliberately different objects and are documented separately.

**How do all reward rules see the same evidence?**

score_round passes the same read-only update matrix to every rule. Its digest is checked before and after scoring and recorded with every participation; a rule cannot silently modify later rules' inputs.

**What would removing the read-only flag risk?**

An in-place scorer could modify the shared matrix, changing another rule's reward or the training update. The digest check and non-mutation tests guard this comparison contract.

**Why is cosine defined as zero for a zero norm?**

Its mathematical denominator vanishes. The explicit convention prevents NaNs from propagating into integer quality, shares or the payout ledger.

**Why truncate cosine quality?**

The source defines a floored nonnegative quality score. Using continuous cosine or rounding to nearest changes reward ratios near bin boundaries.

**What happens when every quality score is zero?**

No cosine payout is minted. The round is counted as empty; the implementation does not replace the rule with equal split.

**Why can a random update earn cosine reward?**

The reference includes that same vector. Its self inner-product contributes positively even if its cross-products with the other updates are small. As other updates shrink, this term can dominate the reference.

**Is the self-inclusion prediction an identity?**

No. It drops cross-products between noise and other updates. The analysis logs every measured residual, including the largest, instead of selecting only well-fitting rounds.

**What is the relative-error denominator?**

The absolute measured cosine. A measured zero is reported as undefined and counted, rather than repaired with an arbitrary epsilon or silently discarded.

**Why are missing noise-round means not zero?**

No sampled noise client means no observation for that group. Filling the gap with zero would falsely imply that noise was present and rejected.

**How does Multi-Krum become a reward baseline?**

The selected set receives equal shares; rejected updates receive none. The selected set does not train the global model in this shadow comparison, so robustness guarantees for Krum aggregation are not claimed.

**Why supply the true fault budget to Multi-Krum?**

It is a favorable baseline assumption, explicitly disclosed. It avoids weakening the comparison through a deliberately underestimated attacker budget.

**Why is stable tie-breaking important?**

Duplicated or all-zero updates can have equal distance scores. Stable sorting makes the chosen set deterministic for a fixed input order; this baseline is not claimed to be permutation invariant at exact ties.

**What does the zero attack change?**

It replaces the local report with an all-zero delta. The experiment still performs local training before the replacement, keeping runtime behavior explicit rather than pretending to measure an optimized free-rider client.

**How is Gaussian noise generated?**

Each coordinate is sampled with standard deviation 0.01 from a run/round/client-derived NumPy stream. This is a replacement attack, not honest differential privacy noise.

**How is stale replay defined?**

Train the client's data from the saved snapshot at max(initial round, current round minus 5), with that source round's shuffle seed. The returned delta is relative to the old snapshot.

**Why is stale replay not just the client's previous report?**

A client may not have participated at the requested earlier round. Recomputing its hypothetical local update from the saved snapshot makes the lag definition independent of participation history.

**What does the own-report sensitivity change?**

It replays the most recent report by that identity at or before the lagged round, with zero when no such report exists. Every other scientific setting and the participation schedule remain fixed. This is a separate declared experiment, not a replacement of the main snapshot-based definition.

**Why are all stale reports zero in that sensitivity?**

A persistent stale identity has no honest warm-up report. Its first missing-history report is zero; later eligible reports recursively copy a zero. This follows from the declared initial condition. A warm-up or a cache of honest computed updates would define a different attack.

**Does the sensitivity isolate only the direct stale payout?**

No. Changing a stale report also changes subsequent FedAvg models and later honest and noise rewards. The paired seed comparison measures the total trajectory effect of that semantic change, not a decomposition into independent causes.

**How do label and sign attacks differ?**

Label flipping changes targets during local SGD; sign flipping negates the resulting update. Their effects on the trained model and on cosine are different, so each has a separate smoke configuration.

**Does shadow scoring show that a reward rule improves accuracy?**

No. All rules share a training trajectory, so accuracy is identical within a run. Claims that incentives change participation or training require a declared closed-loop experiment.

**What is attacker share?**

Total minted payouts to adversarial identities divided by all minted payouts across the run. It is not an unweighted mean of per-round percentages, nor the registered attacker fraction.

**What is an honest false penalty?**

An honest participation receiving less than half of an equal-split share. It is an operational metric, not proof that the update was objectively valuable; separate counts track honest clients slashed.

**Which clients enter honest-reward Gini?**

All registered honest clients, including any with no participation or no payout. Excluding zero-reward clients would understate inequality.

**How is payout rounding handled today?**

Every float share is converted to an exact Fraction, normalized by the exact sum and floored at the integer scale. No client receives the residue; the unminted amount is recorded. Random-matrix conservation tests and replay from immutable CSV guard this boundary. On-chain equality still requires the contract replay.

**What evidence establishes CPU repeatability?**

Run the full seeded protocol twice into separate folders and compare byte hashes of participation CSV, round CSV and ledger. Repeating only a scorer or a reduced smoke run does not establish full-harness repeatability.

**What does a GPU resource fraction mean?**

It is a scheduler reservation per actor, not a hard memory quota. Metadata records the fraction and worker count; changing concurrency must preserve scientific identities and random streams.

**Does passing the cross-device accuracy tolerance establish equal rewards?**

No. Small numerical differences during training can change cosine bins and reward trajectories. The device comparison reports score and share differences explicitly; the predeclared acceptance tolerance concerns final accuracy only.

**Can runtime be reproduced byte for byte?**

No. Hardware and load affect elapsed time. Recorded wall time includes imports and runtime startup; performance is assessed against a bound, while scientific logs use exact-repeat checks on the same CPU environment.

**Where do the figures and numerical macros come from?**

The reproduction script collects committed run artifacts into results/summary.json. The figure, macro and table scripts consume that shared summary; no plot is edited by hand.

**Why do the reproduced shares differ from the source?**

The source does not fully disclose layer, preprocessing, partition and RNG details. Realized adversary participation also varies. The notes report these possible causes and observed payout components without claiming an identified causal explanation.

**What do the current tests establish?**

They guard numerical baselines, model shape, attack transformations, partition coverage, deterministic training and common-update scoring. They do not establish adaptive-attack security, contract safety or cross-platform bit identity.

**What is outside the current threat model?**

Adaptive contribution-gaming attacks such as ACE, coordinator dishonesty, secure aggregation and real incentive-driven client behavior. Success against fixed attacks must not be described as general Byzantine security.

**How is a client's own update excluded?**

ours.py removes that row before taking each coordinate's median. Perturbing that row cannot change its reference. Its own cosine still depends on its report, as it must; the exclusion property applies to the reference, not to the entire score.

**Why use a coordinate median?**

It limits extreme-coordinate influence when the other sampled updates have an honest majority. It is inexpensive for small samples, but basis dependent and potentially unfair to legitimate minority data. There is no universal contribution-value guarantee.

**What happens with a single sampled client?**

Its excluding reference is zero, so its quality and reward are zero. Inventing a self-reference would violate the exclusion contract. This edge behavior is tested explicitly.

**Why rescale vectors before normalizing them?**

Dividing by the largest absolute coordinate avoids overflow or underflow in the norm across wide representable scales. Zero vectors are handled explicitly. An additive epsilon would change the cosine under common rescaling.

**Why key state by client id?**

Sample order and attendance change each round. Reputation and low-quality counters must follow the scientific identity; storing them by row would transfer one client's history to another.

**What does absence do to reputation?**

Nothing. Only participations update the exponential moving average or low-quality counter. This is participation-based history, not wall-clock decay.

**Why multiply reputation by current quality?**

Historical credit alone must not pay a current zero contribution. The product combines immediate alignment and prior history; the ablation measures whether this costs honest clients more than it suppresses attacks.

**How can a penalized identity recover?**

The low-participation counter resets when a slash is emitted and after a useful participation. Future useful contributions can receive rewards. There is no permanent ban in the scorer.

**Why does disabling penalties also disable vesting?**

The economic component includes both deferred rewards and slash exposure. Reference-only and reference-plus-reputation allocations use immediate claims; the full component uses the fixed delay. Historical minted-share metrics remain separate from subsequent burns.

**How do cached pilot matrices remain tied to actual training?**

cache.py checks the completed run and its artifact hashes, exact client order, finite matrix shape and the common-update digest recorded in every participation. Altering an identity, a matrix or its log fails validation.

**Why share reference calculations between pilot candidates?**

Every candidate has identical client-excluding reference geometry. Only its independent reputation and penalty state differs. Computing quality once saves repeated median work; direct-versus-shared scoring is checked for exact equality.

**What prevents selecting a rule that pays nobody?**

A completely empty attacked cell incurs the predeclared unit attacker-allocation cost and its honest false-penalty cost. Undefined shares are never silently dropped. The objective also reports empty rounds and honest slash counts separately.

**What is frozen before evaluation?**

The selected parameters, plan, design prior, scorer, selection program, corpus and selection-summary digests are recorded in a committed configuration. The pipeline rejects changed scientific inputs. Evaluation seeds cannot choose a threshold or repair a disappointing result.

**Are repeated fraction labels independent evidence?**

No. The fixed mixture and attack-free setup repeat the same scientific scenario across those labels. Coverage retains the labels explicitly; statistical uncertainty must resample complete seeds rather than treat duplicated labels as independent observations.

**Why does the repository prescribe LF checkouts?**

Artifact hashes cover exact file bytes. Automatic CRLF conversion changes recorded CSVs without changing their apparent rows. Git attributes preserve LF and retain binary treatment for PDF and PNG assets.
