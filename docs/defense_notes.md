# Defense notes

Generated from the implemented protocol and `results/summary.json` by `python scripts/document_defense.py`.
These notes explain the implemented protocol and its limitations.

Current questions: 128. Empirical mitigation results and later modules require additional questions when delivered.

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

All registered honest clients, including any with no participation or no payout. Excluding zero-reward clients would understate inequality. Sorted payouts stay as integers through the rank-weighted numerator and denominator; Fraction converts the final ratio to float once.

**Why migrate derived metrics without retraining?**

The old floating dot product rounded large integer payouts before reduction, allowing different platform kernels to change Gini's last bits. Exact integer arithmetic fixes the metric definition. The migration preserves training logs and ledgers, records old/new metric hashes and affected fields, and requires Windows and Linux to regenerate identical bytes.

**Why compute F1 directly from integer counts?**

The direct ratio 2*TP/(2*TP+FP+FN) avoids feeding rounded precision and recall into another division. The final conversion is deterministic, and the zero-denominator convention remains explicit.

**How is payout rounding handled today?**

Every float share is converted to an exact Fraction, normalized by the exact sum and floored at the integer scale. No client receives the residue; the unminted amount is recorded. Random-matrix conservation tests and replay from immutable CSV guard this boundary. The contract replay compares full account states with an independent integer oracle.

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

They guard numerical baselines, model shape, attack transformations, partition coverage, deterministic training and common-update scoring. Additional contract checks cover state transitions and fuzz inputs; exact derived-metric reconstruction has been checked on Windows and Linux. These checks do not establish adaptive-attack security or universal contract safety.

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

**Why does the attack-free case appear once per seed?**

Changing an attacker-fraction label does not change an attack-free experiment. The fixed mixture likewise has its own fixed membership. The plan includes each of those conditions once per seed; only the single-attack conditions vary their fractions. Uncertainty resamples complete seeds rather than treating duplicate labels as extra observations.

**Why does the repository prescribe LF checkouts?**

Artifact hashes cover exact file bytes. Automatic CRLF conversion changes recorded CSVs without changing their apparent rows. Git attributes preserve LF and retain binary treatment for PDF and PNG assets.

**Why retain separate mixture cohorts?**

The direct source comparison uses the source's declared seeds. The overall evaluation uses all planned seeds on the same device. Separate summaries, intervals and macro names prevent the larger cohort from silently replacing the original comparison.

**What does total grid wall time include?**

The execution clock starts before the first training attempt and ends after all planned cells validate. Interruptions, retries and validation overhead remain in that interval. The sum of successful run durations is reported separately, and later validation cannot move the recorded endpoints.

**How is the original training implementation verified?**

The mixture analysis requires a clean recorded source commit reachable from the published branch. It verifies every recorded training source file against that commit, along with common configurations, versions and artifact hashes. A matching current filename or seed label alone is insufficient.

**Who can create tokens?**

RewardToken fixes its pool address at deployment. Only that pool can mint or burn its held tokens; the coordinator cannot call the token's mint function directly. The pool controls equal setup grants and later reward creation.

**Does registration prevent Sybil identities?**

No. The coordinator controls the initial grants and thus enrollment. Equal stakes do not prove distinct people or devices. A public identity or admission mechanism is outside this experiment.

**What does the commitment prove?**

Only that a sender committed to a particular digest and salt for that round. It does not prove local computation, useful data, update correctness or score honesty. Replay hashes a canonical logged participation record; it is not a raw-update proof.

**Why include the sender and round in a commitment?**

They prevent another sampled address or another round from reusing the same commitment as its own. Deadlines separate commitment from disclosure; each address can commit and reveal only once per round.

**Can a client be paid without revealing?**

A positive share requires a successful reveal. A nonrevealer can still receive a zero score and a coordinator-posted slash flag. The contract enforces settlement inputs, not a fair scoring policy.

**Why require the posted sample order?**

The recorded sample is unique and registered. Requiring the exact same ordered list in score posting prevents duplicate payment entries, omitted clients and substitution after participation.

**Why normalize exact float shares before fixed-point conversion?**

Binary float shares may sum slightly above or below unity. Fraction captures each recorded value exactly; normalizing their rational sum and flooring guarantees that integer shares cannot overallocate the pool.

**What is the settlement ordering?**

Mature old rewards for every registered client, slash flagged clients' stake and remaining unvested tranches, then create current rewards. This protects rewards whose unlock round has arrived, including for absent clients.

**Why slash each tranche separately?**

Integer division floors each tranche's burn. Flooring their combined balance could differ by base units. Both the Solidity code and the independent oracle implement the declared per-tranche policy.

**Why use a ring for vesting?**

Sequential rounds and a fixed bounded delay let settled rewards reuse a finite set of slots. A slot is checked before reuse; the independent oracle instead stores explicit unlock-round keys, reducing correlated implementation mistakes.

**Does vesting advance with elapsed time?**

No. It advances when rounds settle. Rewards created near the final experiment round remain unvested unless later rounds settle. There is no separate finalization, participant exit or stake withdrawal in this research contract.

**Why clear claimable rewards before transfer?**

The liability is removed before an external token interaction. The token is immutable and has no callback hook, but preserving this state-before-transfer ordering makes duplicate claims impossible.

**What does exact on-chain replay compare?**

The recorded evidence covers 180 round-rule cases. Every state comparison includes stake, wallet, claimable balance, each unlock tranche, pool balance, total supply and unminted remainder in base units, not only final aggregate payouts.

**Why add synthetic state histories to FL replay?**

The recorded baselines do not emit slashes. Independent histories exercise 228 slash participations over 96 rounds with changing participation, empty allocations, immediate and delayed vesting, and deferred claims.

**What prevents a stale result file from validating new code?**

Every contract measurement binds the contract, token library, compiler configuration, environment pins, integer oracle and measurement programs by source hashes. The summarizer also reconstructs every recorded state and rejects mismatched training-log hashes.

**Does a clean static analysis prove safety?**

No. Tests, fuzzing and static analysis cover different failure classes but do not constitute formal verification or an independent audit. Timestamp findings are documented deadline behavior; trusted coordination and missing exit remain explicit limitations.

**Why is the gas cap applied to transaction limits?**

EIP-7825 caps the requested gas limit at 16,777,216. Receipt gas may include refunds. A transaction whose receipt falls below the cap can still need more execution gas than the permitted limit.

**How was the participant boundary established?**

Separate local chains enforce the cap with an explicit transaction limit. In the measured history, 142 clients complete every round, while 143 exhaust settlement gas in round 4. This is a scenario boundary, not a universal capacity promise.

**Why can a round's total gas exceed the transaction cap?**

A round contains many separate client and coordinator transactions. The cap applies to each transaction, not their sum. Total gas measures aggregate cost; the largest single operation controls executability.

**Can our gas values be called a speedup over the source contract?**

No. Deployment, commitment, slashing and vesting functionality and transaction boundaries differ. The comparison lists the published operation costs alongside our measured ranges without claiming equivalent workloads.

**Is CGSV an exact Shapley-value calculation?**

No. The implementation uses the paper's cosine approximation with a reputation-weighted inclusive reference. Exact coalition enumeration is absent; the method name does not imply exact Shapley values.

**Why do the reputation rules begin with uniform registered histories?**

This is an explicit partial-participation adaptation. Zero histories combined with global normalization can penalize an identity when it is sampled for the first time after others accumulated importance. Uniform priors give every registered identity initial mass. The source's zero-history/data-size startup is documented as different.

**What happens to an identity that is not sampled?**

It receives no new cosine observation and is not removed for being absent. Global normalization can rescale its importance units. Permanent RFFL exclusion is evaluated only when an active identity participates.

**Which source parameters are declared for the extensions?**

Both mechanisms use alpha=0.95 and gamma=0.5. CGSV beta candidates are [1.0, 1.2, 1.5, 2.0], with importance floor 0.001. RFFL uses the fixed registered-population threshold 1/(3*N). The sources and implementation differences are linked in EXTENSIONS.md.

**How are gradient rewards adapted into tokens?**

Normalize the continuous gradient-budget quotas over eligible sampled identities. There is no integer gradient mask, personalized download or change to the training aggregator. Original gradient-reward fairness guarantees therefore do not automatically transfer to this token adaptation.

**Can an excluded RFFL identity still change the trained model?**

Yes. Exclusion affects the shadow scorer's reference and payouts only. All compared rules consume the same FedAvg trajectory. Changing which updates train the model would require a separate declared closed-loop comparison.

**Can history pay a current zero update?**

Yes, when other updates create a nonzero reference. Quotas depend on updated reputation, not an additional multiplication by current cosine. If the entire common reference is zero, no gradient information exists and token payment is zero.

**What happens when signed reputation normalization has no positive mass?**

The state remains unchanged, the round pays nothing and the undefined event is counted. Candidate selection refuses such a history; an empty payout must not silently improve the attacker objective.

**Why can CGSV candidates reuse geometry?**

Altruism beta changes the quota transform, not the reputation update or reference. Every candidate therefore replays one shared geometry history while producing its own exact integer payouts. RFFL has an independent reputation trajectory on the same matrices.

**How is self-inclusion measured for weighted normalized updates?**

The exact self term is reference weight times normalized update norm divided by reference norm. The residual completes the measured cosine identity. The log also retains cosine after removing the current update, eligibility and every noise participation. Undefined geometry and zero measured cosine are counted explicitly.

**Why record geometry separately from token allocation?**

Recorded cosine geometry allows exact state and integer-payout reconstruction across platforms without retaining raw matrices in version control. Raw matrix hashes, sampled order and source bindings identify the calculation. Raw geometry recomputation is a separate numerical check; it is not claimed bit-identical across operating systems.

**What prevents extensions from being tuned on evaluation updates?**

Selection requires complete disjoint pilot coverage and committed scientific sources. Evaluation first checks the published frozen configuration and complete evidence, then reconstructs the pilot comparison. Missing publication, changed sources or an incomplete pilot corpus stop execution before evaluation scoring.

**What does the extension excluding-reference contrast establish?**

It holds the observed update set and reputation weights fixed and removes the current update from the reference. This isolates an arithmetic contribution to cosine; it does not establish the outcome of a separately trained or separately evolved leave-one-out reputation mechanism.

**How are undefined extension histories handled in inference?**

A nonpositive normalization event invalidates that run's inferential reward and bias quantities. Seed intervals remain undefined when any required observation is missing. Integer accounting, empty rounds and undefined-event counts are retained so failures are visible.

**Where is privacy noise applied?**

The client clips its complete flattened update and adds Gaussian noise before constructing the Flower reply. The server aggregates exactly that released vector; every scorer, cached matrix and norm log uses it. There is no offline perturbation substituted for a noisy training trajectory.

**How is the clipping bound selected?**

The median norm over every participation in the attack-free pilot seeds [100, 101, 102, 103, 104]. The rule is declared before inspecting outcomes and the selected scale is published before evaluation. A single scale is held fixed across all multipliers.

**Why is replacement sensitivity twice the clipping bound?**

Two possible full-client updates each lie in the L2 ball of radius C. Their distance is at most 2C by the triangle inequality, attained by opposite boundary points. This is client-dataset replacement with public registration, partition assignment and record counts held fixed.

**Why is no subsampling amplification claimed?**

The coordinator sees each sampled identity and each individual update. An aggregate-only or hidden-sampling accountant would describe a different observation model. We compose the unsampled Gaussian bound using the public participation schedule.

**What does the Gaussian multiplier mean?**

Every coordinate has noise standard deviation sigma*C. Dividing by the square root of the model dimension would silently alter that multiplier. If the declared scale overflows or underflows, the mechanism rejects it instead of omitting noise.

**How does the RDP accountant work?**

At Renyi order a, replacement sensitivity 2C gives 2*a/sigma squared per observed release. Compose over t participations, then minimize the standard epsilon conversion over real a greater than one. The analytic optimum yields rho plus twice the square root of rho times log(1/delta), where rho equals 2*t/sigma squared.

**What happens at zero noise or zero participations?**

A client never sampled incurs zero release cost. Positive participation with zero noise has infinite epsilon, represented by finite=false and epsilon=null. The zero-noise training control is still clipped, so it is different from the unclipped main-grid baseline.

**Do the published experimental seeds preserve privacy?**

No. Known noise can be subtracted, and a floating-point seeded sampler is not a certified private mechanism. Reported bounds calibrate ideal undisclosed Gaussian randomness; the actual public-data simulation is not a privacy guarantee.

**Does the accountant cover pilot calibration?**

No. Its ideal bound assumes C and auxiliary metadata are fixed public constants independent of a subsequently protected dataset. Choosing a clipping bound from private data would need separate privacy accounting. Here calibration uses the public benchmark and is disclosed as research evidence.

**Why report both actual participation and all-round bounds?**

Sampling creates different observed release counts per identity. Per-client accounting reports those counts and bounds; the conservative all-round bound shows the cost if one identity participated every round. Neither treats client participations as independent experimental seeds.

**Can privacy-noise results change the reward parameters?**

No. The comparison reuses the already-published reward parameters and includes the reference-only stage diagnostically. No disappointing evaluation result changes the clipping scale, noise multiplier or reward thresholds.

**How are privacy costs and utility compared?**

Every rule within a run receives the same released updates. Noise multipliers have separate perturbed training trajectories on one device. Honest false penalties, slashes, inequality and accuracy retain complete seed-level observations and uncertainty, including poor utility or increased honest penalties.

**Why use Fashion-MNIST for transfer?**

It preserves the image shape and architecture while changing the visual task and class semantics. That isolates a dataset change without claiming that two small image datasets establish broad generality.

**Which transfer conditions were selected?**

The fixed mixture transfers the source comparison; label flip at the largest declared fraction tests direct training-data poisoning. Both are declared before inspecting any Fashion-MNIST run.

**Can the transfer experiment tune the reward rule?**

No. Every numerical parameter comes from the published MNIST pilot selection. The transfer runner uses the same publication gate and unchanged method; its component stages only switch the declared mechanisms.

**What is checked before transfer training?**

The complete dataset checksums and normalization, published source files and method evidence, full split configuration, and declared device. Each completed cell must retain the same source hashes, library versions and method digest.

**Why can three transfer seeds give a misleadingly narrow interval?**

Bootstrap resampling cannot create unseen training outcomes. With only 3 seeds, intervals are coarse and may be unstable; the minimum all-nonzero two-sided seed-level signed-rank p-value is 0.25. The dataset comparison is descriptive.

**Why is model accuracy reported once in the transfer table?**

Every reward rule shadows the same submitted matrix and fixed FedAvg trajectory. Reward differences cannot be credited with changing this model. Separate datasets and poisoning conditions can have different accuracy, but shadow rules within a run cannot.

**Why does the support archive read committed Git objects?**

Working files can contain unfinished edits or unrelated local data. Reading one commit's tree and blobs fixes the exact asset set independently of checkout line endings and later edits. The archive preserves those bytes and records their hashes.

**What does the asset manifest establish?**

It identifies the source commit, summary digest and every packaged file. Present and absent result sections make incomplete coverage visible. Hash agreement establishes byte identity, not scientific correctness; the producing scripts and recorded evidence must still support every claim.

**Why are archival identifiers left pending?**

A DOI and archival date must refer to an actual published record. Software metadata describes the repository and its current version without inventing a release, submission or acceptance. Development snapshots should be cited by exact commit.

**What still needs checking after importing support assets into Overleaf?**

Keep the generated paths and macro names, use the declared figure and table widths, compile the authored source, and inspect the placed assets. The support archive is not a complete manuscript; the exported source and PDF remain separate artifacts.
