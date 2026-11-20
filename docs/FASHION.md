# Fashion-MNIST transfer

This comparison tests whether the reward outcomes transfer to a second image dataset without selecting new reward parameters.

## Fixed experiment

The declared cohort contains 6 complete Flower runs: seeds 42, 43, 44, the fixed Sendros mixture and label flipping at 30% attackers, each for 20 rounds. Every compared run uses CUDA. The fixed mixture transfers the source comparison; label flip at the largest declared fraction tests direct training-data poisoning. Both are declared before inspecting any Fashion-MNIST run.

Reuse every MNIST pilot-selected parameter without retuning on Fashion-MNIST or evaluation seeds. No Fashion-MNIST result changes the clipping-free training protocol, thresholds, reputation decay or penalty patience. The reference-only and reference-plus-reputation stages use those same frozen parameters and independent state.

The model, optimizer and partition protocol match the MNIST experiment: 30 registered clients, 10 sampled per round, Dirichlet concentration 0.5, 5 local epochs, batch size 32, learning rate 0.01 and momentum 0.9. Fashion-MNIST normalization uses mean 0.286 and standard deviation 0.353. Both full official splits are used. Label flipping maps class y to 9-y. The class semantics differ from digits, so an identical poisoning transformation need not cause the same accuracy loss.

All reward rules consume the same submitted updates and fixed FedAvg trajectory in each run. Shadow rewards cannot improve the measured model accuracy. The figures and accuracy table therefore display one shared model result per condition.

## Reproduction

```bash
python scripts/download_data.py --dataset fashion_mnist
# Publish the experiment declaration and selected method before training.
python scripts/run_grid.py --plan configs/fashion_plan.json
python scripts/run_grid.py --plan configs/fashion_plan.json --no-run
python scripts/summarize_grid.py --plan configs/fashion_plan.json
python scripts/document_fashion.py
python scripts/make_macros.py
python scripts/make_tables.py
python scripts/make_figures.py
```

The runner uses the same publication gate as the main evaluation. Its output and persistent elapsed-time clock live in `results/fashion/`. Every raw matrix is retained and hash-checked during validation. The separate download command records checksums; training verifies the complete prepared input set without network access.

The summary requires complete unique coverage, identical resolved protocols and library versions, the published frozen method, and source digests verified against reachable published training commits. It reconstructs every integer ledger and reads the same accuracy in the round and participation logs. Recorded-only analysis needs no raw matrices or dataset download. Successful training time is reported separately from elapsed time including restarts.

## Uncertainty and interpretation

Each interval uses 20000 fixed-seed bootstrap resamples of the 3 complete seed observations, with sample standard deviation and 95% percentile limits. Pooled values average the two fixed conditions within each seed before resampling. Undefined reward outcomes retain their counts and invalidate the corresponding complete-cohort estimate.

The descriptive signed-rank comparison has 6 condition-seed pairs, with Holm correction over both primary costs and all baseline contrasts. A separate descriptive test uses seed-level paired averages; its smallest two-sided p-value with all nonzero differences is 0.25. Three seeds give coarse, potentially unstable intervals. Conditions sharing seeds are dependent. Paired tests are descriptive and do not establish population-wide security or generalization to other datasets. A negative or inconclusive transfer is retained as a result; no additional seed or attack is selected after observing it.

The comparison and accuracy figures use single-column width; the tables are sized for double-column placement. Tables state the common device, seed uncertainty and undefined estimates. The `fashion` record in `results/summary.json` supplies every empirical macro, table and figure.

## Input checksums

| File | SHA-256 |
|---|---|
| t10k-images-idx3-ubyte | `5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b` |
| t10k-images-idx3-ubyte.gz | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| t10k-labels-idx1-ubyte | `0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34` |
| t10k-labels-idx1-ubyte.gz | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |
| train-images-idx3-ubyte | `c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888` |
| train-images-idx3-ubyte.gz | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| train-labels-idx1-ubyte | `bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9` |
| train-labels-idx1-ubyte.gz | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |

Empirical transfer results are pending the published method freeze and complete training cohort.
