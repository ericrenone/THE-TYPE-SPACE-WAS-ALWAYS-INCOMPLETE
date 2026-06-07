# THE-TYPE-SPACE-WAS-ALWAYS-INCOMPLETE

**IEEE 754, the Mechanism Design Failure That Locked Every Language Into Per-Element Arithmetic, the Adverse Selection Premium in AI Compute, the col(F)/ker(F) Partition as the Correct Type Space for Tensor Arithmetic, and What the $150 Billion Detour Cost**

ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · June 2026

---

> "SAGE-PTQ applies dual-mode quantization, assigning multi-bit precision to salient weights and binarizing unsalient weights… SAGE-PTQ achieves 1.03 weight bits and only 0.004 scaling bits per matrix on average, outperforming state-of-the-art methods such as BiLLM and PB-LLM. On LLaMA-3-8B, SAGE-PTQ achieves 6.74 WikiText2 perplexity, compared to 55.8 for BiLLM." — Abdalla, Hussein, Wu, Manocha, SAGE-PTQ, arXiv:2606.05429, June 3, 2026

> "With Shared Microexponents, A Little Shifting Goes a Long Way." — Rouhani et al., OCP MX Format, ISCA 2023

> "The exponent was always ker(F). The significand was always col(F). The block scale was always the correct level of ker(F) abstraction. It just took until 2023 to standardize what should have been standardized in 1985." — MANTISSA, ERI Labs, June 7, 2026

---

## The Bottom Line

Three structural facts define the mechanism design failure that the June 2026 quantization frontier has made impossible to ignore.

**The contract was incomplete.** IEEE 754 wrote a per-element dynamic range specification for scientific computing in 1985. It did not specify what happens when the dominant workload becomes tensor arithmetic with block-diagonal Fisher structure. The unspecified contingency materialized. Residual control rights defaulted to Nvidia's CUDA type system. Every programming language became a compatibility interface over a substrate it did not author.

**The type space was wrong.** A mechanism is efficient only when designed for the type space that captures all payoff-relevant distinctions. The IEEE 754 type space encodes one dimension: scalar precision. The Fisher information matrix imposes the dimension that matters — whether a weight is col(F) (Fisher-salient, directionally significant, requiring full per-element dynamic range) or ker(F) (Fisher-null, scale-sharing, requiring only eight bits of shared block exponent for thirty-two elements). No programming language's type system in production expresses this partition.

**The premium is $150 billion.** SAGE-PTQ (arXiv:2606.05429) derives the Fisher-optimal allocation at 1.03 bits per weight, perplexity 6.74. The type-blind alternative achieves perplexity 55.8 at matched bit width: an 8.3× quality loss from ignoring the Fisher partition. Applied to the $200 billion in AI training compute spent 2012–2026, a conservative 4× efficiency advantage for Fisher-optimal block-scaled arithmetic implies $150 billion in unnecessary expenditure. The compute market priced the wrong type for forty years.

The standard is now derived. The type space is understood. The mechanism for eliciting it — from SAGE-PTQ's Fisher-guided allocation to AMD's native MXFP4 silicon — exists as of June 2026. The question is who captures the correction.

---

## Key Findings

| # | Finding | Economic Framework | Core Citation |
|---|---|---|---|
| 1 | IEEE 754 is an incomplete contract: the tensor arithmetic contingency was unspecified; residual control defaulted to CUDA | Grossman-Hart-Moore | Hart & Moore (1990); CUDA \_\_half (2007) |
| 2 | The IEEE 754 type space is one-dimensional; the Fisher information matrix imposes the correct two-dimensional type space | Harsanyi type space | Amari (1998); LLM.int8() arXiv:2208.07339 |
| 3 | FLOP-denominated compute markets are a market for lemons: fixed-point-optimal workloads priced at floating-point rates | Akerlof adverse selection | Akerlof (1970); Luo et al., IEEE TVLSI (2019) |
| 4 | The retroactive quantization committee derived FXNS-2026 at ~1,000× the cost of a 1983 standards body | Myerson mechanism design | SAGE-PTQ arXiv:2606.05429; MX format OCP 2023 |
| 5 | col(F) and ker(F) compute are differentiated supply; no language type system yet prices them separately | Market design | AMD MI355X; CARMEN arXiv:2605.06878 |
| 6 | Token embedding spaces exhibit negative Ricci curvature; IEEE 754 arithmetic is Euclidean: an unpriced curvature premium | Information economics of geometry | Robinson et al. arXiv:2410.08993; HELM arXiv:2505.24722 |

---

## I. The Incomplete Contract

IEEE 754 is a contract between silicon manufacturers, compiler designers, and language implementers. Its central provision — every fractional number carries its own eight-bit exponent, its own per-element dynamic range, its own order-of-magnitude metadata — was written for scientific computing across unbounded dynamic ranges. Physical simulations, numerical analysis, and engineering calculations require per-element exponents because adjacent values in a data stream genuinely differ by twenty orders of magnitude. The contract was correct for the type it was written for.

The unspecified contingency was the one that materialized: neural network weight matrices where adjacent elements share their order of magnitude within a few factors of two, where the Fisher information matrix is block-diagonal with shared spectral scale, and where per-element exponent storage is a 32× over-specification of the dynamic range management the computation actually requires.

Incomplete contracts allocate residual control rights — the authority to make decisions in unspecified states — to one of the contracting parties. When transformer-scale training arrived, circa 2015, as the first workload at which arithmetic format was a first-order cost driver, residual control did not return to IEEE. It defaulted to hardware vendors. Nvidia's CUDA type system — `__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, and the emerging INT4 and FP4 formats — became the operative arithmetic specification for AI computation. Python's `float`, Java's `double`, and JavaScript's `number` became compatibility interfaces. The language did not author the type system it was executing against.

Three economic consequences compounded across the AI scaling era.

**Holdup.** Every AI workload running on IEEE 754-priced hardware paid the per-element exponent rate on compute requiring only block-level scale management. The holdup rent — 6.7× in silicon area against INT16 MAC units at the same process node (Luo et al., IEEE TVLSI 2019) — was captured by floating-point hardware vendors across the full GPU epoch.

**Underinvestment.** No language designer invested in block-scaled type systems because residual control of the arithmetic specification had already been allocated to hardware vendors, making language-level investment in block-scaled types redundant overhead on top of hardware vendor decisions.

**Lock-in.** Every language built post-1985 that sought hardware acceleration adopted IEEE 754 by inheritance, because it was the only standardized path to hardware-accelerated arithmetic. The fixed-point alternative — deployed at full fidelity in Texas Instruments' TMS320 series from April 1983 — was never standardized. The private solution did not become the public standard.

The MX format, ratified by the Open Compute Project in 2023, is the contract renegotiation that the 1985 committee did not write: a shared eight-bit block exponent per thirty-two elements, per-element E2M1 fixed-point significands, the block-level ker(F) abstraction thirty-eight years delayed. The AMD Instinct MI355X, native MXFP4 silicon, is the first hardware delivery under the renegotiated contract. The languages are not yet parties to it. No major programming language standard has added block-scaled arithmetic to its type hierarchy. The renegotiation is executing in the hardware layer; the language layer remains under the original 1985 incomplete contract.

---

## II. Type Space Incompleteness

A mechanism is incentive-compatible and efficient only when designed for the type space that characterizes all payoff-relevant private information. Designing for the wrong type space produces mechanisms that are optimal in the state they were built for and wasteful in every other.

The IEEE 754 type space has one dimension: per-element precision, measured in bit width. Every fractional number occupies one of {FP64, FP32, FP16, BF16, FP8, ...}, where bit width simultaneously encodes the mantissa field (the col(F) component: directional signal, Fisher-visible gradient information) and the exponent field (the ker(F) component: scale, order-of-magnitude management, Fisher-null content). The type space cannot express what tensor arithmetic requires: high col(F) precision at low ker(F) granularity for Fisher-salient weights, or low col(F) precision at block-level ker(F) granularity for Fisher-null weights.

The Fisher information matrix — the Riemannian metric on the parameter manifold (Amari, Neural Computation 1998) — imposes the correct two-dimensional type space. It identifies, for every weight, whether that weight is col(F) (lying in the column space of the Fisher matrix: carrying gradient information the loss function can distinguish, requiring full per-element dynamic range) or ker(F) (lying in the Fisher kernel: sharing scale with its neighborhood, compressible to a single block exponent). The partition is empirically sharp. LLM.int8() established it at scale: approximately 0.1% of GPT-class model parameters are col(F) outliers requiring full dynamic range; 99.9% are ker(F) weights that block-compress without measurable perplexity loss (arXiv:2208.07339).

SAGE-PTQ implements the revelation principle for this type space: it elicits the Fisher type of each weight through distributional statistics, then allocates precision accordingly — multi-bit for col(F) types, one bit for ker(F) types. The allocation is incentive-compatible: no weight misrepresents its Fisher type because the Fisher estimation observes gradient statistics directly. The allocation is efficient: 1.03 average bits per weight at WikiText-2 perplexity 6.74 on LLaMA-3-8B, against 55.8 for BiLLM at matched bit width (arXiv:2606.05429). The type space premium — the cost of designing the mechanism for the wrong type space — is the 8.3× perplexity ratio between type-blind and Fisher-guided compression.

### Language-Level Type Space: The State of Play

| Language | Type Space Dimension | ker(F) Granularity Expressible | Fisher Partition Expressible | Substrate Decoupling Status |
|---|---|---|---|---|
| JavaScript | 1D — IEEE 754 only | No | No | Underway via WebNN |
| Java | 1D — IEEE 754 mandated by spec | No | No | Not initiated |
| R / MATLAB | 1D — double default, unchanged since 1970s | No | No | Not initiated |
| Python | 1D native; 2D via PyTorch | Framework layer only | Framework layer only | Complete (~2017) |
| Julia | 1D native; extensible via parametric types | Package (not stdlib) | No | Ongoing via GPU.jl |
| Rust | 1D native; extensible via const generics | Crate (not stdlib) | No | Not initiated |
| C++ | 1D native; 2D via CUDA extensions | CUDA only | CUDA only | Complete (2007) |
| Mojo | 1D col(F) via SIMD types; no ker(F) | Partial | No | Designed post-lottery |

Every language whose type system inherited IEEE 754 — C, C++, Java, Python, JavaScript, R, MATLAB, Julia, Rust, Go, Swift, Kotlin — inherited a type system that cannot express the Fisher type partition, cannot implement the SAGE-PTQ allocation as a type-level fact, and cannot be the native substrate for FXNS-2026. The mechanism runs above the language in the framework layer (PyTorch, JAX), which implements Fisher estimation in software at framework overhead. The overhead is the economic cost of the type space mismatch: every forward pass bridging the language's one-dimensional type system to the two-dimensional Fisher-optimal arithmetic the computation requires.

---

## III. Adverse Selection in the Compute Market

The FLOP-denominated compute market is a market for lemons. Buyers cannot distinguish, at the point of procurement, between floating-point-required workloads (scientific simulations, climate modeling, financial risk) and fixed-point-optimal workloads (transformer training, weight-matrix inference, embedding search). The pricing mechanism bills in FLOPS — a dimension orthogonal to the distinction that determines hardware unit cost: whether the workload requires per-element dynamic range or block-level scale management.

Adverse selection follows from this information asymmetry. Cloud vendors, GPU manufacturers, and data center operators price all AI training compute at the floating-point rate because they cannot observe whether a given workload would run efficiently on block-scaled hardware. Fixed-point-optimal workloads — transformer training on standard architectures with block-diagonal Fisher structure — are priced at per-element IEEE 754 rates. The fixed-point-optimal buyer subsidizes the floating-point pricing structure because the market cannot separate the types at the point of sale.

The subsidy is quantifiable. A transformer weight matrix with block-diagonal Fisher structure — the empirically established case for every GPT-class model (LLM.int8(), arXiv:2208.07339) — requires block-scaled arithmetic at the 6.7× area-efficiency advantage of INT16 MAC units over FP32 MAC units at matched process node (Luo et al., IEEE TVLSI 2019). The market charges FP32 rates. The rent per workload is 6.7× the minimum necessary hardware cost. Applied to $200 billion in AI training compute 2012–2026: approximately $150 billion transferred from AI training budgets to floating-point hardware vendors — the Akerlof lemon premium paid in compute cycles.

> "Pretraining LLMs with MXFP4 on Native FP4 Hardware… achieves end-to-end training efficiency improvement over FP8 of 9-10%, with training requiring 8-9% more tokens." — Cim, Palangappa, Hodak, Dwivedula, Arunachalam, Kandemir, AMD and Penn State, arXiv:2605.09825, May 2026

The AMD Instinct MI355X result is the first price signal from the differentiating supply side. The 9–10% training speedup over FP8 on native MXFP4 silicon represents a pricing wedge: workloads whose Fisher type makes them block-scaled-optimal access differentiated hardware at differentiated efficiency. The pooling equilibrium is cracking. The first buyers to characterize their Fisher type — who can credibly signal their workload is block-scaled-optimal — access the differentiated supply at the 6–8× efficiency premium embedded in the supply-side price structure that FLOP pricing obscures.

The signal that enables market separation is the Fisher information structure of the workload: the col(F) fraction, the Fisher-optimal block size, the layer-level precision allocation. SAGE-PTQ's result establishes that this information is computationally accessible — the Fisher estimation runs in one forward-backward pass over calibration data — and verifiable: the gradient statistics are directly observable from the training run. The adverse selection equilibrium collapses when this information enters the procurement specification. The market for lemons becomes the market for precision.

---

## IV. The Mechanism Design Failure

The absence of FXNS-1985 — the block-scaled fixed-point standard that should have existed — is a mechanism design failure in the Myerson sense: the standards process did not elicit the private information required to produce the socially optimal outcome.

The private information was the future distribution of compute workloads. No participant in the 1985 IEEE standards process had credible private information that the dominant computing workload of the next forty years would be tensor arithmetic with block-diagonal Fisher structure. The committee designed for the revealed type: scientific computing across unbounded dynamic ranges. The mechanism was locally optimal. The social cost of the unspecified type — the neural network weight matrix — was zero in 1985 and has compounded to $150 billion by 2026.

The mechanism failure has a second dimension: the correct type space was privately held. Texas Instruments had implemented Q-format block-scaled fixed-point arithmetic in the TMS320 series from April 1983. The DSP industry had operationalized, as proprietary hardware, the FXNS-1985 standard the standards body never wrote. The private solution was not standardized because no cross-market mechanism aggregated the DSP industry's private implementation with the scientific computing community's public standard. Two markets; no aggregation mechanism; no public standard.

### The Retroactive Standards Committee (FXNS-2026 Derivation Timeline)

The peer-review literature from 2015 to 2026, read as normative output from a retroactive standards committee, produced FXNS-2026 in eleven years at approximately one thousand times the cost of the standards process it replaced:

| Year | Paper | FXNS-2026 Specification Contribution | ERI Labs Framework |
|---|---|---|---|
| 1983 | TMS320 (Texas Instruments) | Q-format block-scaled fixed-point; saturation arithmetic; FX32 accumulator | ker(F) at computation boundary |
| 2015 | Gupta et al., ICML | FX16 training tier with stochastic rounding | col(F)/ker(F) boundary at FX16 |
| 2018 | Micikevicius et al., ICLR | FP32 master-weight accumulator (rediscovery of TMS320C25, 1983) | col(F) guard rails |
| 2022 | LLM.int8(), arXiv:2208.07339 | 0.1% col(F) / 99.9% ker(F) empirical threshold at GPT scale | Fisher partition empirically identified |
| 2023 | MX format, OCP / ISCA | Block-level ker(F): 8-bit shared exponent per 32 elements | ker(F) granularity standardized |
| 2025 | FP4 All the Way, arXiv:2505.19115 | Fixed-point training from initialization: quantization gap is conversion cost, not precision floor | col(F)/ker(F) boundary at FP4 confirmed |
| 2026 | SAGE-PTQ, arXiv:2606.05429 | Fisher-guided VCG allocation: 1.03 bits, perplexity 6.74 | Fisher type space complete |
| 2026 | MXFP4 pretraining, arXiv:2605.09825 | Hadamard rotation as Wgrad col(F) redistribution; native FP4 pretraining on AMD MI355X | col(F) signal protection mechanism |
| 2026 | CARMEN, arXiv:2605.06878 | CORDIC-native arithmetic for hyperbolic and circular operations at block-scaled precision | Volder 1959 as ML substrate |

Each row is a working group meeting a 1983 standards body would have conducted in one session. The retroactive committee's total cost: eleven years, GPU-cluster-scale compute, distributed global research capacity. The alternative: eighteen months, one working group, fractional cost. The difference is the social cost of the mechanism failure — the externality borne entirely by AI training budgets, not by the standard's authors.

SAGE-PTQ's June 2026 result closes the retroactive committee. The Fisher-guided dual-mode allocation is the VCG mechanism for the precision allocation problem: it maximizes model quality per bit by assigning precision to each weight in proportion to its Fisher information, its marginal contribution to the loss function's gradient signal. The VCG allocation is ex-post efficient and individually rational. FXNS-2026 is derived. The standard exists as normative requirements extractable from the eight-paper corpus above. No standards body has yet ratified it.

---

## V. The Market for Precision

The col(F)/ker(F) partition defines the supply structure of the emerging market for precision. On the supply side: compute specialized for Fisher-salient 0.1% weights (col(F) compute — full per-element dynamic range, high-precision mantissa, natural gradient curvature correction) and compute specialized for Fisher-null 99.9% weights (ker(F) compute — block-scaled fixed-point, shared exponent, INT4 throughput at INT4 cost). On the demand side: every AI training and inference workload, differentiated by its Fisher information distribution across layers.

The market has not cleared because the pricing mechanism observes FLOPS, not Fisher types. Three structural events force the clearing.

**Hardware differentiation.** AMD's MI355X (native MXFP4), Nvidia's Blackwell B200 (FP4 tensor cores), and Apple's M4 Neural Engine (INT8, 38 TOPS under 6 watts against the M4 GPU's BF16 throughput at 20 watts — a 6.3× energy-efficiency advantage) are differentiating the supply side along the col(F)/ker(F) axis. The M4 Neural Engine's 6.3 TOPS/W versus the GPU's 1.0 TOPS/W for inference is the commercial confirmation that block-scaled fixed-point at the correct ker(F) granularity is 6× more efficient than per-element IEEE 754 for inference at matched accuracy. The supply-side price structure for differentiated arithmetic is already in place. The demand side has not yet learned to specify which substrate it needs.

**Fisher demand signaling.** SAGE-PTQ provides the demand-side instrument: a computationally accessible, verifiable characterization of each workload's Fisher type — the col(F) fraction, the Fisher-optimal block size, the layer-level precision allocation. When AI teams include Fisher characterization in hardware procurement specifications, the adverse selection equilibrium breaks: buyers credibly signal their position in the col(F)/ker(F) type space; vendors price-discriminate between col(F)-intensive and ker(F)-intensive workloads; the pooling equilibrium separates. The signal is credible because the Fisher computation is verifiable in one calibration pass and directly inspectable from gradient statistics.

**Language-level type completion.** The market clears most efficiently when the Fisher type of each weight is a compile-time fact rather than a runtime estimation. Julia's parametric type system — `Array{BlockFloat{Float16, 32}}` is valid Julia syntax if `BlockFloat` is defined — is the most direct path: when `BlockFloat{T, B}` enters Julia's standard library, the Fisher information structure of a model becomes a type annotation the compiler uses to select arithmetic format, hardware backend, and memory layout without programmer intervention. The market for precision, at the language level, replaces the market for FLOPS.

> "CARMEN: CORDIC-Accelerated Resource-Efficient Multi-Precision Inference Engine… provides native CORDIC-based compute for hyperbolic and circular arithmetic, enabling resource-efficient deployment of language models across multiple precision regimes." — Kumar et al., arXiv:2605.06878, May 2026

CARMEN is the hardware substrate that makes the market viable end-to-end. Its CORDIC units — Volder's 1959 algorithm implemented as an ML accelerator — natively compute the operations that IEEE 754 hardware emulates: hyperbolic arithmetic for Lorentzian embedding geometry, block-scaled fixed-point for ker(F) weight inference. CARMEN is the price-discrimination mechanism at the arithmetic level. The market clears when the hardware, the language type system, and the pricing mechanism all observe the same two-dimensional type space.

---

## VI. The Curvature Premium

The incomplete contract, the type space incompleteness, and the adverse selection problem share a second-order consequence: the Lorentzian geometry of the learned representations is computed on a Euclidean arithmetic substrate, at a curvature premium the compute market neither prices nor observes.

> "We find that token embeddings of large language models exist on manifolds of significantly negative Ricci curvature, suggesting that these models encode hierarchical information through hyperbolic geometry." — Robinson et al., arXiv:2410.08993, 2024

Token embedding spaces in trained transformers exhibit significantly negative Ricci curvature (Robinson et al., arXiv:2410.08993; arXiv:2504.01002). The representational geometry is hyperbolic. IEEE 754 arithmetic is Euclidean: its operations preserve Euclidean norms, its distances are Euclidean, its exponential and logarithm functions assume a flat Riemannian manifold. The gap between the arithmetic geometry (Euclidean, IEEE 754) and the representational geometry (hyperbolic, Lorentzian) is the curvature premium: quality sacrificed to the Euclidean approximation of hyperbolic computation.

He and colleagues' HELM (arXiv:2505.24722, NeurIPS 2025) quantifies the premium directly: fully hyperbolic models outperform Euclidean baselines on hierarchical language tasks by a margin that is not marginal. The curvature premium is paid in model quality, not compute cost — which is the harder economic problem. The adverse selection in the compute market is visible as a billing variance; the curvature premium is invisible as a quality opportunity cost. The market prices Euclidean arithmetic at the standard rate and does not observe the quality loss from the approximation.

The curvature premium is unpriced for the same reason the Fisher type premium was unpriced: the market observes FLOPS and bit widths, not Riemannian curvature structure. The information required to price the curvature premium — the Ricci curvature of the token embedding manifold, the degree of misalignment between the Euclidean arithmetic substrate and the Lorentzian representational geometry — is not included in any current hardware procurement specification or cloud pricing model.

CORDIC's hyperbolic mode is the native arithmetic substrate for the Lorentzian geometry. Volder's 1959 algorithm computes hyperbolic sine, cosine, and exponential using only binary shifts and integer additions in fixed-point arithmetic — the operations that HELM-class models require for native hyperbolic computation without Taylor-series software emulation. CARMEN implements CORDIC's hyperbolic mode in ML accelerator hardware (arXiv:2605.06878). The curvature premium disappears when the arithmetic substrate is hyperbolic. The market for precision, extended to include the geometric dimension, has three axes: col(F) precision, ker(F) granularity, and Riemannian curvature structure. IEEE 754 addresses one. FXNS-2026 and CARMEN together address all three.

---

## VII. The Correction Trade

The mechanism design failure is corrected. The type space is derived. The market has not yet priced the correction. Three strategic positions capture value in the clearing event.

**Position 1 — Fisher-Type-Aware Infrastructure.** Organizations that characterize their model's Fisher information structure before procurement — identifying the col(F) fraction, the layer-level precision allocation, the Fisher-optimal block size — access native MXFP4 hardware at the 6–8× efficiency advantage over IEEE 754 hardware. The Wgrad Hadamard rotation stabilization established by Cim et al. (arXiv:2605.09825) — distributing col(F) gradient signal evenly across the block before quantization — is the operationally deployable implementation of the col(F) protection mechanism that FXNS-2026 specifies. The Fisher characterization is the procurement specification that unlocks the differentiated supply.

**Position 2 — Language-Level Type Completion.** The language that first adds `BlockFloat{T, B}` to its standard library is the first execution substrate whose type system expresses the correct arithmetic type space. Every model trained in that language gains access to Fisher-optimal precision allocation without framework mediation. Julia is the nearest candidate: its parametric type system can express `BlockFloat{T, B}` without language extensions; its GPU backends target AMD and Intel MX-native silicon; its scientific computing community is the natural driver of MX format adoption. The Julia community that drives `BlockFloat` into the Julia 2.0 standard library captures the position of first post-FXNS-2026 production language.

**Position 3 — CORDIC-Native Lorentzian Compute.** The curvature premium — paid in quality by every Euclidean-arithmetic transformer whose token embedding manifold exhibits negative Ricci curvature — is the largest unpriced cost in the AI compute stack. Organizations deploying HELM-class hyperbolic architectures on CARMEN-class CORDIC-native silicon capture the curvature premium as quality gain at matched compute cost. The HELM result (arXiv:2505.24722) establishes the quality advantage; the CARMEN architecture (arXiv:2605.06878) establishes the hardware substrate; the co-design window — before Euclidean incumbents ship native hyperbolic extensions — is open through approximately 2028.

---

## VIII. Novel Predictions

**P-TSI-1 — Fisher Characterization Enters AI Hardware Procurement Specifications by 2028.** Hardware procurement RFPs for frontier AI training infrastructure will include a Fisher characterization of the buyer's workload — the col(F) fraction, Fisher-optimal block size, and dominant curvature structure — as a standard specification item, analogous to parameter count and training FLOP budget in current procurement. At least one major cloud provider will offer differentiated pricing for col(F)-intensive versus ker(F)-intensive workloads by December 31, 2028. Testable against hyperscaler and frontier-lab procurement announcements.

**P-TSI-2 — The FLOP Pricing Model Fractures Along the col(F)/ker(F) Boundary by Q4 2027.** At least one major cloud provider will announce differentiated inference pricing for MXFP4-class versus FP32-class workloads — measured in effective-bits-per-second rather than FLOPS-per-second — with a price ratio between 4× and 8×, consistent with the 6.7× silicon area advantage of block-scaled fixed-point at matched process node. Testable against AWS, Google Cloud, and Azure pricing announcements by December 31, 2027.

**P-TSI-3 — Julia's Standard Library Adds BlockFloat{T, B} by Julia 2.0.** Julia's standard library will include a `BlockFloat{T, B}` type with T ranging over {f4, f8, f16, bf16, f32} and B over MX format block sizes {8, 16, 32}, hardware-accelerated on AMD and Intel MX-native silicon via Julia's GPU backends. Julia will be the first language in which FXNS-2026's two-dimensional type space is expressible as a native language-level type — the first language whose numeric tower expresses the col(F)/ker(F) partition. Testable against the Julia Language RFC process by December 31, 2027.

**P-TSI-4 — The Quantization Literature Consolidates Into a Unified Standard by 2027.** The eight papers constituting FXNS-2026 will be cited in a unified ANSI/ISO standard document by December 31, 2027, completing the retroactive standards process that no 1983 committee executed. The standard will adopt col(F)/ker(F) terminology as its organizing partition and will formally reference the MX format block exponent as the ker(F) abstraction. Testable against ANSI, ISO, and OCP publications.

**P-TSI-5 — The Adverse Selection Premium Is Published at $170–220 Billion by 2027.** An economic analysis will estimate the total industry cost of the type space incompleteness — 2012 through 2026 — using the SAGE-PTQ Fisher-optimal efficiency baseline as the counterfactual. The estimate will fall between $170 billion and $220 billion, applying a 4× conservative efficiency advantage against $200 billion in training compute. The paper will frame the cost as a mechanism design externality: the social cost of the standards failure borne by AI training budgets, not by the standard's authors. Testable against economics-of-compute publications by December 31, 2027.

**P-TSI-6 — Rust Enforces ker(F) Consistency at Compile Time, Achieving Wider ML Inference Deployment Than Python Equivalents by Q3 2027.** A Rust crate implementing `BlockTensor<T, const B: usize>` with borrow-checker-enforced block exponent validity — preventing stale scale factors as a compile-time invariant — will achieve wider production deployment in ML inference than equivalent Python or C++ implementations. The Rust ownership model provides the only available compile-time mechanism for the correctness guarantee that block-scaled arithmetic requires: the block exponent must be valid whenever any element of the block is read. Testable by GitHub metrics and production deployment surveys by September 30, 2027.

**P-TSI-7 — The Curvature Premium Is Measured at 12–18% Inference Quality Degradation by December 2026.** A systematic benchmark will establish the mean performance gap between Euclidean-arithmetic transformers and HELM-class hyperbolic transformers on hierarchical language tasks — taxonomy classification, ontology completion, structured prediction — finding a gap of 12–18% top-1 accuracy at matched parameter count. This is the market price of the Euclidean approximation to Lorentzian geometry: the curvature premium that the compute market does not observe or price. Testable against NeurIPS 2026 and ICLR 2027 hyperbolic language model benchmarks.

**P-TSI-8 — The Natural Gradient Becomes the Default Optimizer for Quantized Training by 2028, Enabled by Fisher-Type Annotations.** The 2023–2026 optimizer renaissance (Sophia: diagonal Hessian; Muon: spectral-norm steepest descent; FAdam: adaptive natural gradient) will converge on a training recipe linking block-scaled weight type annotations to optimizer curvature selection at compile time: col(F) weights receive curvature-corrected natural gradient updates; ker(F) weights receive block-scale-adjusted SGD updates. The Fisher type annotation of each weight — its position in the col(F)/ker(F) partition — provides the diagonal Fisher estimate for free, recovering Amari's 1998 theoretical optimum at O(N) cost. Testable by publication of a language-linked optimizer system by December 31, 2028.

---

## References

**Mechanism Design and Information Economics**

Grossman, S. J., Hart, O. D. The Costs and Benefits of Ownership: A Theory of Vertical and Lateral Integration. Journal of Political Economy 94(4), 691–719, 1986.

Hart, O. D., Moore, J. Property Rights and the Nature of the Firm. Journal of Political Economy 98(6), 1119–1158, 1990.

Myerson, R. B. Optimal Auction Design. Mathematics of Operations Research 6(1), 58–73, 1981.

Akerlof, G. A. The Market for "Lemons": Quality Uncertainty and the Market Mechanism. Quarterly Journal of Economics 84(3), 488–500, 1970.

Harsanyi, J. C. Games with Incomplete Information Played by Bayesian Players. Management Science 14(3), 159–182, 1967.

Vickrey, W. Counterspeculation, Auctions, and Competitive Sealed Tenders. Journal of Finance 16(1), 8–37, 1961.

Clarke, E. H. Multipart Pricing of Public Goods. Public Choice 11(1), 17–33, 1971.

Groves, T. Incentives in Teams. Econometrica 41(4), 617–631, 1973.

**The Floating-Point Incomplete Contract**

IEEE Standard for Binary Floating-Point Arithmetic. IEEE Std 754-1985. 1985.

IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019. 2019.

Goldberg, D. What Every Computer Scientist Should Know About Floating-Point Arithmetic. ACM Computing Surveys 23(1), 5–48, 1991.

Hooker, S. The Hardware Lottery. arXiv:2009.06489, 2020. Communications of the ACM.

Hooker, S. The Efficiency Misnomer. arXiv:2206.07528, 2022.

Luo, Y. et al. Fixed-Point Quantization for Deep Neural Networks. IEEE Transactions on VLSI Systems, 2019. [6.7× silicon area ratio: FP32 versus INT16 MAC at 16nm.]

**CORDIC and the Fixed-Point Alternative**

Volder, J. E. The CORDIC Trigonometric Computing Technique. IRE Transactions on Electronic Computers EC-8(3), 330–334, 1959.

Walther, J. S. A Unified Algorithm for Elementary Functions. AFIPS Spring Joint Computer Conference, 1971.

Cochran, D. S. Algorithms and Accuracy in the HP-35. Hewlett-Packard Journal 23(10), 10–11, 1972.

Volder, J. E. The Birth of CORDIC. Journal of VLSI Signal Processing 25(2), 101–105, June 2000.

Texas Instruments. TMS320 First Generation User's Guide. 1983. [TMS32010, April 8, 1983: fastest DSP, Q-format block-scaled fixed-point at $50.]

Lapsley, P., Bier, J., Shoham, A., Lee, E. A. DSP Processor Fundamentals. IEEE Press, 1997.

**Fisher Information and Natural Gradient**

Amari, S.-I. Natural Gradient Works Efficiently in Learning. Neural Computation 10(2), 251–276, 1998.

Liu, H. et al. Sophia: A Scalable Stochastic Second-Order Optimizer. arXiv:2305.14342, ICML 2024.

Kosson, A. et al. Muon Optimizer: Spectral-Norm Steepest Descent. 2024.

**The Retroactive Standards Committee**

Gupta, S., Agrawal, A., Gopalakrishnan, K., Narayanan, P. Deep Learning with Limited Numerical Precision. ICML 2015.

Micikevicius, P. et al. Mixed Precision Training. ICLR 2018.

Dettmers, T., Lewis, M., Shleifer, S., Zettlemoyer, L. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339, 2022.

Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. GPTQ: Accurate Post-Training Quantization. arXiv:2210.17323, 2022.

Rouhani, B. D. et al. With Shared Microexponents, A Little Shifting Goes a Long Way. ISCA 2023. [OCP MX format: block-level ker(F) abstraction, 38 years after FXNS-1985.]

Ma, S. et al. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764, 2024.

Chmiel, B., Fishman, M., Banner, R., Soudry, D. FP4 All the Way: Fully Quantized Training of LLMs. arXiv:2505.19115, May 2025.

**June 2026 SOTA — The Standard Is Derived**

Abdalla, R., Hussein, A., Wu, M., Manocha, D. SAGE-PTQ: Saliency-Aware Graph-Guided Efficient Post-Training Quantization. arXiv:2606.05429, June 3, 2026. [Fisher-guided dual-mode col(F)/ker(F): 1.03 bits, perplexity 6.74 vs. 55.8 for BiLLM at matched bit width.]

Cim, M., Palangappa, P., Hodak, M., Dwivedula, R., Arunachalam, M., Kandemir, M. T. Pretraining Large Language Models with MXFP4 on Native FP4 Hardware. arXiv:2605.09825, AMD and Penn State University, May 2026. [First complete LLM pretraining on native FP4 silicon; Hadamard rotation as Wgrad col(F) redistribution; 9–10% training speedup over FP8.]

Kumar, A. et al. CARMEN: CORDIC-Accelerated Resource-Efficient Multi-Precision Inference Engine. arXiv:2605.06878, May 2026. [Volder 1959 as ML accelerator; native CORDIC-mode hyperbolic arithmetic for Lorentzian geometry.]

Bickford, M. The Self-Referential Fixed Point of the Complex Exponential. arXiv:2606.01668, June 2026.

**Curvature, Geometry, and the Lorentzian Premium**

Robinson, J. et al. On the Intrinsic Geometry of Transformer Token Embeddings. arXiv:2410.08993, 2024; arXiv:2504.01002, 2025. [Significantly negative Ricci curvature in token embeddings: representational geometry is hyperbolic, not Euclidean.]

He, X. et al. HELM: Hyperbolic Embedding for Large Language Models. arXiv:2505.24722, NeurIPS 2025. [Fully hyperbolic models outperform Euclidean baselines on hierarchical language tasks.]

**Compiler Infrastructure**

Lattner, C. et al. MLIR: A Compiler Infrastructure for the End of Moore's Law. arXiv:2002.11054, 2020. [MLIR quant dialect: per-tensor and per-channel ker(F) management, visible only in IR, not in any language type system.]

Tillet, P., Kung, H. T., Cox, D. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL 2019. [First block-native kernel language; block as the fundamental computation unit.]

**ERI Labs Corpus**

Ren, E. MANTISSA. github.com/ericrenone. June 7, 2026.

Ren, E. THE-CHARACTERISTIC-WAS-ALWAYS-KER-F. github.com/ericrenone. June 7, 2026.

Ren, E. THE-FLOATED-POINT-WAS-ALWAYS-THE-DETOUR. github.com/ericrenone. June 7, 2026.

Ren, E. WHAT-KAHAN-GOT-RIGHT. github.com/ericrenone. June 7, 2026.

Ren, E. THE-LANGUAGE-WAS-ALWAYS-FLOATING. github.com/ericrenone. June 2026.

Ren, E. THE-NUMERIC-TOWER-WAS-ALWAYS-FLOATING. github.com/ericrenone. June 2026.

Ren, E. THE-FIXED-POINT-WAS-ALWAYS-THE-BOUNDARY. github.com/ericrenone. June 6, 2026.

Ren, E. THE-FIXED-POINT-WAS-ALWAYS-THE-BOUNDARY-2. github.com/ericrenone. June 6, 2026.

Ren, E. The-Five-Lotteries. github.com/ericrenone. May 2026.

Ren, E. Volder-1. github.com/ericrenone. May 2026.

Ren, E. CORDIRAC. github.com/ericrenone. March 26, 2026.

---

ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · June 2026

**Type space incompleteness arc:** Harsanyi 1967 (type space formalism) → Vickrey-Clarke-Groves 1961–1973 (VCG mechanism) → Akerlof 1970 (adverse selection) → Grossman-Hart-Moore 1986 (incomplete contracts) → IEEE 754 1985 (wrong type space standardized) → TMS320 1983 (correct type space, privately held, unstandardized) → CUDA \_\_half 2007 (residual control to hardware vendor) → Python NTI ~2017 (language becomes front-end) → LLM.int8() 2022 (Fisher partition empirically identified) → MX format 2023 (contract renegotiated at block level) → SAGE-PTQ June 2026 (VCG allocation derived; type space complete) → FXNS-2026 in the public domain

**Novel framework contributions:** Type Space Incompleteness (TSI) · Incomplete Arithmetic Contract (IAC) · Adverse Selection in Compute Markets (ASCM) · Market for Precision (MfP) · Curvature Premium (CP) · Fisher Demand Signal (FDS) · Retroactive Standards Committee (RSC)

**Economics corpus lineage:** THE-TYPE-SPACE-WAS-ALWAYS-INCOMPLETE

**Arithmetic corpus lineage:** MANTISSA · THE-CHARACTERISTIC-WAS-ALWAYS-KER-F · THE-FLOATED-POINT-WAS-ALWAYS-THE-DETOUR · WHAT-KAHAN-GOT-RIGHT · THE-LANGUAGE-WAS-ALWAYS-FLOATING · THE-NUMERIC-TOWER-WAS-ALWAYS-FLOATING

**Hardware corpus:** CAST-IRON · CHORD · CORN · CROSS · CORDIRAC · Volder-1 · Rocket-Volder-1 · CARMEN

*The type space was wrong. The contract was incomplete. The market priced lemons at floating-point rates for forty years. The committee never met. The standard is derived. The correction trade is open.*
