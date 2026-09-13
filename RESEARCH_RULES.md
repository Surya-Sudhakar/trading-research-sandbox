# Frozen Research Rules — Stage 1

1. Primary broker/data environment = IC Markets MT5.
2. Raw broker data is immutable.
3. UTC is the canonical internal timestamp.
4. Missing candles are never fabricated silently.
5. Every dataset must have provenance.
6. Research results must eventually identify the exact dataset used.
7. Strategy development must eventually use separate Discovery, Validation, and Final Test datasets.
8. Final Test data must eventually remain inaccessible during strategy development.
9. Failed experiments must eventually remain recorded rather than deleted.
10. The purpose of the sandbox is to evaluate hypotheses, not manufacture profitable backtests.
11. Every scientific experiment must have a registered question and hypothesis.
12. Success criteria must be frozen before execution.
13. Approved experiment specifications are immutable.
14. Material rule changes require a new experiment.
15. Duplicate experiments must be detected.
16. Replication runs must be distinguished from new experiments.
17. Failed experiments remain permanently recorded.
18. Results and verdicts are separate records.
19. Every experiment must identify exact dataset provenance and code/execution versions.
20. Research history must not depend on ChatGPT conversation history.
21. The research journal is append-only and tamper-evident.
22. A profitable result is not automatically a valid result.
23. Changing a rule because an experiment failed must remain visible through experiment ancestry.
24. Research validity and profitability are separate.
25. Every approved experiment requires a pre-run methodological audit.
26. Profitability must never improve a pre-run audit classification.
27. Explicit look-ahead is prohibited.
28. Known independent-test contamination invalidates independence claims.
29. Result-driven modifications remain part of the same discovery family.
30. Parameter searching must remain visible.
31. Near-duplicate experiments must remain identifiable.
32. High-risk research requires explicit override.
33. Hard methodological blocks cannot be overridden.
34. Audit findings are permanent research records.
35. Auditor configuration and version must be recorded.
36. A strategy may satisfy performance criteria while still being scientifically high-risk.
37. Research risk scores are warnings, not probabilities that a strategy is false.
38. Discovery may be exploratory but must remain fully recorded.
39. Discovery evidence cannot later be represented as independent validation.
40. Exploration is not prohibited merely because previous experiments failed.
41. Every experiment must identify the question it answers.
42. Every experiment must record why it exists and what PASS/FAIL would teach us.
43. Research branches may be explored, but their origin must remain traceable.
44. Closed questions cannot be silently recreated.
45. Legitimate reopening requires explicit new evidence/reason.
46. Candidate strategies must be frozen before validation.
47. Material changes to a frozen candidate create a new discovery version.
48. Validation failure cannot be repaired using the same validation data while retaining an independence claim.
49. Exhausting one strategy family does not reset or close the root research question.
50. Discovery search burden must remain visible.
51. The sandbox organizes research but does not invent the next trading hypothesis.
52. `What next?` may only identify registered unresolved work or request a human research decision.
53. Discovery, Validation and Final Test are distinct data-access contexts.
54. Discovery cannot inspect Validation or Final-Test market data.
55. Validation requires a frozen candidate and uncontaminated validation data.
56. Final testing requires successful validation and explicit authorization.
57. Final-test data must not be used for exploratory analysis.
58. Data exposure is permanent historical information and cannot be reset.
59. Validation contamination is candidate-lineage aware.
60. Modified candidates return to Discovery.
61. Validation evidence does not automatically transfer to modified candidate versions.
62. Final-test execution is a one-way scientific event.
63. Failed final tests remain permanent evidence.
64. Previously explored data cannot later be relabeled as unseen evidence.
65. Dataset access must be audited.
66. Integrity checks must avoid leaking strategy-relevant information.
67. Time-series partitions are chronological by default.
68. Derived candles must not cross research partition boundaries.
69. Warm-up access must be distinguished from evaluation access.
70. The sandbox provides application-level research isolation, not adversarial operating-system security.
71. Observed performance and statistical evidence are different concepts.
72. A displayed win rate is an estimate, not the true win probability.
73. Sample size must be interpreted through uncertainty, not a universal magic threshold.
74. Statistical evidence must derive from immutable ledgers where possible.
75. Confidence-interval methods must be identified explicitly.
76. Resampling methods and random seeds must be recorded.
77. Resampled trade sequences are stress tests, not predictions of future markets.
78. Temporal concentration must remain visible.
79. Symbol-level results must remain visible beneath pooled results.
80. Parameter sensitivity may analyze registered variants but must not generate optimized variants.
81. Cost sensitivity must not silently replace realistic baseline assumptions.
82. Discovery search burden must accompany discovery evidence.
83. Exploratory subgroup findings are not independent confirmation.
84. Statistical criteria defined after seeing results cannot be represented as pre-registered criteria.
85. Validation statistics must use frozen evaluation specifications.
86. Final-Test statistics must remain limited to authorized evaluation.
87. Robustness is multi-dimensional and should not be reduced to one opaque score.
88. Statistical warnings do not automatically reject Discovery research.
89. Strong statistical evidence does not repair contaminated validation or methodological invalidity.
90. Stage 7 must not become a strategy-generation or parameter-optimization engine.

The sandbox contains no live execution, strategy optimization, machine learning, or automatic strategy generation.
