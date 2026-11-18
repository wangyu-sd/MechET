# PR116 execution snapshot

The existing Stage-II policy samples four routes per training target. The ReFIT collector adds up to two matched credit continuations per original tree. Each training arm uses the same fixed 2,048 draws over 64 optimizer updates. The 512-target clean validation cohort excludes all canonical training targets. The 256-target prefix is used for the interim checks at updates 8 and 16. Comparisons for selecting the checkpoint use all 512 at initialization and updates 32 and 64.

A committed reaction can be scored after its accepted FINISH. The adapter reconstructs the complete mapped species inventory, including imported participants and explicit hydrogens, then checks applicability before calling the frozen forward generator. The route learner uses the score as bounded progress and rolls it back at terminal rejection. A preference requires two different sampled FLOW actions from the same observation, each with an eligible independent continuation. Both scores must be available and differ by at least 0.25. The learner uses one optimizer step for the existing ReFIT objective including KL plus 0.1 times the DPO loss. An arm without preferences retains its original policy update.

The execution scripts are archived from the shared Ceph deployment. The source files overlay the existing PR112 native runtime. The independent R arm was frozen before later teacher diagnostics, with contract SHA-256 0671327714bcb787d74799fa5499150360e55d6c2cab40ae91583e563ecafb99. Its input and code bindings passed the production verifier. The model registry and private task configuration stay on the cluster.

At 23:17 CST on November 18, the A100 engineering run had completed 8 targets and the H20 native run had published outcomes for 64 of 512 targets. Stock closure and reference-verified closure on the completed H20 targets were 14/64 and 1/64. The teacher-free R arm was registered with a launch dependency on completion and release of this baseline allocation. The forward-reward and Joint arms require the official checkpoint archive, a forward compatibility run and a measured GPU deployment profile. No optimizer updates from those arms were observed.

- [Committed reactant inventory](source/src/mechet/route_joint_inventory.py)
- [Forward replay scorer](source/scripts/pr116_flow_teacher.py)
- [Reward and preference construction](source/src/mechet/route_joint_forward.py)
- [Batched evidence and cache](source/src/mechet/route_joint_phase.py)
- [Joint objective in the learner](source/src/mechet/external_route_rl.py)
- [Fixed root stream and validation protocol](source/src/mechet/route_joint_schedule.py)
- [Public checkpoint verification and binding](execution/bind_teacher.py)

The frozen 2,048 training draws share zero canonical targets and zero complete reference routes with clean validation. Eight canonical reaction signatures occur in both cohorts, affecting six validation targets. Forty-two source patents are shared, affecting 17 validation targets. The fixed validation cohort therefore also supports a 506-target reaction-disjoint analysis and a 492-target reaction-and-patent-disjoint analysis. The 512-target primary denominator stays fixed. These counts compare the actual online-training draws to the clean validation cohort. [Counts and input digests](cohort_overlap.json)
