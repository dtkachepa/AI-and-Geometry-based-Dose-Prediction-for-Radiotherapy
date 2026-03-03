  ## Potential Improvements

  1. DVH-Differentiable Loss (High impact)
  Directly optimize the evaluation metric (DVS/DVHS) instead of just L1. Using a soft differentiable DVH loss would target what the paper actually measures.

  2. PTV Boundary-Aware Loss (Addresses stated failure case)
  Add a boundary/edge loss term specifically at PTV borders to prevent dose from bleeding outside for large PTVs.

  3. Mamba/SSM Blocks in the Encoder (Architecture)
  The repo already has Model_MTASP variants — replacing or augmenting MSFE with Mamba blocks would capture long-range 3D dependencies that convolutions struggle with.

  4. Deformable Convolutions in MSFE (Architecture)
  Replace the fixed {3,5,7} kernels with deformable 3D convolutions that can adaptively sample based on anatomy geometry.

  5. OAR-Free Inference (Stated future work)
  The paper explicitly says they want to remove OAR as a required input — implementing OAR prediction as an auxiliary task or using learned OAR estimation.

  6. Dynamic Loss Weighting (Training)
  Make w_PTV and w_OAR dynamically scheduled during training (e.g., curriculum-style, increasing PTV weight later in training).


  ---                                                                                                                                                                                         
  ## What was built                                                                                                                                                                                
                                                                                                                                                                                              
  utils/mamba_3d.py — TriDirectionalMamba                                                                                                                                                       
                                                                                                                                                                                              
  Implements eq. 10 from the proposal. For input (B, C, D, H, W):

  ┌──────────┬───────────────┬─────────────┐
  │   Scan   │  Reshape to   │ Scans along │
  ├──────────┼───────────────┼─────────────┤
  │ Axial    │ (B·H·W, D, C) │ depth z     │
  ├──────────┼───────────────┼─────────────┤
  │ Coronal  │ (B·D·W, H, C) │ height y    │
  ├──────────┼───────────────┼─────────────┤
  │ Sagittal │ (B·D·H, W, C) │ width x     │
  └──────────┴───────────────┴─────────────┘

  Fused via softmax(α) · [F_axial, F_coronal, F_sagittal].

  models/model_mamba_ssm.py

  Two new model classes, both preserving the full RANDose structure (MSFE, CSA, PI, AF):

  - Model_RANDose_MambaA — Strategy A (eq. 11): MSFE → TriMamba → CSSE3D
  - Model_RANDose_MambaB — Strategy B (eq. 12): CSSE3D(MSFE(x)) + γ·TriMamba(MSFE(x))

  train.py

  Two new --model options: Model_RANDose_MambaA and Model_RANDose_MambaB.

  ---
  Training commands:
  # Strategy A
  python train.py --model Model_RANDose_MambaA --loss Loss_DC_PTV \
    --batch_size 2 --list_GPU_ids 0 --max_iter 80000 \
    --project_name RANDose_MambaA ...

  # Strategy B
  python train.py --model Model_RANDose_MambaB ...

  One practical note: at the shallowest encoder stage (full 128³ resolution), the Mamba batch sizes are large (e.g., B·H·W = 32768 for B=2). If you hit OOM, the simplest fix is to remove
  self.mamba from enc1/dec1 blocks, since long-range context matters most at the deeper bottleneck stages anyway.

