# Dataset Summary for "Show and Segment" Paper

## Training Datasets (12 datasets)

1. **AMOS** (`ji2022amos`) - Abdominal Multi-Organ Segmentation
   - Modalities: CT and MRI
   - 500 CT + 100 MRI scans
   - 15 anatomical structures

2. **BCV** (`bcv`) - Multi-Atlas Labeling Beyond Cranial Vault
   - Modality: CT
   - 30 abdominal CT scans
   - 13 abdominal organs

3. **LiTS** (`bilic2019liver`) - Liver Tumor Segmentation
   - Modality: CT
   - 131 training cases
   - Liver and tumor annotations

4. **KiTS19** (`heller2019kits19`) - Kidney Tumor Segmentation
   - Modality: CT
   - 210 cases (300 total, 90 private)
   - Kidney and tumor annotations

5. **StructSeg** (`structseg`) - Radiotherapy Planning
   - Modality: CT
   - Head & Neck (50 cases, 22 organs)
   - Thorax (50 cases, 6 organs)

6. **CHAOS** (`CHAOS2021`) - Combined CT-MR Healthy Abdominal
   - Modalities: CT and MRI (T1, T2)
   - 20 patients
   - 4 organs: liver, kidneys, spleen

7. **M&Ms** (`campello2021multi`) - Multi-center Cardiac Segmentation
   - Modality: Cardiac MRI
   - 320 cases
   - 3 structures: LV, RV, myocardium

8. **Brain Aging** (`rodrigue2012beta`) - Brain Tissue Segmentation
   - Modality: T1 MRI
   - 213 scans
   - 3 tissue types: CSF, gray matter, white matter

9. **AutoPET** (`gatidis2022whole`) - Whole-body FDG-PET/CT
   - Modalities: PET/CT
   - 1014 studies
   - Tumor lesions

10. **CSI-wat** (`ivdm3seg`) - Spine MRI Water Image
    - Modality: MRI (water sequence)
    - 16 scans
    - Intervertebral discs

11. **M&Ms RV** (`martin2023deep`) - Right Ventricle Segmentation
    - Modality: Cardiac MRI
    - Extension of M&Ms dataset

12. Additional training dataset mentioned but not specified

## Held-out Test Datasets (5 datasets)

### For OOD Generalization:
1. **ACDC** (`bernard2018deep`) - Automated Cardiac Diagnosis
   - Modality: Cardiac MRI
   - 100 cases
   - Cross-center variation test

2. **SegTHOR** (`lambert2020segthor`) - Thoracic Organs at Risk
   - Modality: CT
   - 40 cases
   - 4 structures: heart, aorta, trachea, esophagus

3. **CSI variants** (`ivdm3seg`) - Spine MRI Modalities
   - CSI-inn (in-phase)
   - CSI-opp (opposed-phase)
   - CSI-fat (fat image)
   - Cross-modality adaptation test

### For Novel Class Adaptation:
4. **MSD Pancreas** (`antonelli2022medical`) - Medical Segmentation Decathlon
   - Modality: CT
   - 281 images
   - Pancreas tumor (novel class)

5. **Pelvic1K** (`liu2021deep`) - Pelvic Bone Segmentation
   - Modality: CT
   - 103 scans (subset)
   - 4 skeletal structures (novel classes)

## Key Implementation References

- **nnUNet** (`isensee2021nnu`) - Baseline comparison method
- **Lamb Optimizer** (`you2019large`) - Training optimization
- **Visual In-context Learning** (`zhang2023makes`) - Inference strategy analysis