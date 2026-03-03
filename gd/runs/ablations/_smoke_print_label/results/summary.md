# Ablation Results: minimal_global

- Base Config: `D:\GreenDiff\GD\gd\configs\default.yaml`
- Stage: `print`

## Overview

| Name | Group | Status | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lg_backbone_cnn | proxy_backbone | PLANNED | - | - | - | - | - |
| lg_backbone_fno | proxy_backbone | PLANNED | - | - | - | - | - |
| lg_backbone_hybrid_fno | proxy_backbone | PLANNED | - | - | - | - | - |
| teacher_no_guidance | teacher_inference | PLANNED | - | - | - | - | - |
| teacher_legacy | teacher_inference | PLANNED | - | - | - | - | - |
| teacher_pmd1 | teacher_inference | PLANNED | - | - | - | - | - |
| teacher_pmd2 | teacher_inference | PLANNED | - | - | - | - | - |
| teacher_active_b1 | teacher_budget | PLANNED | - | - | - | - | - |

## proxy_backbone

| Name | Status | Run Dir | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lg_backbone_cnn | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\proxy_backbone\lg_backbone_cnn` | - | - | - | - | - |
| lg_backbone_fno | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\proxy_backbone\lg_backbone_fno` | - | - | - | - | - |
| lg_backbone_hybrid_fno | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\proxy_backbone\lg_backbone_hybrid_fno` | - | - | - | - | - |

## teacher_inference

| Name | Status | Run Dir | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher_no_guidance | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\teacher_inference\teacher_no_guidance` | - | - | - | - | - |
| teacher_legacy | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\teacher_inference\teacher_legacy` | - | - | - | - | - |
| teacher_pmd1 | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\teacher_inference\teacher_pmd1` | - | - | - | - | - |
| teacher_pmd2 | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\teacher_inference\teacher_pmd2` | - | - | - | - | - |

## teacher_budget

| Name | Status | Run Dir | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| teacher_active_b1 | PLANNED | `D:\GreenDiff\GD\gd\runs\ablations\_smoke_print_label\runs\teacher_budget\teacher_active_b1` | - | - | - | - | - |
