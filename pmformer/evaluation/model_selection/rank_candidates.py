#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from pmformer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "PMFormerPlans"

    overwrite_plans = {
        'PMFormerTrainerV2_2': ["PMFormerPlans", "PMFormerPlansisoPatchesInVoxels"], # r
        'PMFormerTrainerV2': ["PMFormerPlansnonCT", "PMFormerPlansCT2", "PMFormerPlansallConv3x3",
                            "PMFormerPlansfixedisoPatchesInVoxels", "PMFormerPlanstargetSpacingForAnisoAxis",
                            "PMFormerPlanspoolBasedOnSpacing", "PMFormerPlansfixedisoPatchesInmm", "PMFormerPlansv2.1"],
        'PMFormerTrainerV2_warmup': ["PMFormerPlans", "PMFormerPlansv2.1", "PMFormerPlansv2.1_big", "PMFormerPlansv2.1_verybig"],
        'PMFormerTrainerV2_cycleAtEnd': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_cycleAtEnd2': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_reduceMomentumDuringTraining': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_graduallyTransitionFromCEToDice': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_independentScalePerAxis': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Mish': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Ranger_lr3en4': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_GN': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_momentum098': ["PMFormerPlans", "PMFormerPlansv2.1"],
        'PMFormerTrainerV2_momentum09': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_DP': ["PMFormerPlansv2.1_verybig"],
        'PMFormerTrainerV2_DDP': ["PMFormerPlansv2.1_verybig"],
        'PMFormerTrainerV2_FRN': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_resample33': ["PMFormerPlansv2.3"],
        'PMFormerTrainerV2_O2': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ResencUNet': ["PMFormerPlans_FabiansResUNet_v2.1"],
        'PMFormerTrainerV2_DA2': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_allConv3x3': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ForceBD': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ForceSD': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_LReLU_slope_2en1': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_lReLU_convReLUIN': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ReLU': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ReLU_biasInSegOutput': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_ReLU_convReLUIN': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_lReLU_biasInSegOutput': ["PMFormerPlansv2.1"],
        #'PMFormerTrainerV2_Loss_MCC': ["PMFormerPlansv2.1"],
        #'PMFormerTrainerV2_Loss_MCCnoBG': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Loss_DicewithBG': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Loss_Dice_LR1en3': ["PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Loss_Dice': ["PMFormerPlans", "PMFormerPlansv2.1"],
        'PMFormerTrainerV2_Loss_DicewithBG_LR1en3': ["PMFormerPlansv2.1"],
        # 'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],
        # 'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],
        # 'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],
        # 'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],
        # 'PMFormerTrainerV2_fp32': ["PMFormerPlansv2.1"],

    }

    trainers = ['PMFormerTrainer'] + ['PMFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'PMFormerTrainerNewCandidate24_2',
        'PMFormerTrainerNewCandidate24_3',
        'PMFormerTrainerNewCandidate26_2',
        'PMFormerTrainerNewCandidate27_2',
        'PMFormerTrainerNewCandidate23_always3DDA',
        'PMFormerTrainerNewCandidate23_corrInit',
        'PMFormerTrainerNewCandidate23_noOversampling',
        'PMFormerTrainerNewCandidate23_softDS',
        'PMFormerTrainerNewCandidate23_softDS2',
        'PMFormerTrainerNewCandidate23_softDS3',
        'PMFormerTrainerNewCandidate23_softDS4',
        'PMFormerTrainerNewCandidate23_2_fp16',
        'PMFormerTrainerNewCandidate23_2',
        'PMFormerTrainerVer2',
        'PMFormerTrainerV2_2',
        'PMFormerTrainerV2_3',
        'PMFormerTrainerV2_3_CE_GDL',
        'PMFormerTrainerV2_3_dcTopk10',
        'PMFormerTrainerV2_3_dcTopk20',
        'PMFormerTrainerV2_3_fp16',
        'PMFormerTrainerV2_3_softDS4',
        'PMFormerTrainerV2_3_softDS4_clean',
        'PMFormerTrainerV2_3_softDS4_clean_improvedDA',
        'PMFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'PMFormerTrainerV2_3_softDS4_radam',
        'PMFormerTrainerV2_3_softDS4_radam_lowerLR',

        'PMFormerTrainerV2_2_schedule',
        'PMFormerTrainerV2_2_schedule2',
        'PMFormerTrainerV2_2_clean',
        'PMFormerTrainerV2_2_clean_improvedDA_newElDef',

        'PMFormerTrainerV2_2_fixes', # running
        'PMFormerTrainerV2_BN', # running
        'PMFormerTrainerV2_noDeepSupervision', # running
        'PMFormerTrainerV2_softDeepSupervision', # running
        'PMFormerTrainerV2_noDataAugmentation', # running
        'PMFormerTrainerV2_Loss_CE', # running
        'PMFormerTrainerV2_Loss_CEGDL',
        'PMFormerTrainerV2_Loss_Dice',
        'PMFormerTrainerV2_Loss_DiceTopK10',
        'PMFormerTrainerV2_Loss_TopK10',
        'PMFormerTrainerV2_Adam', # running
        'PMFormerTrainerV2_Adam_PMFormerTrainerlr', # running
        'PMFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'PMFormerTrainerV2_SGD_lr1en1', # running
        'PMFormerTrainerV2_SGD_lr1en3', # running
        'PMFormerTrainerV2_fixedNonlin', # running
        'PMFormerTrainerV2_GeLU', # running
        'PMFormerTrainerV2_3ConvPerStage',
        'PMFormerTrainerV2_NoNormalization',
        'PMFormerTrainerV2_Adam_ReduceOnPlateau',
        'PMFormerTrainerV2_fp16',
        'PMFormerTrainerV2', # see overwrite_plans
        'PMFormerTrainerV2_noMirroring',
        'PMFormerTrainerV2_momentum09',
        'PMFormerTrainerV2_momentum095',
        'PMFormerTrainerV2_momentum098',
        'PMFormerTrainerV2_warmup',
        'PMFormerTrainerV2_Loss_Dice_LR1en3',
        'PMFormerTrainerV2_NoNormalization_lr1en3',
        'PMFormerTrainerV2_Loss_Dice_squared',
        'PMFormerTrainerV2_newElDef',
        'PMFormerTrainerV2_fp32',
        'PMFormerTrainerV2_cycleAtEnd',
        'PMFormerTrainerV2_reduceMomentumDuringTraining',
        'PMFormerTrainerV2_graduallyTransitionFromCEToDice',
        'PMFormerTrainerV2_insaneDA',
        'PMFormerTrainerV2_independentScalePerAxis',
        'PMFormerTrainerV2_Mish',
        'PMFormerTrainerV2_Ranger_lr3en4',
        'PMFormerTrainerV2_cycleAtEnd2',
        'PMFormerTrainerV2_GN',
        'PMFormerTrainerV2_DP',
        'PMFormerTrainerV2_FRN',
        'PMFormerTrainerV2_resample33',
        'PMFormerTrainerV2_O2',
        'PMFormerTrainerV2_ResencUNet',
        'PMFormerTrainerV2_DA2',
        'PMFormerTrainerV2_allConv3x3',
        'PMFormerTrainerV2_ForceBD',
        'PMFormerTrainerV2_ForceSD',
        'PMFormerTrainerV2_ReLU',
        'PMFormerTrainerV2_LReLU_slope_2en1',
        'PMFormerTrainerV2_lReLU_convReLUIN',
        'PMFormerTrainerV2_ReLU_biasInSegOutput',
        'PMFormerTrainerV2_ReLU_convReLUIN',
        'PMFormerTrainerV2_lReLU_biasInSegOutput',
        'PMFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'PMFormerTrainerV2_Loss_MCCnoBG',
        'PMFormerTrainerV2_Loss_DicewithBG',
        # 'PMFormerTrainerV2_Loss_Dice_LR1en3',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
        # 'PMFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
