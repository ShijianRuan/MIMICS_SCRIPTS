
from pathlib import Path
import os
from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import json
import Action1_ConvertLabeledToTrainData
import Action2_PlanAndPreprocess
import Action3_Train
import Action4_Predict
import Action5_Evaluation
import SetEnvionmentVariables
import ResampleImageAndMask

def _get_preprocess_configuration(config):
    return config["PREPROCESS"]["configuration"]


def _dataset_id_from_name(dataset_name: str) -> int:
    # 例如 Dataset201_MR1_Head -> 201
    token = Path(dataset_name).name.split("_")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        raise ValueError(f"无法从数据集名解析 dataset_id: {dataset_name}")
    return int(digits)



def ReadConfigFile(file_path):
    '''
    本函数提供配置文件读入的功能
    并根据读入的配置文件，组合操作，得到各数据路径等后续所需所有配置
    '''

    SCRIPT_DIR = Path(__file__).resolve().parent
    configfile = SCRIPT_DIR/file_path

    with open(configfile, "rb") as f:
        config = tomllib.load(f)

    # 记录配置文件名，供后续生成 .sh 脚本时引用
    config["_config_file"] = str(file_path)


    modelfile = config['MODEL']['segment_model_file']
    with open(SCRIPT_DIR/modelfile, "rb") as f:
        modelmap = tomllib.load(f)


    #从modelmap中，根据“segment_list_name”，得到具体要分割的类别
    config['MODEL']["segment_list"] = []
    for segmentlistname in config["MODEL"]["segment_list_name"]:
        config['MODEL']["segment_list"].append(modelmap[segmentlistname])

    
    #根据配置文件，从标注数据生成训练数据集
    config["PATHS"]["dataset_path"] = []
    for dataset in config["PATHS"]["labeled_dataset"]:
        datasetpath = Path(config["PATHS"]["labeled_path"])/dataset
        config["PATHS"]["dataset_path"].append(str(datasetpath))

    config["PATHS"]["train_path"] = str(Path(config["PATHS"]["train_path"])/Path(config["PATHS"]["train_project"]))
    trainpath = Path(config["PATHS"]["train_path"])

    if(not trainpath.exists()):
        os.mkdir(trainpath)

    #建立nnUnet的路径，并加入环境变量
    nnUNet_raw = trainpath/config["PATHS"]["nnUNet_raw"]
    nnUNet_preprocessed = trainpath/config["PATHS"]["nnUNet_preprocessed"]
    nnUNet_results = trainpath/config["PATHS"]["nnUNet_results"]
    if (not nnUNet_raw.exists()):
        os.mkdir(nnUNet_raw)
    if (not nnUNet_preprocessed.exists()):
        os.mkdir(nnUNet_preprocessed)
    if (not nnUNet_results.exists()):
        os.mkdir(nnUNet_results)

    # SetEnvionmentVariables.add_to_user_shell_config("nnUNet_raw", nnUNet_raw)
    # SetEnvionmentVariables.add_to_user_shell_config("nnUNet_preprocessed", nnUNet_preprocessed)
    # SetEnvionmentVariables.add_to_user_shell_config("nnUNet_results", nnUNet_results)
    # os.environ["nnUNet_raw"] = str(nnUNet_raw)
    # os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
    # os.environ["nnUNet_results"] = str(nnUNet_results)

    config["PATHS"]["nnUNet_raw"] = str(nnUNet_raw)
    config["PATHS"]["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
    config["PATHS"]["nnUNet_results"] = str(nnUNet_results)

    config["PATHS"]["nnUNet_path"] = [f"{nnUNet_raw}/{train_dataset}" for train_dataset in config["MODEL"]["train_dataset"]]
    
    #把当前配置写出到训练输出的目录，并打上时间戳，因为这个目录下可能会进行多次训练
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for train_dataset in config["MODEL"]["train_dataset"]:
        jsonpath = f"{trainpath}/config_{train_dataset}_{timestamp}.json"
        #config["COMMON"]["config_json_path"] = jsonpath
        with open(jsonpath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    return config



def set_environment_variables(config):

    nnUNet_raw=config["PATHS"]["nnUNet_raw"]
    nnUNet_preprocessed=config["PATHS"]["nnUNet_preprocessed"]
    nnUNet_results=config["PATHS"]["nnUNet_results"]

    SetEnvionmentVariables.add_to_user_shell_config("nnUNet_raw", nnUNet_raw)
    SetEnvionmentVariables.add_to_user_shell_config("nnUNet_preprocessed", nnUNet_preprocessed)
    SetEnvionmentVariables.add_to_user_shell_config("nnUNet_results", nnUNet_results)
    os.environ["nnUNet_raw"] = str(nnUNet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
    os.environ["nnUNet_results"] = str(nnUNet_results)



def convertdata(config):

    #生成本次训练的数据集
    dataset_paths = config["PATHS"]["dataset_path"]
    nnunet_paths = config["PATHS"]["nnUNet_path"]
    class_maps = config["MODEL"]["segment_list"]
    spacing = config["PREPROCESS"]["spacing"]
    orientation = config["PREPROCESS"]["orientation"]
    # 空字符串视为不做重定向
    if isinstance(orientation, str) and len(orientation.strip()) == 0:
        orientation = None

    # 从配置文件读取模态和 image reader/writer
    modality = config["COMMON"]["modality"]
    image_reader_writer = config["PREPROCESS"]["reorientaion"]

    if len(class_maps) == len(nnunet_paths):
        for nnunet_path, class_map in zip(nnunet_paths, class_maps):
            Action1_ConvertLabeledToTrainData.convert(
                dataset_paths, Path(nnunet_path), class_map, spacing,
                target_orientation=orientation,
                modality=modality,
                image_reader_writer=image_reader_writer)

    return



def preprocess(config):

    datasets = config["MODEL"]['train_dataset']
    configuration = config["PREPROCESS"]["configuration"]
    num_processes = config["PREPROCESS"]["num_processes"]
    spacing = config["PREPROCESS"]["spacing"]
    # 空列表视为不指定 target_spacing，由 nnUNet 自动规划
    target_spacing = spacing if spacing is not None and len(spacing) > 0 else None

    # patch_size：空列表视为不指定，由 nnUNet 自动规划
    patch_size_cfg = config["PREPROCESS"].get("patch_size", [])
    target_patch_size = patch_size_cfg if patch_size_cfg is not None and len(patch_size_cfg) > 0 else None

    # batch_size：0 视为不指定，由 nnUNet 自动规划
    batch_size_cfg = config["PREPROCESS"].get("batch_size", 0)
    target_batch_size = batch_size_cfg if batch_size_cfg is not None and batch_size_cfg > 0 else None

    #预处理
    for dataset in datasets:
        dataset_id = _dataset_id_from_name(dataset)
        Action2_PlanAndPreprocess.stage_preprocess(
                dataset_id = dataset_id,
                configuration = configuration,
                num_processes = num_processes,
                target_spacing = target_spacing,
                target_patch_size = target_patch_size,
                target_batch_size = target_batch_size,
            )
    
    return


def _normalize_gpu_ids(config, n):
    """将 config 中的 gpu_id 归一化为长度 n 的列表。"""
    gpu_ids = config["GPU"]["gpu_id"]
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids] * n
    if len(gpu_ids) != n:
        raise ValueError(f"gpu_id 列表长度 ({len(gpu_ids)}) 与数据集数量 ({n}) 不一致")
    return gpu_ids


def _get_single_gpu_id(config):
    """单 GPU 时返回 gpu_id（int），多 GPU 时返回 None。"""
    gpu_ids = config["GPU"]["gpu_id"]
    if not isinstance(gpu_ids, list):
        return gpu_ids
    return gpu_ids[0] if len(set(gpu_ids)) <= 1 else None


def _is_multigpu(configs):
    """判断多个配置中是否存在不同的 gpu_id，有则返回 True。"""
    all_gpu_ids = set()
    for cfg in configs:
        gpu_ids = cfg["GPU"]["gpu_id"]
        if isinstance(gpu_ids, list):
            all_gpu_ids.update(gpu_ids)
        else:
            all_gpu_ids.add(gpu_ids)
    return len(all_gpu_ids) > 1


def train(config):
    """在当前进程中执行 nnUNet 训练（gpu_id 从 config 中读取）。"""
    datasets      = config["MODEL"]["train_dataset"]
    configuration = config["PREPROCESS"]["configuration"]
    trainer       = config["TRAIN"]["trainer"]
    plans         = config["TRAIN"]["plans"]
    fold          = config["TRAIN"]["fold"]
    gpu_id        = _get_single_gpu_id(config)

    for ds_name in datasets:
        Action3_Train.stage_train(
            dataset_id=_dataset_id_from_name(ds_name),
            configuration=configuration, 
            fold=fold,
            trainer=trainer, 
            plans=plans, 
            gpu_id=gpu_id,
        )



def predict(config):
    """在当前进程中执行 nnUNet 预测（gpu_id 从 config 中读取）。"""
    datasets      = config["MODEL"]["train_dataset"]
    configuration = config["PREPROCESS"]["configuration"]
    fold          = config["TRAIN"]["fold"]
    trainer       = config["TRAIN"]["trainer"]
    plans         = config["TRAIN"]["plans"]
    gpu_id        = _get_single_gpu_id(config)

    for ds_name in datasets:
        Action4_Predict.stage_predict(
            dataset_id=_dataset_id_from_name(ds_name),
            configuration=configuration, fold=fold,
            trainer=trainer, plans=plans,
            dataset_folder_name=ds_name, gpu_id=gpu_id,
        )


def evaluation(config):
    nnunet_paths = config["PATHS"]["nnUNet_path"]
    class_maps = config["MODEL"]["segment_list"]
    evaluate_files = []
    if len(class_maps) == len(nnunet_paths):
        for nnunet_path, class_map in zip(nnunet_paths, class_maps):
            # 分组格式的 class_map 需要转换为 {group_name: label} 的扁平字典
            if Action1_ConvertLabeledToTrainData._is_grouped_class_map(class_map):
                _, class_map = Action1_ConvertLabeledToTrainData._expand_grouped_class_map(class_map)
            evaluate_files.append(Action5_Evaluation.evaluate(
                Path(nnunet_path) / "labelsTs",
                Path(nnunet_path) / "labelsTs_predicted",
                class_map,
                Path(nnunet_path) / "evaluation"))
    evaluate_path = Path(nnunet_paths[0]).parent
    Action5_Evaluation.run_evaluation_aggregation(evaluate_files, evaluate_path, 'evaluation')


def train_and_predict_multigpu(configs):
    """
    根据配置中的 gpu_id 分配，为每个 GPU 生成一个独立的 .sh 脚本。
    支持单个 config（dict）或多个 config（list）。
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    from collections import defaultdict

    if isinstance(configs, dict):
        configs = [configs]

    gpu_tasks = defaultdict(list)
    for cfg in configs:
        datasets    = cfg["MODEL"]["train_dataset"]
        gpu_ids     = _normalize_gpu_ids(cfg, len(datasets))
        conf        = cfg["PREPROCESS"]["configuration"]
        fold        = cfg["TRAIN"]["fold"]
        trainer     = cfg["TRAIN"]["trainer"]
        plans       = cfg["TRAIN"]["plans"]
        disable_tta = cfg["PREDICT"]["disable_tta"]
        nnunet_raw  = cfg["PATHS"]["nnUNet_raw"]
        cfg_file    = cfg.get("_config_file", "unknown")

        for ds_name, gid in zip(datasets, gpu_ids):
            gpu_tasks[gid].append({
                "ds_id": _dataset_id_from_name(ds_name), "ds_name": ds_name,
                "configuration": conf, "fold": fold,
                "trainer": trainer, "plans": plans,
                "disable_tta": disable_tta, "nnunet_raw": nnunet_raw,
                "cfg_file": cfg_file,
            })

    for gid in sorted(gpu_tasks.keys()):
        tasks = gpu_tasks[gid]
        sh_path = SCRIPT_DIR / f"gpu{gid}.sh"
        lines = [
            "#!/bin/bash",
            f"# gpu{gid}.sh —— 自动生成的 GPU {gid} 训练和预测脚本",
            f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# 配置来源: {', '.join(sorted(set(t['cfg_file'] for t in tasks)))}",
            "", "set -e", "",
            "# ── 训练 ──────────────────────────────────────────────────",
        ]
        for t in tasks:
            lines.append(f"echo \"[GPU {gid}] Training Start: {t['ds_name']} (ID={t['ds_id']})\"")
            lines.append(
                f"CUDA_VISIBLE_DEVICES={gid} nnUNetv2_train "
                f"{t['ds_id']} {t['configuration']} {t['fold']} "
                f"-tr {t['trainer']} -p {t['plans']}"
            )
            lines.append(f"echo \"[GPU {gid}] Training End: {t['ds_name']}\"")
            lines.append("")

        lines.append("# ── 预测 ──────────────────────────────────────────────────")
        for t in tasks:
            tta = " --disable_tta" if t["disable_tta"] else ""
            lines.append(f"echo \"[GPU {gid}] Predicting Start: {t['ds_name']} (ID={t['ds_id']})\"")
            lines.append(f"cd {t['nnunet_raw']}/{t['ds_name']}")
            lines.append(
                f"CUDA_VISIBLE_DEVICES={gid} nnUNetv2_predict "
                f"-i imagesTs -o labelsTs_predicted "
                f"-d {t['ds_id']} -c {t['configuration']} "
                f"-tr {t['trainer']} -p {t['plans']} -f {t['fold']}{tta}"
            )
            lines.append(f"echo \"[GPU {gid}] Predicting End: {t['ds_name']}\"")
            lines.append("")
        lines.append(f"echo \"[GPU {gid}] 所有任务完成！\"")

        with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  已生成: {sh_path}")

    print(f"\n多 GPU 模式，请在不同终端中分别运行:")
    for gid in sorted(gpu_tasks.keys()):
        print(f"  bash gpu{gid}.sh")


# ── Workflow ─────────────────────────────────────────────────────

def workflow1_nnUnet_train_and_predict():
    """单配置文件的训练流程。"""
    config = ReadConfigFile("Config_CTWholeBodyBone.toml")
    set_environment_variables(config)
    #convertdata(config)
    preprocess(config)

    # if _get_single_gpu_id(config) is not None:
    #     train(config)
    #     predict(config)
    # else:
    train_and_predict_multigpu(config)
    print("训练和预测完成后，可单独调用 evaluation(config) 进行评估。")

    # evaluation(config)


def workflow2_nnUnet_train_and_predict_batch():
    """多配置文件的训练流程。"""
    config_files = [
        "Config_CoarseSeg_CT.toml",
        "Config_CoarseSeg_MR.toml",
    ]
    all_configs = []
    for cfg_file in config_files:
        config = ReadConfigFile(cfg_file)
        set_environment_variables(config)
        all_configs.append(config)
        # convertdata(config)
        # preprocess(config)

    if _is_multigpu(all_configs):
        train_and_predict_multigpu(all_configs)
        print("训练和预测完成后，可逐个调用 evaluation(config) 进行评估。")
    else:
        for cfg in all_configs:
            train(cfg)
            predict(cfg)
        for cfg in all_configs:
            evaluation(cfg)




def workflow3_sample_and_predict():

    """
        推理流程。
        对给出的图像目录，可以分别调用多个模型，输出多个预测结果mask
        如需要把多个预测结果拼成一个mask, 可调用combine_multimask_to_one完成
    """

    #模型文件路径，可以为多个
    model_paths = [
        "D:\\data\\CoarseSegmentation\\Dataset102_MRCoarseSegmentation\\nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    ]
    #预测结果路径，数量需和模型文件数量保持一致（因为不同模型预测结果的文件名相同）
    out_paths = [
        "D:\\data\\CoarseSegmentation\\Dataset102_MRCoarseSegmentation\\results"
    ]
    #待预测的图像
    input_path = "D:\\data\\MIv500_MR\\imagesTs"


    for model_path, out_path in zip(model_paths, out_paths):

        # 使用预插值加速预测：
        # 当输入图像分辨率（如 1.5mm）与模型训练分辨率（如 3.0mm）不一致时，
        # 先将输入图像快速下采样到模型分辨率，预测完再将 mask 上采样回原始分辨率。
        # 绕过 nnUNet 内部极慢的 skimage+order3 插值，速度可从 90s 降至几秒。
        Action4_Predict.easy_predict_with_preresample(
            model_folder=model_path,
            input_path=input_path,
            output_path=out_path,
            enable_stats=True,
        )

    return


def combine_multimask_to_one():
    """
        把分成多个模型训练的mask拼到一个里边
        根据modelmap中设定的label值进行拼接
        同时也需要每个mask是根据modelmap中哪个列表训练的
    """

    #输入待拼接的mask路径，可以是多个文件，也可以是多个路径
    #如果是多个路径，需保证路径中的mask文件名一样，仅拼接相同文件名的mask
    inputmask_paths = [
        "D:/data/MR_windows/nnUNet_raw/Dataset101_MR1test/labelsTs",
        "D:/data/MR_windows/nnUNet_raw/Dataset102_MR2test/labelsTs",
    ]
    #拼接前各mask的label值列表
    model_parts = [
        "MR1_Head",
        "MR2_Chest", 
    ]
    #拼接后mask的label值列表
    model_combine = [
        "MR_Combine"
    ]
    #拼接后的mask存放路径
    outputmask_path = "D:/data/MR_windows/nnUNet_raw/combine"


    SCRIPT_DIR = Path(__file__).resolve().parent
    with open(SCRIPT_DIR/"ModelMap.toml", "rb") as f:
        modelmap = tomllib.load(f)

    class_map = []
    for segmentlistname in model_parts:
        class_map.append(modelmap[segmentlistname])

    combine_map = {}
    for combinename in model_combine:
        combine_map.update(modelmap[combinename])

    Action1_ConvertLabeledToTrainData.convert_multilabel_to_one(
        inputmask_paths, outputmask_path,
        class_map=class_map, combine_map=combine_map)



def workflow4_shared_spacing_predict_and_merge():
    """
    推理流程。
    
    模拟c++实现流程，先对图像插值，然后依次调用所有模型共享相同分辨率预测结果，最后回插mask
    主要为测试时间性能用。

    与 workflow3 + combine_multimask_to_one 的区别：
      workflow3: 对每个模型分别做 预插值→推理→后插值，最后在原始分辨率上拼接。
                 若 N 个模型分辨率相同，图像被重复插值 N 次，mask 也被重复回插 N 次。
      workflow4: 图像仅预插值 1 次 → N 个模型在低分辨率上依次推理 → 在低分辨率上拼接 →
                 拼接后的总 mask 仅后插值 1 次回原始分辨率。节省 (N-1) 次图像/mask 插值。
    """

    # model_paths = [
    #     "D:/data/MIv500/models/Dataset101_CT1_Head/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset102_CT2_Chest/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset103_CT3_Abdomen/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset104_CT5_Vertebrae/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset105_CT6_Rib/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset106_CT7_Chestbone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset107_CT8_Abdomenbone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "D:/data/MIv500/models/Dataset108_CT9_Abdomenmuscle/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    # ]
    # model_parts = [
    #     "CT1_Head",
    #     "CT2_Chest",
    #     "CT3_Abdomen",
    #     "CT5_Vertebrae",
    #     "CT6_Rib",
    #     "CT7_Chestbone",
    #     "CT8_Abdomenbone",
    #     "CT9_Abdomenmuscle",
    # ]
    # input_path = "D:/data/testwholebody/image"
    # outputmask_path = "D:/data/testwholebody/labelsTs_predicted"

    model_paths = [
        "D:/data/MIv500_MR/models/Dataset201_MR1_Head/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
        "D:/data/MIv500_MR/models/Dataset202_MR2_Chest/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
        "D:/data/MIv500_MR/models/Dataset203_MR3_Abdomen/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
        "D:/data/MIv500_MR/models/Dataset204_MR5_Vertebrae/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
        "D:/data/MIv500_MR/models/Dataset205_MR7_AbdomenBone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
        "D:/data/MIv500_MR/models/Dataset206_MR8_muscle/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    ]
    model_parts = [
        "MR1_Head", 
        "MR2_Chest",
        "MR3_Abdomen",
        "MR5_Vertebrae",
        "MR7_AbdomenBone",
        "MR8_muscle",
    ]
    model_combine = [
        "MR_Combine",
    ]
    input_path = "D:/data/MIv500_MR/image"
    outputmask_path = "D:/data/MIv500_MR/labelsTs_predicted"


    # 读取 ModelMap，构建 class_map
    SCRIPT_DIR = Path(__file__).resolve().parent
    with open(SCRIPT_DIR / "ModelMap.toml", "rb") as f:
        modelmap = tomllib.load(f)

    class_maps = []
    for segmentlistname in model_parts:
        class_maps.append(modelmap[segmentlistname])

    combine_map = {}
    for combinename in model_combine:
        combine_map.update(modelmap[combinename])

    # 调用 Action4_Predict 中的多模型共享分辨率预测接口
    Action4_Predict.multimodel_predict_and_merge(
        model_folders=model_paths,
        input_path=input_path,
        output_path=outputmask_path,
        class_maps=class_maps,
        combine_map=combine_map,
        model_names=model_parts,
    )







if __name__ == "__main__":
    
    #workflow1: 训练流程。适用于单个训练，或参数不变的多个模型一起训练
    workflow1_nnUnet_train_and_predict()

    #workflow2: 训练流程。适用于不同配置的多个训练，为节省时间使用多块GPU并行训练
    #   每个 .toml 配置文件中 [GPU].gpu_id 指定 GPU 编号
    #   不同 GPU 上的任务并行, 同一 GPU 上串行
    #workflow2_nnUnet_train_and_predict_batch()

    #workflow3: 推理流程。各模型可使用不同分辨率，逐模型独立插值→推理→回插，最后拼接
    #workflow3_sample_and_predict()
    #combine_multimask_to_one()        #单独的拼接mask的函数，可将多个mask拼接到一个中

    # workflow4: 推理流程。模拟c++实现流程，先对图像插值，然后依次调用所有模型共享相同分辨率预测结果，最后回插mask
    # workflow4_shared_spacing_predict_and_merge()


    print("End!")