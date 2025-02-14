import yaml
import subprocess

# 模型列表
model_names = ["Alike", "SuperPoint", "XFeat", "D2Net", "DISK", "r2d2", "sfd2", "EdgePoint", "GoodPoint", "LETNet", "Harris"]

dataset_set_num = 16
dataset_sets = [
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH000/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH001/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH002/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH003/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH004/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH005/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH006/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/MH007/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME000/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME001/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME002/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME003/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME004/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME005/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME006/",
    "/home/data/Dataset/tartanair/mono/tartanair-test-mono-release/mono/ME007/",
]

gt_sets = [
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH000.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH001.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH002.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH003.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH004.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH005.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH006.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/MH007.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME000.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME001.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME002.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME003.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME004.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME005.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME006.txt",
    "/home/data/Dataset/tartanair/tartanair_cvpr_gt/mono_gt/ME007.txt",
]

# 读取并修改config.yaml文件中的模型类型
def update_model_type(model_name, dataset_set, gt_set):
    # 加载配置文件
    with open("config/config_fund.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 更新模型类型
    config["test"]["model"]["params"]["model_type"] = model_name
    config["test"]["data"]["params"]["tartanair_params"]["root"] = dataset_set
    config["test"]["data"]["params"]["tartanair_params"]["gt"] = gt_set

    # 保存修改后的配置文件
    with open("config/config_run.yaml", "w") as file:
        yaml.dump(config, file)



# 运行python命令
def run_command():
    subprocess.run(["python3", "main.py", "-c", "./config/config_run.yaml", "test"])

# 主函数
def main():
    for dataset_set, gt_set in zip(dataset_sets, gt_sets):
        print(f"Running with dataset set: {dataset_set}, gt set: {gt_set}")
        for model_name in model_names:
            print(f"Running with model: {model_name}")
            update_model_type(model_name, dataset_set, gt_set)  # 更新配置文件
            run_command()  # 运行命令

if __name__ == "__main__":
    main()

