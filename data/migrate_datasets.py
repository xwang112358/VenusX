import os
import subprocess
import shutil
import argparse
from huggingface_hub import HfApi, list_datasets
from huggingface_hub.utils import HfFolder, HfHubHTTPError

# --- 配置区 ---
ORIGINAL_ORG = "AI4Protein"
ANONYMOUS_USER = "anonymous-researcher-123"
LOCAL_TEMP_DIR = "./migration_temp_venusx"
# ----------------

def run_command(command, working_dir):
    """在指定目录下安全地运行一个shell命令"""
    print(f"[{working_dir}]$ {command}")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 命令执行失败: {e.stderr}")
        raise

def anonymize_readme(readme_path, original_author):
    """对README文件进行基本的匿名化处理"""
    if not os.path.exists(readme_path): return
    print(f"  -> 正在匿名化 README...")
    try:
        with open(readme_path, 'r', encoding='utf-8') as f: content = f.read()
        content = content.replace(original_author, "[Anonymous Organization]")
        with open(readme_path, 'w', encoding='utf-8') as f: f.write(content)
    except Exception as e:
        print(f"  ⚠️  无法匿名化 README: {e}")

def migrate_venusx_datasets(api: HfApi):
    """
    任务一：迁移 VenusX 数据集。
    - 从 ORIGINAL_ORG 查找所有 VenusX 数据集。
    - 检查 ANONYMOUS_USER 下对应的仓库，如果非空则跳过。
    - 执行迁移。
    """
    print("\n=============================================")
    print("========= 任务1: 开始迁移数据集 =========")
    print("=============================================\n")

    if os.path.exists(LOCAL_TEMP_DIR):
        shutil.rmtree(LOCAL_TEMP_DIR)
    os.makedirs(LOCAL_TEMP_DIR)

    print(f"正在获取组织 '{ORIGINAL_ORG}' 下的数据集...")
    all_original_datasets = list(list_datasets(author=ORIGINAL_ORG, full=False))
    venusx_datasets = [ds for ds in all_original_datasets if "VenusX" in ds.id]
    
    print(f"找到 {len(venusx_datasets)} 个包含 'VenusX' 的数据集，准备开始迁移。")

    for i, dataset in enumerate(venusx_datasets):
        original_repo_id = dataset.id
        dataset_name = original_repo_id.split('/')[-1]
        new_repo_id = f"{ANONYMOUS_USER}/{dataset_name}"
        
        print(f"\n--- ({i+1}/{len(venusx_datasets)}) 处理: {original_repo_id} ---")

        original_repo_local_path = os.path.join(LOCAL_TEMP_DIR, "original")
        new_repo_local_path = os.path.join(LOCAL_TEMP_DIR, "new")

        try:
            print(f"  -> 检查目标仓库 {new_repo_id} 状态...")
            try:
                repo_files = api.list_repo_files(repo_id=new_repo_id, repo_type="dataset")
                if repo_files:
                    # 过滤掉只有 .gitattributes 的情况，认为这是空仓库
                    meaningful_files = [f for f in repo_files if f != '.gitattributes']
                    if meaningful_files:
                        print(f"  ✅ 目标仓库已有 {len(meaningful_files)} 个有效文件，跳过迁移。")
                        continue
                    else:
                        print(f"  ℹ️  目标仓库只有 .gitattributes 文件，视为空仓库，继续迁移。")
            except HfHubHTTPError as e:
                if e.response.status_code != 404: raise

            print(f"  -> 目标仓库为空或不存在，开始迁移...")
            api.create_repo(repo_id=new_repo_id, repo_type="dataset", private=True, exist_ok=True)

            print(f"  -> 正在克隆原始仓库...")
            run_command(f"git clone https://huggingface.co/datasets/{original_repo_id} {original_repo_local_path}", working_dir=".")
            run_command("git lfs pull", working_dir=original_repo_local_path)
            
            print(f"  -> 正在克隆空的匿名仓库...")
            token = HfFolder.get_token()
            run_command(f"git clone https://{ANONYMOUS_USER}:{token}@huggingface.co/datasets/{new_repo_id} {new_repo_local_path}", working_dir=".")

            print(f"  -> 正在复制文件...")
            copied_files = []
            for item in os.listdir(original_repo_local_path):
                if item == '.git': continue
                s = os.path.join(original_repo_local_path, item)
                d = os.path.join(new_repo_local_path, item)
                if os.path.isdir(s): 
                    shutil.copytree(s, d, dirs_exist_ok=True)
                    copied_files.append(f"{item}/ (目录)")
                else: 
                    shutil.copy2(s, d)
                    copied_files.append(item)
            print(f"  -> 已复制 {len(copied_files)} 个项目: {', '.join(copied_files)}")
            
            anonymize_readme(os.path.join(new_repo_local_path, "README.md"), ORIGINAL_ORG)

            print(f"  -> 正在检查新仓库中的文件...")
            new_files = os.listdir(new_repo_local_path)
            print(f"  -> 新仓库包含 {len(new_files)} 个文件/目录: {new_files}")

            print(f"  -> 正在提交并推送到匿名仓库...")
            run_command("git add .", working_dir=new_repo_local_path)
            
            # 检查 Git 状态
            print(f"  -> 检查 Git 状态...")
            run_command("git status", working_dir=new_repo_local_path)
            
            run_command('git commit -m "Initial anonymous commit"', working_dir=new_repo_local_path)
            
            # 强制推送以确保覆盖远程仓库的内容
            run_command("git push --force", working_dir=new_repo_local_path)

            print(f"  ✅ 成功迁移 {original_repo_id} 到 {new_repo_id}")

        except Exception as e:
            print(f"  ❌ 处理 {original_repo_id} 时发生错误: {e}")
        
        finally:
            if os.path.exists(original_repo_local_path): shutil.rmtree(original_repo_local_path, ignore_errors=True)
            if os.path.exists(new_repo_local_path): shutil.rmtree(new_repo_local_path, ignore_errors=True)
    
    print("\n迁移任务执行完毕！")

def manage_venusx_datasets(api: HfApi):
    """
    任务二：管理 VenusX 数据集。
    - 将 ANONYMOUS_USER 下所有 VenusX 仓库设为公开。
    """
    print("\n=============================================")
    print("======= 任务2: 开始管理已迁移仓库 =======")
    print("=============================================\n")

    print(f"正在获取用户 '{ANONYMOUS_USER}' 名下的 'VenusX' 数据集...")
    try:
        user_datasets = list(api.list_datasets(author=ANONYMOUS_USER))
        venusx_datasets = [ds for ds in user_datasets if "VenusX" in ds.id]
        if not venusx_datasets:
            print("未找到与 'VenusX' 相关的数据集进行管理。")
            return
        print(f"找到 {len(venusx_datasets)} 个 'VenusX' 数据集进行管理。")
    except Exception as e:
        print(f"❌ 无法获取数据集列表: {e}")
        return

    for i, dataset in enumerate(venusx_datasets):
        repo_id = dataset.id
        print(f"\n--- ({i+1}/{len(venusx_datasets)}) 管理仓库: {repo_id} ---")
        
        try:
            print(f"  -> 设置为公开...")
            api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=False)
            print(f"  ✅ 已设为公开。")
        except HfHubHTTPError as e:
            if "is already public" in str(e): print(f"  ℹ️  已经是公开的。")
            else: print(f"  ❌ 设置可见性时出错: {e}")
                
    print("\n管理任务执行完毕！")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="迁移并管理Hugging Face上的VenusX数据集。")
    parser.add_argument("--migrate-only", action="store_true", help="只执行迁移任务，不进行管理。")
    parser.add_argument("--manage-only", action="store_true", help="只执行管理任务（设为公开），不进行迁移。")
    args = parser.parse_args()

    # 检查是否同时使用了互斥的参数
    if args.migrate_only and args.manage_only:
        print("错误：--migrate-only 和 --manage-only 参数不能同时使用。")
        exit(1)

    # 准备工作
    print("正在初始化 Hugging Face API...")
    try:
        # 确保已通过 huggingface-cli login 登录
        api = HfApi()
        print("API 初始化成功。")
    except Exception as e:
        print(f"API 初始化失败: {e}")
        print("请确保您已通过 'huggingface-cli login' 命令登录。")
        exit(1)

    # 根据参数执行相应任务
    if args.manage_only:
        manage_venusx_datasets(api)
    elif args.migrate_only:
        migrate_venusx_datasets(api)
    else:
        # 默认情况下，先迁移，再管理
        print("默认模式：将先执行迁移任务，然后执行管理任务。")
        migrate_venusx_datasets(api)
        manage_venusx_datasets(api)
    
    print("\n🎉 全部任务已完成！")