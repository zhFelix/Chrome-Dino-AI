from sys import version
import numpy as np
import time
import random
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common import service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import cv2
from PIL import Image
import io
import base64
import os

# 加载配置文件
def validate_config(config):
    """验证配置参数的有效性"""
    errors = []
    
    # 验证训练参数
    training = config.get("training", {})
    if training.get("population_size", 0) < 5:
        errors.append("种群大小至少为5")
    if training.get("generations", 0) < 1:
        errors.append("训练代数至少为1")
    if training.get("runs_per_individual", 0) < 1:
        errors.append("每个个体运行次数至少为1")
    
    # 验证遗传算法参数
    genetic = config.get("genetic", {})
    if not (0 <= genetic.get("mutation_rate", 0) <= 1):
        errors.append("变异率必须在0-1之间")
    if genetic.get("mutation_scale", 0) <= 0:
        errors.append("变异幅度必须大于0")
    if genetic.get("tournament_size", 0) < 2:
        errors.append("锦标赛大小至少为2")
    if genetic.get("elite_count", 0) < 1:
        errors.append("精英个体数量至少为1")
    
    # 验证游戏参数
    game = config.get("game", {})
    if game.get("delay", 0) < 0:
        errors.append("游戏延迟不能为负数")
    
    return errors

def load_config_templates():
    """加载配置模板"""
    try:
        templates_path = os.path.join(os.path.dirname(__file__), 'config_templates.json')
        with open(templates_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("配置模板文件不存在")
        return {}
    except Exception as e:
        print(f"加载配置模板时出错: {e}")
        return {}

def save_custom_config(config_name, config, description=None):
    """保存自定义配置到模板文件"""
    try:
        templates = load_config_templates()
        
        # 添加新的自定义配置
        templates[config_name] = {
            "description": description if description else "用户自定义配置",
            "config": config
        }
        
        # 保存到文件
        templates_path = os.path.join(os.path.dirname(__file__), 'config_templates.json')
        with open(templates_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 配置 '{config_name}' 已保存到模板文件")
        return True
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")
        return False

def delete_config_template(config_name):
    """删除配置模板"""
    try:
        templates = load_config_templates()
        
        if config_name not in templates:
            print(f"❌ 配置 '{config_name}' 不存在")
            return False
        
        # 删除配置
        del templates[config_name]
        
        # 保存到文件
        templates_path = os.path.join(os.path.dirname(__file__), 'config_templates.json')
        with open(templates_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 配置 '{config_name}' 已删除")
        return True
    except Exception as e:
        print(f"❌ 删除配置失败: {e}")
        return False

def get_user_input_config():
    """让用户手动输入自定义配置参数"""
    print("\n=== 自定义配置输入 ===")
    print("请输入训练参数（直接回车使用默认值）:")
    
    # 从配置模板中获取标准配置作为默认值
    try:
        with open('config_templates.json', 'r', encoding='utf-8') as f:
            templates = json.load(f)
            standard_config = templates["标准训练"]["config"]
    except (FileNotFoundError, KeyError):
        # 如果无法加载配置文件，使用硬编码的标准配置
        standard_config = {
            "training": {
                "population_size": 20,
                "generations": 20,
                "runs_per_individual": 3,
                "save_file": "dino_population.json",
                "checkpoint_interval": 5,
                "checkpoint_dir": "checkpoints",
                "max_checkpoints": 10
            },
            "genetic": {
                "mutation_rate": 0.1,
                "mutation_scale": 0.2,
                "tournament_size": 3,
                "elite_count": 3,
                "elite_diversity_threshold": 0.1
            },
            "game": {
                "window_width": 800,
                "window_height": 600,
                "delay": 0.01,
                "simulation_mode": False
            }
        }
    
    config = {
        "training": {},
        "genetic": {},
        "game": {}
    }
    
    # 训练参数
    print("\n--- 训练参数 ---")
    try:
        default_pop_size = standard_config["training"]["population_size"]
        pop_size = input(f"种群大小 (默认: {default_pop_size}): ").strip()
        config["training"]["population_size"] = int(pop_size) if pop_size else default_pop_size
        
        default_generations = standard_config["training"]["generations"]
        generations = input(f"训练代数 (默认: {default_generations}): ").strip()
        config["training"]["generations"] = int(generations) if generations else default_generations
        
        default_runs = standard_config["training"]["runs_per_individual"]
        runs = input(f"每个个体运行次数 (默认: {default_runs}): ").strip()
        config["training"]["runs_per_individual"] = int(runs) if runs else default_runs
        
        default_save_file = standard_config["training"]["save_file"]
        save_file = input(f"保存文件名 (默认: {default_save_file}): ").strip()
        config["training"]["save_file"] = save_file if save_file else default_save_file
        
        default_checkpoint_interval = standard_config["training"]["checkpoint_interval"]
        checkpoint_interval = input(f"检查点保存间隔 (默认: {default_checkpoint_interval}): ").strip()
        config["training"]["checkpoint_interval"] = int(checkpoint_interval) if checkpoint_interval else default_checkpoint_interval
        
        config["training"]["checkpoint_dir"] = standard_config["training"]["checkpoint_dir"]
        config["training"]["max_checkpoints"] = standard_config["training"]["max_checkpoints"]
        
    except ValueError:
        print("输入格式错误，使用默认训练参数")
        config["training"] = standard_config["training"].copy()
    
    # 遗传算法参数
    print("\n--- 遗传算法参数 ---")
    try:
        default_mutation_rate = standard_config["genetic"]["mutation_rate"]
        mutation_rate = input(f"变异率 (0.0-1.0, 默认: {default_mutation_rate}): ").strip()
        config["genetic"]["mutation_rate"] = float(mutation_rate) if mutation_rate else default_mutation_rate
        
        default_mutation_scale = standard_config["genetic"]["mutation_scale"]
        mutation_scale = input(f"变异幅度 (0.0-1.0, 默认: {default_mutation_scale}): ").strip()
        config["genetic"]["mutation_scale"] = float(mutation_scale) if mutation_scale else default_mutation_scale
        
        default_tournament_size = standard_config["genetic"]["tournament_size"]
        tournament_size = input(f"锦标赛大小 (默认: {default_tournament_size}): ").strip()
        config["genetic"]["tournament_size"] = int(tournament_size) if tournament_size else default_tournament_size
        
        default_elite_count = standard_config["genetic"]["elite_count"]
        elite_count = input(f"精英个体数量 (默认: {default_elite_count}): ").strip()
        config["genetic"]["elite_count"] = int(elite_count) if elite_count else default_elite_count
        
        default_elite_diversity = standard_config["genetic"]["elite_diversity_threshold"]
        elite_diversity = input(f"精英多样性阈值 (0.0-1.0, 默认: {default_elite_diversity}): ").strip()
        config["genetic"]["elite_diversity_threshold"] = float(elite_diversity) if elite_diversity else default_elite_diversity
        
    except ValueError:
        print("输入格式错误，使用默认遗传算法参数")
        config["genetic"] = standard_config["genetic"].copy()
    
    # 游戏参数
    print("\n--- 游戏参数 ---")
    try:
        default_delay = standard_config["game"]["delay"]
        delay = input(f"游戏延迟 (秒, 默认: {default_delay}): ").strip()
        config["game"]["delay"] = float(delay) if delay else default_delay
        
        config["game"]["window_width"] = standard_config["game"]["window_width"]
        config["game"]["window_height"] = standard_config["game"]["window_height"]
        config["game"]["simulation_mode"] = standard_config["game"]["simulation_mode"]
        
    except ValueError:
        print("输入格式错误，使用默认游戏参数")
        config["game"] = standard_config["game"].copy()
    
    print("\n✅ 自定义配置设置完成！")
    
    # 询问是否保存配置
    save_choice = input("\n是否保存此配置为模板？(y/n, 默认: n): ").strip().lower()
    if save_choice in ['y', 'yes', '是']:
        config_name = input("请输入配置名称: ").strip()
        if config_name:
            description = input("请输入配置描述 (可选): ").strip()
            save_custom_config(config_name, config, description if description else "用户自定义配置")
        else:
            print("配置名称不能为空，跳过保存")
    
    return config

def edit_config_template(config_name):
    """编辑已保存的配置模板"""
    templates = load_config_templates()
    if config_name not in templates:
        print(f"❌ 配置 '{config_name}' 不存在")
        return False
    
    # 显示当前配置
    print("\n当前配置:")
    print(json.dumps(templates[config_name], indent=4, ensure_ascii=False))
    
    # 让用户编辑
    print("\n请输入新的配置参数（直接回车保持当前值）:")
    new_config = get_user_input_config()
    
    # 合并新配置
    templates[config_name].update(new_config)
    
    # 保存更新
    templates_path = os.path.join(os.path.dirname(__file__), 'config_templates.json')
    with open(templates_path, 'w', encoding='utf-8') as f:
        json.dump(templates, f, indent=4, ensure_ascii=False)

def edit_config_template():
    """编辑现有配置模板"""
    templates = load_config_templates()
    if not templates:
        print("没有可编辑的配置模板")
        return None
    
    print("\n可编辑的配置:")
    template_names = list(templates.keys())
    editable_configs = []
    system_configs = ["快速测试", "标准训练", "高强度训练", "探索性训练"]
    
    # 显示可编辑的配置（包括系统预设和用户自定义）
    for name in template_names:
        editable_configs.append(name)
        desc = templates[name].get("description", "无描述")
        config_type = "[系统预设]" if name in system_configs else "[用户自定义]"
        print(f"{len(editable_configs)}. {name} - {desc} {config_type}")
    
    try:
        edit_choice = input("\n选择要编辑的配置 (输入数字，回车返回): ").strip()
        if not edit_choice:
            return None
        
        edit_idx = int(edit_choice) - 1
        if 0 <= edit_idx < len(editable_configs):
            config_to_edit = editable_configs[edit_idx]
            current_config = templates[config_to_edit]["config"]
            
            print(f"\n=== 编辑配置: {config_to_edit} ===")
            
            # 编辑描述
            current_description = templates[config_to_edit].get("description", "无描述")
            print(f"\n--- 配置描述 ---")
            new_description = input(f"配置描述 (当前: {current_description}): ").strip()
            if not new_description:
                new_description = current_description
            
            print("\n当前配置值（直接回车保持不变）:")
            
            # 创建新配置对象
            new_config = {
                "training": {},
                "genetic": {},
                "game": {}
            }
            
            # 编辑训练参数
            print("\n--- 训练参数 ---")
            try:
                current_pop = current_config["training"]["population_size"]
                pop_size = input(f"种群大小 (当前: {current_pop}): ").strip()
                new_config["training"]["population_size"] = int(pop_size) if pop_size else current_pop
                
                current_gen = current_config["training"]["generations"]
                generations = input(f"训练代数 (当前: {current_gen}): ").strip()
                new_config["training"]["generations"] = int(generations) if generations else current_gen
                
                current_runs = current_config["training"]["runs_per_individual"]
                runs = input(f"每个个体运行次数 (当前: {current_runs}): ").strip()
                new_config["training"]["runs_per_individual"] = int(runs) if runs else current_runs
                
                current_save = current_config["training"]["save_file"]
                save_file = input(f"保存文件名 (当前: {current_save}): ").strip()
                new_config["training"]["save_file"] = save_file if save_file else current_save
                
                current_checkpoint = current_config["training"]["checkpoint_interval"]
                checkpoint_interval = input(f"检查点保存间隔 (当前: {current_checkpoint}): ").strip()
                new_config["training"]["checkpoint_interval"] = int(checkpoint_interval) if checkpoint_interval else current_checkpoint
                
                new_config["training"]["checkpoint_dir"] = current_config["training"].get("checkpoint_dir", "checkpoints")
                new_config["training"]["max_checkpoints"] = current_config["training"].get("max_checkpoints", 10)
                
            except (ValueError, KeyError):
                print("输入格式错误，保持原有训练参数")
                new_config["training"] = current_config["training"]
            
            # 编辑遗传算法参数
            print("\n--- 遗传算法参数 ---")
            try:
                current_mut_rate = current_config["genetic"]["mutation_rate"]
                mutation_rate = input(f"变异率 (当前: {current_mut_rate}): ").strip()
                new_config["genetic"]["mutation_rate"] = float(mutation_rate) if mutation_rate else current_mut_rate
                
                current_mut_scale = current_config["genetic"]["mutation_scale"]
                mutation_scale = input(f"变异幅度 (当前: {current_mut_scale}): ").strip()
                new_config["genetic"]["mutation_scale"] = float(mutation_scale) if mutation_scale else current_mut_scale
                
                current_tournament = current_config["genetic"]["tournament_size"]
                tournament_size = input(f"锦标赛大小 (当前: {current_tournament}): ").strip()
                new_config["genetic"]["tournament_size"] = int(tournament_size) if tournament_size else current_tournament
                
                current_elite = current_config["genetic"]["elite_count"]
                elite_count = input(f"精英个体数量 (当前: {current_elite}): ").strip()
                new_config["genetic"]["elite_count"] = int(elite_count) if elite_count else current_elite
                
                current_diversity = current_config["genetic"]["elite_diversity_threshold"]
                elite_diversity = input(f"精英多样性阈值 (当前: {current_diversity}): ").strip()
                new_config["genetic"]["elite_diversity_threshold"] = float(elite_diversity) if elite_diversity else current_diversity
                
            except (ValueError, KeyError):
                print("输入格式错误，保持原有遗传算法参数")
                new_config["genetic"] = current_config["genetic"]
            
            # 编辑游戏参数
            print("\n--- 游戏参数 ---")
            try:
                current_delay = current_config["game"]["delay"]
                delay = input(f"游戏延迟 (当前: {current_delay}): ").strip()
                new_config["game"]["delay"] = float(delay) if delay else current_delay
                
                new_config["game"]["window_width"] = current_config["game"].get("window_width", 800)
                new_config["game"]["window_height"] = current_config["game"].get("window_height", 600)
                new_config["game"]["simulation_mode"] = current_config["game"].get("simulation_mode", False)
                
            except (ValueError, KeyError):
                print("输入格式错误，保持原有游戏参数")
                new_config["game"] = current_config["game"]
            
            # 询问保存选项
            print("\n=== 保存选项 ===")
            print("1. 覆盖原配置")
            print("2. 另存为新配置")
            print("3. 不保存，直接使用")
            
            save_option = input("选择保存方式 (1/2/3): ").strip()
            
            if save_option == "1":
                # 覆盖原配置
                save_custom_config(config_to_edit, new_config, new_description)
                print(f"✅ 配置 '{config_to_edit}' 已更新")
            elif save_option == "2":
                # 另存为新配置
                new_name = input("请输入新配置名称: ").strip()
                if new_name:
                    save_as_description = input("请输入新配置描述 (可选): ").strip()
                    save_custom_config(new_name, new_config, save_as_description if save_as_description else "用户自定义配置")
                else:
                    print("配置名称不能为空，跳过保存")
            elif save_option == "3":
                print("✅ 配置编辑完成，将直接使用此配置")
            else:
                print("无效选择，跳过保存")
            
            return new_config
        else:
            print("无效选择")
            return None
    except ValueError:
        print("无效输入")
        return None

def select_config_template():
    """选择配置模板和运行模式"""
    while True:
        templates = load_config_templates()
        if not templates:
            return None, None
        
        print("\n=== 配置和模式选择 ===")
        print("\n📋 可用的配置模板:")
        template_names = list(templates.keys())
        for i, name in enumerate(template_names, 1):
            desc = templates[name].get("description", "无描述")
            tag = templates[name].get("tag", "未知")
            print(f"{i}. {name} - {desc} [{tag}]")
        
        # 添加功能选项
        print(f"\n🔧 配置管理选项:")
        print(f"{len(template_names) + 1}. 自定义输入 - 手动输入所有配置参数")
        print(f"{len(template_names) + 2}. 编辑配置 - 编辑现有配置模板")
        print(f"{len(template_names) + 3}. 删除配置 - 删除已保存的配置模板")
        
        # 添加运行模式选项
        print(f"\n🎮 运行模式选项:")
        print(f"{len(template_names) + 4}. 训练模式 - 使用遗传算法训练AI")
        print(f"{len(template_names) + 5}. 展示模式 - 使用历史最佳个体进行演示")
        
        try:
            choice = input("\n请选择配置模板或操作 (输入数字，回车使用默认配置): ").strip()
            if not choice:
                return None, None
            
            choice_idx = int(choice) - 1
            if choice_idx == len(template_names):  # 选择了自定义输入
                config = get_user_input_config()
                if config:
                    # 询问运行模式
                    mode = ask_run_mode()
                    return config, mode
                continue
            elif choice_idx == len(template_names) + 1:  # 选择了编辑配置
                config = edit_config_template()
                if config:
                    # 询问运行模式
                    mode = ask_run_mode()
                    return config, mode
                continue
            elif choice_idx == len(template_names) + 2:  # 选择了删除配置
                print("\n可删除的配置:")
                # 显示可删除的配置（排除系统预设配置）
                deletable_configs = []
                system_configs = ["快速测试", "标准训练", "高强度训练", "探索性训练"]
                
                for i, name in enumerate(template_names, 1):
                    if name not in system_configs:
                        deletable_configs.append(name)
                        print(f"{len(deletable_configs)}. {name}")
                
                if not deletable_configs:
                    print("没有可删除的自定义配置")
                    continue
                
                try:
                    del_choice = input("\n选择要删除的配置 (输入数字，回车返回): ").strip()
                    if not del_choice:
                        continue
                    
                    del_idx = int(del_choice) - 1
                    if 0 <= del_idx < len(deletable_configs):
                        config_to_delete = deletable_configs[del_idx]
                        confirm = input(f"确认删除配置 '{config_to_delete}'？(y/n): ").strip().lower()
                        if confirm in ['y', 'yes', '是']:
                            delete_config_template(config_to_delete)
                        continue
                    else:
                        print("无效选择")
                        continue
                except ValueError:
                    print("无效输入")
                    continue
            elif choice_idx == len(template_names) + 3:  # 选择了训练模式
                # 先选择配置
                print("\n请先选择训练配置:")
                for i, name in enumerate(template_names, 1):
                    desc = templates[name].get("description", "无描述")
                    tag = templates[name].get("tag", "未知")
                    print(f"{i}. {name} - {desc} [{tag}]")
                
                config_choice = input("\n选择配置 (输入数字): ").strip()
                try:
                    config_idx = int(config_choice) - 1
                    if 0 <= config_idx < len(template_names):
                        selected_name = template_names[config_idx]
                        print(f"已选择配置: {selected_name}")
                        return templates[selected_name]["config"], "train"
                    else:
                        print("无效选择")
                        continue
                except ValueError:
                    print("无效输入")
                    continue
            elif choice_idx == len(template_names) + 4:  # 选择了展示模式
                # 展示模式直接使用标准训练配置
                if "标准训练" in templates:
                    print("\n展示模式使用标准训练配置")
                    return templates["标准训练"]["config"], "demo"
                else:
                    # 如果没有标准训练配置，使用第一个可用配置
                    first_config = list(templates.keys())[0]
                    print(f"\n展示模式使用 {first_config} 配置")
                    return templates[first_config]["config"], "demo"
            elif 0 <= choice_idx < len(template_names):
                selected_name = template_names[choice_idx]
                print(f"已选择配置模板: {selected_name}")
                # 询问运行模式
                mode = ask_run_mode()
                return templates[selected_name]["config"], mode
            else:
                print("无效选择，使用默认配置")
                return None, None
        except ValueError:
            print("无效输入，使用默认配置")
            return None, None

def get_score_emoji(score):
    """根据分数返回相应的表情包"""
    if score >= 2000:
        return "🏆👑🎉"  # 超级高分
    elif score >= 1500:
        return "🥇🔥💪"  # 很高分
    elif score >= 1000:
        return "🥈⭐😎"  # 高分
    elif score >= 500:
        return "🥉👍😊"  # 中等分数
    elif score >= 200:
        return "👌😐📈"  # 一般分数
    elif score >= 100:
        return "😅💦🤔"  # 较低分数
    elif score >= 50:
        return "😰😵💀"  # 很低分数
    else:
        return "💀👻😭"  # 极低分数

def ask_run_mode():
    """询问运行模式"""
    print("\n🎮 请选择运行模式:")
    print("1. 训练模式 - 使用遗传算法训练AI")
    print("2. 展示模式 - 使用历史最佳个体进行演示")
    
    while True:
        try:
            mode_choice = input("\n请选择模式 (1 或 2): ").strip()
            if mode_choice == '1':
                return "train"
            elif mode_choice == '2':
                return "demo"
            else:
                print("请输入有效选择 (1 或 2)")
        except KeyboardInterrupt:
            print("\n程序退出")
            return None

def load_config():
    """加载配置文件和运行模式"""
    # 首先尝试选择配置模板和运行模式
    template_config, run_mode = select_config_template()
    if template_config:
        config = template_config
    else:
        # 使用默认配置（标准训练配置）
        print("使用默认配置（标准训练）")
        config = {
            "training": {
                "population_size": 20,
                "generations": 20,
                "runs_per_individual": 3,
                "save_file": "dino_population.json",
                "checkpoint_interval": 5,
                "checkpoint_dir": "checkpoints",
                "max_checkpoints": 10
            },
            "genetic": {
                "mutation_rate": 0.1,
                "mutation_scale": 0.2,
                "tournament_size": 3,
                "elite_count": 3,
                "elite_diversity_threshold": 0.1
            },
            "game": {
                "window_width": 800,
                "window_height": 600,
                "delay": 0.01,
                "simulation_mode": False
            }
        }
        # 询问运行模式
        run_mode = ask_run_mode()
    
    # 验证配置
    errors = validate_config(config)
    if errors:
        print("\n⚠️  配置验证失败:")
        for error in errors:
            print(f"   - {error}")
        print("\n请修正配置后重新运行程序")
        exit(1)
    
    print("\n✅ 配置验证通过")
    return config, run_mode

# 游戏控制类
class DinoGame:
    def __init__(self, config):
        print("使用Chrome浏览器模式")
        self.simulation_mode = False
        
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--no-default-browser-check")
        chrome_options.add_argument("--disable-default-apps")
        # chrome_options.add_argument("--headless")  # 移除headless模式，因为chrome://dino在headless模式下无法访问
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        # 使用随机端口避免冲突
        import random
        debug_port = random.randint(9000, 9999)
        chrome_options.add_argument(f"--remote-debugging-port={debug_port}")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        # 设置用户代理
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")
        
        # 尝试多种方式初始化ChromeDriver
        print("正在初始化ChromeDriver...")
        driver_initialized = False
        
        # 方法1: 优先使用本地ChromeDriver
        local_chromedriver_path = "/Users/zhangyunjian/felixspace/rat/python/遗传/谷歌小恐龙AI/chromedriver-mac-arm64/chromedriver"
        if os.path.exists(local_chromedriver_path):
            try:
                print(f"使用本地ChromeDriver: {local_chromedriver_path}")
                print(f"使用调试端口: {debug_port}")
                service = Service(local_chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("本地ChromeDriver初始化成功")
                driver_initialized = True
            except Exception as e:
                print(f"本地ChromeDriver初始化失败: {e}")
        
        # 方法2: 仅在本地ChromeDriver不可用时，使用webdriver-manager下载最新版本
        if not driver_initialized:
            try:
                print("本地ChromeDriver不可用，使用webdriver-manager下载最新版本...")
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("webdriver-manager ChromeDriver初始化成功")
                driver_initialized = True
            except Exception as e:
                print(f"webdriver-manager初始化失败: {e}")
        
        if not driver_initialized:
            print("\n=== Chrome浏览器初始化失败 ===")
            print("请检查以下问题:")
            print("1. Chrome浏览器是否已正确安装")
            print("2. Chrome版本是否与ChromeDriver兼容")
            print("3. 是否有其他Chrome进程在运行")
            print("4. 系统权限是否允许启动Chrome")
            raise Exception("Chrome浏览器初始化失败，无法继续运行")
        
        # 设置窗口大小
        self.driver.set_window_size(
            config["game"]["window_width"], 
            config["game"]["window_height"]
        )
        
        # 打开Chrome恐龙游戏
        print("正在打开在线版本的Chrome恐龙游戏...")
        try:
            # 直接使用在线版本的恐龙游戏
            self.driver.get("https://chromedino.com/")
            print("成功连接到在线版本的恐龙游戏")
        except Exception as e:
            print(f"无法连接到在线版本的恐龙游戏: {e}")
            print("尝试访问chrome://dino...")
            try:
                self.driver.get("chrome://dino")
                print("成功连接到chrome://dino")
            except Exception as e2:
                print(f"无法访问chrome://dino: {e2}")
                raise Exception("无法连接到任何版本的恐龙游戏，请检查网络连接")
        
        # 等待游戏加载
        time.sleep(2)
        print("游戏加载完成")
        
        # 初始化游戏状态
        self.is_playing = False
        self.current_speed = 6
        self.delay = config["game"]["delay"]
    
    def start_game(self):
        """开始游戏"""
        if not self.is_playing:
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
            self.is_playing = True
            time.sleep(0.5)  # 等待游戏开始
    
    def jump(self):
        """恐龙跳跃"""
        try:
            self.driver.execute_script("Runner.instance_.tRex.startJump(Runner.instance_.currentSpeed)")
        except:
            try:
                self.driver.execute_script("window.Runner.instance_.tRex.startJump(window.Runner.instance_.currentSpeed)")
            except:
                # 备选方案：模拟空格键或上箭头键
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
    
    def duck(self):
        """恐龙下蹲"""
        try:
            self.driver.execute_script("Runner.instance_.tRex.setDuck(true)")
            self.is_ducking = True
        except:
            try:
                self.driver.execute_script("window.Runner.instance_.tRex.setDuck(true)")
                self.is_ducking = True
            except:
                # 备选方案：模拟下箭头键
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)
                self.is_ducking = True
    
    def release_duck(self):
        """释放下蹲"""
        try:
            self.driver.execute_script("Runner.instance_.tRex.setDuck(false)")
            self.is_ducking = False
        except:
            try:
                self.driver.execute_script("window.Runner.instance_.tRex.setDuck(false)")
                self.is_ducking = False
            except:
                # 备选方案：释放下箭头键
                action = webdriver.ActionChains(self.driver)
                action.key_up(Keys.ARROW_DOWN).perform()
                self.is_ducking = False
                
    def start_duck(self):
        """开始持续下蹲"""
        if not hasattr(self, 'is_ducking'):
            self.is_ducking = False
        if not self.is_ducking:
            self.duck()
            
    def stop_duck(self):
        """停止持续下蹲"""
        if hasattr(self, 'is_ducking') and self.is_ducking:
            self.release_duck()
    
    def get_score(self):
        """获取当前分数"""
        try:
            # 尝试多种方式获取分数
            score = self.driver.execute_script("""
                var runner = Runner.instance_ || (window.Runner ? window.Runner.instance_ : null);
                if (runner) {
                    // 方法1: 从distanceMeter获取
                    if (runner.distanceMeter && runner.distanceMeter.digits) {
                        var digits = runner.distanceMeter.digits;
                        if (Array.isArray(digits)) {
                            return parseInt(digits.join('')) || 0;
                        }
                    }
                    
                    // 方法2: 从distanceRan计算
                    if (runner.distanceRan) {
                        return Math.floor(runner.distanceRan / 10);
                    }
                    
                    // 方法3: 从DOM元素获取
                    var scoreElement = document.querySelector('.score') || 
                                     document.querySelector('#score') ||
                                     document.querySelector('[class*="score"]');
                    if (scoreElement) {
                        return parseInt(scoreElement.textContent.replace(/[^0-9]/g, '')) || 0;
                    }
                }
                return 0;
            """)
            
            return score if score is not None else 0
            
        except Exception as e:
            print(f"获取分数失败: {e}")
            return 0
    
    def is_game_over(self):
        """检查游戏是否结束"""
        try:
            # 尝试chrome://dino的接口
            return self.driver.execute_script("return Runner.instance_.crashed")
        except:
            try:
                # 尝试在线版本的接口
                crashed = self.driver.execute_script("return window.Runner ? window.Runner.instance_.crashed : false")
                if crashed is not None:
                    return crashed
                # 备选方案：检查游戏结束画面
                game_over_element = self.driver.find_elements(By.CLASS_NAME, "game-over")
                return len(game_over_element) > 0
            except:
                return False
    
    def restart(self):
        """重新开始游戏"""
        print("正在重启游戏...")
        try:
            # 检查游戏是否已经开始
            game_started = self.driver.execute_script("""
                var runner = Runner.instance_ || (window.Runner ? window.Runner.instance_ : null);
                return runner ? runner.activated : false;
            """)
            # print(f"游戏是否已激活: {game_started}")
            
            if not game_started:
                # 如果游戏未开始，先点击开始
                # print("游戏未开始，尝试启动游戏...")
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
                time.sleep(2)
            
            # 尝试重启游戏
            self.driver.execute_script("Runner.instance_.restart()")
            # print("使用 Runner.instance_.restart() 重启成功")
        except:
            try:
                self.driver.execute_script("window.Runner ? window.Runner.instance_.restart() : null")
                # print("使用 window.Runner.instance_.restart() 重启成功")
            except:
                # 备选方案：按空格键重新开始
                # print("使用空格键重启游戏")
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
        
        self.is_playing = True
        time.sleep(2)  # 等待游戏重新开始
        
        # 验证游戏状态
        try:
            score = self.get_score()
            game_over = self.is_game_over()
            # print(f"重启后状态: 分数={score}, 游戏结束={game_over}")
        except Exception as e:
            print(f"获取重启后状态失败: {e}")
    
    def get_game_state(self):
        """获取游戏状态"""
        try:
            # 首先检查游戏是否正在运行
            game_info = self.driver.execute_script("""
                var runner = Runner.instance_ || (window.Runner ? window.Runner.instance_ : null);
                if (runner) {
                    return {
                        activated: runner.activated,
                        playing: runner.playing,
                        crashed: runner.crashed,
                        currentSpeed: runner.currentSpeed || 6,
                        distanceRan: runner.distanceRan || 0
                    };
                }
                return null;
            """)
            
            # print(f"游戏信息: {game_info}")
            
            if not game_info:
                print("无法获取游戏实例，尝试启动游戏")
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)
                time.sleep(1)
                return {
                    'dino': {'x': 50, 'y': 130, 'width': 40, 'height': 50},
                    'obstacles': [],
                    'speed': self.current_speed,
                    'score': 0
                }
            
            # 更新当前速度
            self.current_speed = game_info.get('currentSpeed', 6)
            
            # 获取分数（基于距离）
            distance_score = int(game_info.get('distanceRan', 0) / 10)  # 距离转换为分数
            
            # 检测障碍物
            obstacles = []
            try:
                obstacle_data = self.driver.execute_script("""
                    var runner = Runner.instance_ || (window.Runner ? window.Runner.instance_ : null);
                    if (runner && runner.horizon && runner.horizon.obstacles) {
                        var obstacles = [];
                        for (var i = 0; i < runner.horizon.obstacles.length; i++) {
                            var obstacle = runner.horizon.obstacles[i];
                            if (obstacle.xPos > 0) {  // 只获取屏幕内的障碍物
                                // 尝试多种方式获取高度
                                var height = obstacle.height || obstacle.size || (obstacle.typeConfig && obstacle.typeConfig.height) || 40;
                                var width = obstacle.width || (obstacle.typeConfig && obstacle.typeConfig.width) || 20;
                                var type = 'CACTUS';
                                
                                // 直接从对象属性获取类型信息
                                if (obstacle.typeConfig && obstacle.typeConfig.type) {
                                    type = obstacle.typeConfig.type;
                                } else if (obstacle.type) {
                                    type = obstacle.type;
                                } else if (obstacle.constructor && obstacle.constructor.name) {
                                    // 从构造函数名称推断类型
                                    var constructorName = obstacle.constructor.name;
                                    if (constructorName.includes('Pterodactyl')) {
                                        type = 'PTERODACTYL';
                                    } else if (constructorName.includes('Cactus')) {
                                        type = 'CACTUS';
                                    }
                                } else if (obstacle.className) {
                                    // 从CSS类名推断类型
                                    if (obstacle.className.includes('pterodactyl')) {
                                        type = 'PTERODACTYL';
                                    } else if (obstacle.className.includes('cactus')) {
                                        type = 'CACTUS';
                                    }
                                } else {
                                    // 尝试从其他属性推断类型
                                    var yPos = obstacle.yPos || 0;
                                    var spritePos = obstacle.spritePos || obstacle.sourceXPos || 0;
                                    
                                    // 检查是否有特定的标识属性
                                    if (obstacle.isPterodactyl || obstacle.flying) {
                                        type = 'PTERODACTYL';
                                    } else if (obstacle.isCactus || obstacle.ground) {
                                        type = 'CACTUS';
                                    } else if (obstacle.animFrames && obstacle.animFrames.length > 1) {
                                        // 有动画帧的通常是翼龙
                                        type = 'PTERODACTYL';
                                    } else if (obstacle.collisionBoxes && obstacle.collisionBoxes.length > 0) {
                                        // 根据碰撞盒的数量和位置判断
                                        var firstBox = obstacle.collisionBoxes[0];
                                        if (firstBox && firstBox.y < 50) {
                                            type = 'PTERODACTYL';
                                        } else {
                                            type = 'CACTUS';
                                        }
                                    } else {
                                        // 最后根据Y位置判断（翼龙在空中，仙人掌在地面）
                                        if (yPos < 100) {
                                            type = 'PTERODACTYL';
                                        } else {
                                            type = 'CACTUS';
                                        }
                                    }
                                    
                                    // 如果确定是翼龙，进一步区分高低空
                                    if (type === 'PTERODACTYL') {
                                        var dinoGroundY = 75;
                                        // 使用Y位置而非高度来判断
                                        if (yPos >= dinoGroundY - 10) {
                                            type = 'PTERODACTYL_LOW';  // 低空翼龙
                                        } else {
                                            type = 'PTERODACTYL_HIGH'; // 高空翼龙
                                        }
                                    }
                                }
                                
                                obstacles.push({
                                    x: obstacle.xPos,
                                    y: obstacle.yPos,
                                    width: width,
                                    height: height,
                                    type: type
                                });
                                
                                // 调试信息：输出障碍物信息
                                console.log('障碍物检测:', {
                                    x: obstacle.xPos,
                                    y: obstacle.yPos,
                                    width: width,
                                    height: height,
                                    type: type
                                });
                            }
                        }
                        return obstacles;
                    }
                    return [];
                """)
                
                # print(f"检测到 {len(obstacle_data)} 个障碍物")
                
                # 处理障碍物数据
                for obstacle in obstacle_data:
                    # 确保所有数值都是有效的
                    x = obstacle.get('x', 0)
                    y = obstacle.get('y', 0)
                    width = obstacle.get('width', 20)
                    height = obstacle.get('height', 40)
                    obstacle_type = obstacle.get('type', 'CACTUS')
                    
                    # 验证数值有效性
                    if x is not None and y is not None and width is not None and height is not None:
                        obstacles.append({
                            'x': float(x),
                            'y': float(y),
                            'width': float(width),
                            'height': float(height),
                            'type': str(obstacle_type)
                        })
                    
            except Exception as e:
                print(f"获取障碍物数据失败: {e}")
                obstacles = []
            
            # 获取恐龙位置
            try:
                dino_pos = self.driver.execute_script("""
                    var runner = Runner.instance_ || (window.Runner ? window.Runner.instance_ : null);
                    if (runner && runner.tRex) {
                        return {
                            x: runner.tRex.xPos,
                            y: runner.tRex.yPos,
                            width: runner.tRex.config ? runner.tRex.config.WIDTH : 40,
                            height: runner.tRex.config ? runner.tRex.config.HEIGHT : 50,
                            jumping: runner.tRex.jumping,
                            ducking: runner.tRex.ducking
                        };
                    }
                    return {x: 50, y: 130, width: 40, height: 50, jumping: false, ducking: false};
                """)
                # print(f"恐龙位置: {dino_pos}")
            except Exception as e:
                # print(f"获取恐龙位置失败: {e}")
                # 默认恐龙位置
                dino_pos = {'x': 50, 'y': 130, 'width': 40, 'height': 50, 'jumping': False, 'ducking': False}
        
        except Exception as e:
            # print(f"获取游戏状态时出错: {e}")
            # 返回默认状态
            return {
                'dino': {'x': 50, 'y': 130, 'width': 40, 'height': 50},
                'obstacles': [],
                'speed': self.current_speed,
                'score': 0
            }
        
        game_state = {
            'dino': dino_pos,
            'obstacles': obstacles,
            'speed': self.current_speed,
            'score': distance_score
        }
        
        # 每10步打印一次详细状态
        # if hasattr(self, '_debug_counter'):
            # self._debug_counter += 1
        # else:
            # self._debug_counter = 0
            
        # if self._debug_counter % 10 == 0:
            # print(f"游戏状态: 分数={distance_score}, 速度={self.current_speed}, 障碍物数量={len(obstacles)}")
            # if obstacles:
                # print(f"最近障碍物: x={obstacles[0]['x']}, type={obstacles[0]['type']}")
        
        return game_state
    
    def close(self):
        """关闭浏览器"""
        self.driver.quit()

# 个体类（DinosaurAI）
class DinosaurAI:
    def __init__(self, weights=None, bias=None, config=None):
        self.config = config or {}
        self.mutation_rate = self.config.get("mutation_rate", 0.1)
        self.mutation_scale = self.config.get("mutation_scale", 0.2)
        
        # 初始化权重和偏置
        # 输入特征：[距离下一个障碍物的距离, 障碍物宽度, 障碍物高度, 障碍物类型(0=仙人掌,1=翼龙), 游戏速度]
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 5)  
        else:
            self.weights = np.array(weights)
            
        # 跳跃和下蹲的偏置
        if bias is None:
            self.jump_bias = np.random.uniform(-1, 1)
            self.duck_bias = np.random.uniform(-1, 1)
        else:
            self.jump_bias = bias[0]
            self.duck_bias = bias[1]
    
    def relu(self, x):
        """ReLU激活函数"""
        return max(0, x)
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def predict(self, game_state):
        """基于游戏状态预测动作"""
        # 如果没有障碍物，不执行任何动作
        if not game_state['obstacles']:
            return {'jump': False, 'duck': False}
        
        # 获取最近的障碍物和恐龙状态
        obstacle = game_state['obstacles'][0]
        dino = game_state['dino']
        
        # 检查恐龙是否在跳跃状态
        is_jumping = dino.get('jumping', False)
        has_ducked_in_jump = dino.get('has_ducked_in_jump', False)
        
        # 安全获取特征值，确保不为None
        distance = float(obstacle.get('x', 0) - (dino.get('x', 0) + dino.get('width', 40)))
        obstacle_width = float(obstacle.get('width', 20))
        obstacle_height = float(obstacle.get('height', 40))
        
        # 处理新的翼龙类型
        obstacle_type_str = obstacle.get('type', 'CACTUS')
        if obstacle_type_str in ['PTERODACTYL', 'PTERODACTYL_LOW', 'PTERODACTYL_HIGH']:
            obstacle_type = 1.0
        else:
            obstacle_type = 0.0
            
        speed = float(game_state.get('speed', 6))
        
        # 特征向量
        features = np.array([distance, obstacle_width, obstacle_height, obstacle_type, speed])
        
        # 验证特征向量
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"警告：特征向量包含无效值: {features}")
            return {'jump': False, 'duck': False}
        
        try:
            # 计算跳跃和下蹲的决策值
            jump_value = np.dot(self.weights, features) + self.jump_bias
            duck_value = np.dot(self.weights * -0.5, features) + self.duck_bias  # 下蹲使用不同的权重
            
            # 应用激活函数
            jump_prob = self.sigmoid(jump_value)
            duck_prob = self.sigmoid(duck_value)
            
            # 决策逻辑
            jump = jump_prob > 0.5 and not is_jumping  # 只有在不跳跃时才能开始跳跃
            
            # 改进的决策逻辑：根据翼龙高度做出不同反应
            duck = False
            
            # 获取障碍物类型
            obstacle_type_str = obstacle.get('type', 'CACTUS')
            
            if is_jumping:
                # 跳跃中下蹲：当距离障碍物较近且需要快速落地时，且本次跳跃还未下蹲过
                if distance < 120 and duck_prob > 0.5 and not has_ducked_in_jump:
                    duck = True
            else:
                # 地面决策逻辑
                if obstacle_type_str == 'PTERODACTYL_LOW':
                    # 低空翼龙：需要跳跃，不下蹲
                    jump = jump_prob > 0.4  # 降低跳跃阈值，更容易跳跃
                    duck = False
                    print(f"检测到低空翼龙，执行跳跃！距离: {distance:.1f}, 跳跃概率: {jump_prob:.3f}")
                elif obstacle_type_str == 'PTERODACTYL_HIGH':
                    # 高空翼龙：需要下蹲
                    distance_threshold = max(0.3, 0.7 - distance / 200)
                    duck = duck_prob > distance_threshold and not jump
                    jump = False  # 确保不跳跃
                    if duck:
                        print(f"检测到高空翼龙，执行下蹲！距离: {distance:.1f}, 下蹲概率: {duck_prob:.3f}, 阈值: {distance_threshold:.3f}")
                elif obstacle_type_str == 'PTERODACTYL':
                    # 旧版翼龙类型，默认下蹲
                    distance_threshold = max(0.3, 0.8 - distance / 200)
                    duck = duck_prob > distance_threshold and not jump
                    if duck:
                        print(f"检测到翼龙，执行下蹲！距离: {distance:.1f}, 下蹲概率: {duck_prob:.3f}, 阈值: {distance_threshold:.3f}")
                else:
                    # 仙人掌：完全禁止下蹲，只能跳跃
                    duck = False
                    # 对仙人掌提高跳跃概率
                    if jump_prob > 0.4:
                        jump = True
            
            return {'jump': jump, 'duck': duck}
            
        except Exception as e:
            print(f"AI预测时出错: {e}")
            print(f"特征: {features}")
            print(f"权重: {self.weights}")
            return {'jump': False, 'duck': False}

    def mutate(self):
        """随机变异"""
        # 权重变异
        mask = np.random.random(self.weights.shape) < self.mutation_rate
        self.weights += mask * np.random.uniform(-self.mutation_scale, self.mutation_scale, self.weights.shape)
        
        # 偏置变异
        if random.random() < self.mutation_rate:
            self.jump_bias += np.random.uniform(-self.mutation_scale, self.mutation_scale)
        if random.random() < self.mutation_rate:
            self.duck_bias += np.random.uniform(-self.mutation_scale, self.mutation_scale)

    def to_dict(self):
        """将个体的基因保存为字典"""
        return {
            "weights": self.weights.tolist(),
            "bias": [self.jump_bias, self.duck_bias]
        }

    @staticmethod
    def from_dict(data, config=None):
        """从字典加载个体"""
        return DinosaurAI(weights=data["weights"], bias=data["bias"], config=config)

# 遗传算法类
class GeneticAlgorithm:
    def __init__(self, config):
        self.config = config
        self.population_size = config["training"]["population_size"]
        self.save_file = config["training"]["save_file"]
        self.tournament_size = config["genetic"]["tournament_size"]
        self.elite_count = config["genetic"]["elite_count"]
        self.elite_diversity_threshold = config["genetic"].get("elite_diversity_threshold", 0.1)
        
        # 检查点保存配置
        self.checkpoint_interval = config["training"].get("checkpoint_interval", 5)
        self.checkpoint_dir = config["training"].get("checkpoint_dir", "checkpoints")
        self.max_checkpoints = config["training"].get("max_checkpoints", 10)
        
        # 创建检查点目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # 初始化种群
        genetic_config = config["genetic"]
        self.population = [DinosaurAI(config=genetic_config) for _ in range(self.population_size)]
        self.generation = 0
        self.best_fitness = 0
        self.best_individual = None
        self.training_history = []

    def select(self, fitness_scores):
        """选择操作 - 锦标赛选择"""
        selected = []
        for _ in range(self.population_size // 2):
            # 随机选择tournament_size个个体进行锦标赛
            tournament = random.sample(list(zip(fitness_scores, self.population)), self.tournament_size)
            # 选择适应度最高的个体
            winner = max(tournament, key=lambda x: x[0])
            selected.append(winner[1])
        return selected

    def crossover(self, parent1, parent2):
        """交叉操作 - 均匀交叉"""
        child = DinosaurAI(config=self.config["genetic"])
        # 对每个权重，有50%的概率从父亲1继承，50%的概率从父亲2继承
        for i in range(len(parent1.weights)):
            if random.random() < 0.5:
                child.weights[i] = parent1.weights[i]
            else:
                child.weights[i] = parent2.weights[i]
        
        # 偏置的交叉
        if random.random() < 0.5:
            child.jump_bias = parent1.jump_bias
        else:
            child.jump_bias = parent2.jump_bias
            
        if random.random() < 0.5:
            child.duck_bias = parent1.duck_bias
        else:
            child.duck_bias = parent2.duck_bias
            
        return child

    def calculate_diversity(self, individual1, individual2):
        """计算两个个体之间的多样性（权重差异）"""
        weights1 = individual1.weights.flatten()
        weights2 = individual2.weights.flatten()
        return np.linalg.norm(weights1 - weights2)
    
    def select_diverse_elites(self, fitness_scores):
        """选择多样化的精英个体"""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elites = []
        diversity_threshold = self.elite_diversity_threshold
        
        # 总是保留最佳个体
        elites.append(self.population[sorted_indices[0]])
        
        # 选择其他精英个体，确保多样性
        for i in range(1, len(sorted_indices)):
            candidate = self.population[sorted_indices[i]]
            is_diverse = True
            
            # 检查与已选择精英的多样性
            for elite in elites:
                if self.calculate_diversity(candidate, elite) < diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                elites.append(candidate)
                if len(elites) >= self.elite_count:
                    break
        
        # 如果没有足够的多样化精英，填充剩余位置
        while len(elites) < self.elite_count and len(elites) < len(self.population):
            for i in range(len(sorted_indices)):
                candidate = self.population[sorted_indices[i]]
                if candidate not in elites:
                    elites.append(candidate)
                    break
        
        return elites
    
    def evolve(self, fitness_scores):
        """进化到下一代"""
        # 更新最佳个体
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[max_fitness_idx]
            self.best_individual = self.population[max_fitness_idx]
        
        # 选择操作
        selected = self.select(fitness_scores)
        
        # 创建新一代
        new_population = []
        
        # 增强的精英保留策略 - 保留多样化的精英个体
        elites = self.select_diverse_elites(fitness_scores)
        new_population.extend(elites)
        
        print(f"保留了 {len(elites)} 个精英个体")
        
        # 通过交叉和变异生成其余个体
        while len(new_population) < self.population_size:
            # 随机选择两个父母
            parent1, parent2 = random.sample(selected, 2)
            # 生成子代
            child = self.crossover(parent1, parent2)
            # 变异
            child.mutate()
            # 添加到新种群
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # 自动检查点保存
        if self.generation % self.checkpoint_interval == 0:
            self.save_checkpoint()

    def save_population(self):
        """保存种群到文件"""
        with open(self.save_file, "w") as f:
            data = {
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "best_individual": self.best_individual.to_dict() if self.best_individual else None,
                "population": [individual.to_dict() for individual in self.population]
            }
            json.dump(data, f)
        # print(f"种群保存到 {self.save_file}")

    def load_population(self):
        """从文件加载种群"""
        try:
            with open(self.save_file, "r") as f:
                data = json.load(f)
                self.generation = data["generation"]
                self.best_fitness = data["best_fitness"]
                if data["best_individual"]:
                    self.best_individual = DinosaurAI.from_dict(data["best_individual"], config=self.config["genetic"])
                self.population = [DinosaurAI.from_dict(ind, config=self.config["genetic"]) for ind in data["population"]]
            print(f"种群从 {self.save_file} 加载成功，当前代数: {self.generation}，最佳适应度: {self.best_fitness}")
            return True
        except FileNotFoundError:
            print(f"文件 {self.save_file} 不存在，初始化新种群")
            return False
    
    def save_checkpoint(self):
        """保存训练检查点"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{self.generation}_{timestamp}.json")
        
        checkpoint_data = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "population": [individual.to_dict() for individual in self.population],
            "training_history": self.training_history,
            "config": self.config,
            "timestamp": timestamp
        }
        
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        
        print(f"检查点保存到: {checkpoint_file}")
        
        # 清理旧的检查点文件
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """清理旧的检查点文件，只保留最新的N个"""
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
            checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)), reverse=True)
            
            # 删除超出限制的旧文件
            for old_file in checkpoint_files[self.max_checkpoints:]:
                old_path = os.path.join(self.checkpoint_dir, old_file)
                os.remove(old_path)
                print(f"删除旧检查点: {old_file}")
        except Exception as e:
            print(f"清理检查点文件时出错: {e}")
    
    def load_latest_checkpoint(self):
        """加载最新的检查点"""
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
            if not checkpoint_files:
                print("没有找到检查点文件")
                return False
            
            # 按创建时间排序，获取最新的
            latest_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)))
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
            
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            
            self.generation = data["generation"]
            self.best_fitness = data["best_fitness"]
            if data["best_individual"]:
                self.best_individual = DinosaurAI.from_dict(data["best_individual"], config=self.config["genetic"])
            self.population = [DinosaurAI.from_dict(ind, config=self.config["genetic"]) for ind in data["population"]]
            self.training_history = data.get("training_history", [])
            
            print(f"从检查点恢复: {latest_file}")
            print(f"当前代数: {self.generation}，最佳适应度: {self.best_fitness}")
            return True
            
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            return False
    
    def list_checkpoints(self):
        """列出所有可用的检查点"""
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
            if not checkpoint_files:
                print("没有找到检查点文件")
                return []
            
            checkpoints = []
            for file in checkpoint_files:
                file_path = os.path.join(self.checkpoint_dir, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    checkpoints.append({
                        "file": file,
                        "generation": data["generation"],
                        "best_fitness": data["best_fitness"],
                        "timestamp": data.get("timestamp", "未知")
                    })
                except:
                    continue
            
            # 按代数排序
            checkpoints.sort(key=lambda x: x["generation"], reverse=True)
            return checkpoints
            
        except Exception as e:
             print(f"列出检查点时出错: {e}")
             return []
    
    def generate_training_report(self):
        """生成详细的训练统计报告"""
        if not self.training_history:
            print("没有训练历史数据")
            return
        
        print("\n" + "="*80)
        print("🎯 训练统计报告")
        print("="*80)
        
        # 基本统计信息
        total_generations = len(self.training_history)
        total_time = sum(record['generation_time'] for record in self.training_history)
        avg_time_per_gen = total_time / total_generations if total_generations > 0 else 0
        
        print(f"📊 基本统计:")
        print(f"   总训练代数: {total_generations}")
        print(f"   总训练时间: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"   平均每代时间: {avg_time_per_gen:.2f} 秒")
        print(f"   最终最佳适应度: {self.best_fitness:.2f}")
        
        # 改进统计
        improvements = sum(1 for record in self.training_history if record['improved'])
        improvement_rate = improvements / total_generations * 100 if total_generations > 0 else 0
        
        print(f"\n📈 改进统计:")
        print(f"   改进次数: {improvements}/{total_generations}")
        print(f"   改进率: {improvement_rate:.1f}%")
        
        # 适应度统计
        best_fitnesses = [record['best_fitness'] for record in self.training_history]
        avg_fitnesses = [record['avg_fitness'] for record in self.training_history]
        
        print(f"\n🏆 适应度统计:")
        print(f"   最高适应度: {max(best_fitnesses):.2f}")
        print(f"   最低适应度: {min(best_fitnesses):.2f}")
        print(f"   平均适应度: {sum(avg_fitnesses)/len(avg_fitnesses):.2f}")
        print(f"   适应度标准差: {np.std(best_fitnesses):.2f}")
        
        # 性能统计
        generation_times = [record['generation_time'] for record in self.training_history]
        print(f"\n⏱️  性能统计:")
        print(f"   最快一代: {min(generation_times):.2f} 秒")
        print(f"   最慢一代: {max(generation_times):.2f} 秒")
        print(f"   时间标准差: {np.std(generation_times):.2f} 秒")
        
        # 趋势分析
        if total_generations >= 5:
            recent_best = best_fitnesses[-5:]
            early_best = best_fitnesses[:5]
            recent_avg = sum(recent_best) / len(recent_best)
            early_avg = sum(early_best) / len(early_best)
            trend = recent_avg - early_avg
            
            print(f"\n📊 趋势分析 (最近5代 vs 前5代):")
            print(f"   早期平均最佳适应度: {early_avg:.2f}")
            print(f"   近期平均最佳适应度: {recent_avg:.2f}")
            print(f"   趋势: {'上升' if trend > 0 else '下降' if trend < 0 else '平稳'} ({trend:+.2f})")
        
        # 保存报告到文件
        self.save_training_report()
        
        print("\n" + "="*80)
    
    def save_training_report(self):
        """保存训练报告到文件"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"training_report_{timestamp}.json"
            
            report_data = {
                "timestamp": timestamp,
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "training_history": self.training_history,
                "summary": {
                    "total_generations": len(self.training_history),
                    "total_time": sum(record['generation_time'] for record in self.training_history),
                    "improvements": sum(1 for record in self.training_history if record['improved']),
                    "final_best_fitness": self.best_fitness
                },
                "config": self.config
            }
            
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)
            
            print(f"📄 训练报告已保存到: {report_file}")
            
        except Exception as e:
            print(f"保存训练报告时出错: {e}")

# 模拟游戏类
class SimulatedDinoGame:
    def __init__(self, config):
        self.config = config
        self.is_playing = False
        self.current_speed = 6
        self.delay = config["game"]["delay"]
        self.score = 0
        self.game_over = False
        self.obstacles = []
        self.next_obstacle_time = 0
        self.dino_pos = {"x": 50, "y": 130, "width": 40, "height": 50}
        self.time_elapsed = 0
        self.jump_height = 0
        self.is_ducking = False
        self.has_ducked_in_jump = False  # 跟踪当前跳跃中是否已经下蹲过
        
    def start_game(self):
        """开始游戏"""
        self.is_playing = True
        self.score = 0
        self.game_over = False
        self.obstacles = []
        self.next_obstacle_time = random.uniform(1, 3)
        self.time_elapsed = 0
        print("模拟游戏开始")
    
    def jump(self):
        """恐龙跳跃"""
        if self.jump_height == 0:  # 只有在地面上才能跳跃
            self.jump_height = 10
            self.has_ducked_in_jump = False  # 重置跳跃中的下蹲标记
    
    def duck(self):
        """恐龙下蹲"""
        self.is_ducking = True
        if self.jump_height > 0:  # 如果在跳跃中下蹲，设置标记
            self.has_ducked_in_jump = True
    
    def release_duck(self):
        """释放下蹲"""
        self.is_ducking = False
    
    def get_score(self):
        """获取当前分数"""
        return int(self.score)
    
    def is_game_over(self):
        """检查游戏是否结束"""
        return self.game_over
    
    def restart(self):
        """重新开始游戏"""
        self.start_game()
    
    def update_game_state(self):
        """更新游戏状态"""
        # 更新时间和分数
        self.time_elapsed += self.delay
        self.score += self.current_speed * self.delay
        
        # 更新恐龙位置（跳跃动画）
        if self.jump_height > 0:
            self.dino_pos["y"] = 130 - (self.jump_height * 5)  # 跳跃高度
            # 如果在跳跃中下蹲，增加下降速度实现快速落地
            gravity = 1.0 if self.is_ducking else 0.5
            self.jump_height -= gravity  # 重力（下蹲时重力加倍）
            if self.jump_height <= 0:
                self.jump_height = 0
                self.dino_pos["y"] = 130  # 回到地面
        
        # 更新障碍物
        # 生成新障碍物
        if self.time_elapsed >= self.next_obstacle_time:
            obstacle_type = "CACTUS" if random.random() < 0.7 else "PTERODACTYL"
            y_pos = 130 if obstacle_type == "CACTUS" else random.choice([100, 130])
            width = random.randint(20, 40)
            height = random.randint(40, 70) if obstacle_type == "CACTUS" else 30
            
            self.obstacles.append({
                "x": 800,  # 屏幕右侧
                "y": y_pos,
                "width": width,
                "height": height,
                "type": obstacle_type
            })
            
            # 设置下一个障碍物出现的时间
            self.next_obstacle_time = self.time_elapsed + random.uniform(1, 3)
        
        # 移动障碍物
        for obstacle in self.obstacles:
            obstacle["x"] -= self.current_speed
        
        # 移除屏幕外的障碍物
        self.obstacles = [obs for obs in self.obstacles if obs["x"] > -obs["width"]]
        
        # 检测碰撞
        if not self.game_over:
            dino_hitbox = {
                "x": self.dino_pos["x"],
                "y": self.dino_pos["y"],
                "width": self.dino_pos["width"] * (0.6 if self.is_ducking else 1),
                "height": self.dino_pos["height"] * (0.5 if self.is_ducking else 1)
            }
            
            for obstacle in self.obstacles:
                if (dino_hitbox["x"] < obstacle["x"] + obstacle["width"] and
                    dino_hitbox["x"] + dino_hitbox["width"] > obstacle["x"] and
                    dino_hitbox["y"] < obstacle["y"] + obstacle["height"] and
                    dino_hitbox["y"] + dino_hitbox["height"] > obstacle["y"]):
                    self.game_over = True
                    break
        
        # 随着分数增加，增加速度
        if int(self.score) % 100 == 0 and int(self.score) > 0:
            self.current_speed = min(self.current_speed + 0.01, 13)
    
    def get_game_state(self):
        """获取游戏状态"""
        # 更新游戏状态
        self.update_game_state()
        
        # 添加恐龙状态信息
        dino_state = self.dino_pos.copy()
        dino_state['jumping'] = self.jump_height > 0
        dino_state['ducking'] = self.is_ducking
        dino_state['has_ducked_in_jump'] = self.has_ducked_in_jump
        
        return {
            'dino': dino_state,
            'obstacles': self.obstacles,
            'speed': self.current_speed,
            'score': self.get_score()
        }
    
    def close(self):
        """关闭游戏"""
        print("模拟游戏关闭")

# 主函数
def main():
    # 加载配置和运行模式
    config, run_mode = load_config()
    
    if not run_mode:
        print("程序退出")
        return
    
    # 初始化游戏
    print("使用Chrome浏览器模式")
    game = DinoGame(config)
    
    if run_mode == 'demo':
        # 展示模式 - 运行3次求平均
        ga = GeneticAlgorithm(config)
        ga.load_population()
        
        if ga.best_individual:
            print("\n🎯 使用历史最佳个体进行展示（3次运行求平均）...")
            scores = []
            
            for run in range(3):
                print(f"\n🎮 第 {run + 1} 次运行:")
                try:
                    game.restart()
                    
                    while not game.is_game_over():
                        game_state = game.get_game_state()
                        action = ga.best_individual.predict(game_state)
                        
                        if action['jump']:
                            game.jump()
                        
                        # 持续下蹲逻辑
                        if action['duck']:
                            game.start_duck()
                        else:
                            # 检查是否需要停止下蹲
                            obstacles = game_state.get('obstacles', [])
                            should_stop_duck = True
                            
                            # 如果还有高空翼龙在附近，继续下蹲
                            for obstacle in obstacles:
                                if obstacle.get('type') == 'PTERODACTYL_HIGH':
                                    distance = obstacle.get('x', 0) - (game_state['dino'].get('x', 0) + game_state['dino'].get('width', 40))
                                    if distance > -50 and distance < 150:  # 障碍物在附近
                                        should_stop_duck = False
                                        break
                            
                            if should_stop_duck:
                                game.stop_duck()
                        
                        time.sleep(game.delay)
                    
                    run_score = game.get_score()
                    scores.append(run_score)
                    emoji = get_score_emoji(run_score)
                    print(f"   第 {run + 1} 次得分: {run_score} {emoji}")
                    
                    # 如果不是最后一次运行，等待一下再开始下一次
                    if run < 2:
                        print("   准备下一次运行...")
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"   第 {run + 1} 次运行出错: {e}")
                    scores.append(0)
            
            # 计算并显示统计结果
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                print(f"\n📊 展示结果统计:")
                print(f"   🎯 平均得分: {avg_score:.1f} {get_score_emoji(avg_score)}")
                print(f"   🏆 最高得分: {max_score} {get_score_emoji(max_score)}")
                print(f"   📉 最低得分: {min_score} {get_score_emoji(min_score)}")
                print(f"   📈 得分范围: {max_score - min_score}")
            
            game.close()
        else:
            print("未找到历史最佳个体，请先进行训练")
            game.close()
        return
    
    # 训练模式
    # 初始化参数
    population_size = config["training"]["population_size"]
    generations = config["training"]["generations"]
    runs_per_individual = config["training"]["runs_per_individual"]
    
    ga = GeneticAlgorithm(config)
    
    # 检查是否有可用的检查点
    checkpoints = ga.list_checkpoints()
    if checkpoints:
        print("\n发现可用的训练检查点:")
        for i, cp in enumerate(checkpoints[:5]):  # 只显示最新的5个
            print(f"{i+1}. 第{cp['generation']}代 - 最佳适应度: {cp['best_fitness']:.2f} - 时间: {cp['timestamp']}")
        
        choice = input("\n是否从检查点恢复训练？(y/n，默认n): ").strip().lower()
        if choice == 'y':
            if ga.load_latest_checkpoint():
                print("成功从检查点恢复训练")
            else:
                print("检查点恢复失败，从普通保存文件加载")
                ga.load_population()
        else:
            # 尝试加载之前的种群
            ga.load_population()
    else:
        # 尝试加载之前的种群
        ga.load_population()
    
    # 训练统计信息
    training_stats = {
        'generation_times': [],
        'best_fitness_history': [],
        'avg_fitness_history': [],
        'improvement_count': 0
    }
    
    try:
        # 训练循环
        for generation in range(generations):
            generation_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"🚀 开始第 {ga.generation + 1} 代训练 (剩余 {generations - generation} 代)")
            print(f"{'='*60}")
            
            fitness_scores = []
            
            # 评估每个个体
            for i, individual in enumerate(ga.population):
                individual_start_time = time.time()
                individual_scores = []
                
                # 显示个体评估进度
                progress = (i + 1) / population_size * 100
                print(f"\n📊 评估个体 {i+1}/{population_size} ({progress:.1f}%)")
                
                # 每个个体运行多次，取平均分数
                for run in range(runs_per_individual):
                    run_progress = (run + 1) / runs_per_individual * 100
                    print(f"  🎮 运行 {run+1}/{runs_per_individual} ({run_progress:.1f}%)", end=" ")
                    
                    # 重启游戏
                    game.restart()
                    time.sleep(0.5)  # 等待游戏重启
                    
                    # 游戏循环
                    step_count = 0
                    max_steps = 10000  # 防止无限循环
                    
                    while not game.is_game_over() and step_count < max_steps:
                        try:
                            # 获取游戏状态
                            game_state = game.get_game_state()
                            
                            # 获取AI的决策
                            action = individual.predict(game_state)
                            
                            # 执行动作 - 支持跳跃中下蹲的快速落地和持续下蹲
                            if action['jump']:
                                game.jump()
                            
                            # 持续下蹲逻辑：开始下蹲后持续到障碍物通过
                            if action['duck']:
                                game.start_duck()  # 开始持续下蹲
                            else:
                                # 检查是否需要停止下蹲
                                game_state_current = game.get_game_state()
                                obstacles = game_state_current.get('obstacles', [])
                                should_stop_duck = True
                                
                                # 如果还有高空翼龙在附近，继续下蹲
                                for obstacle in obstacles:
                                    if obstacle.get('type') == 'PTERODACTYL_HIGH':
                                        distance = obstacle.get('x', 0) - (game_state_current['dino'].get('x', 0) + game_state_current['dino'].get('width', 40))
                                        if distance > -50 and distance < 150:  # 障碍物在附近
                                            should_stop_duck = False
                                            break
                                
                                if should_stop_duck:
                                    game.stop_duck()  # 停止下蹲
                            
                            # 短暂延迟，避免过度操作
                            time.sleep(game.delay)
                            step_count += 1
                            
                        except Exception as e:
                            print(f"游戏循环中出错: {e}")
                            break
                    
                    if step_count >= max_steps:
                        print(f"达到最大步数限制 {max_steps}，强制结束游戏")
                    
                    # 记录分数
                    score = game.get_score()
                    individual_scores.append(score)
                    print(f"得分: {score}")
                
                # 计算平均分数作为适应度
                avg_score = sum(individual_scores) / len(individual_scores)
                fitness_scores.append(avg_score)
                
                individual_time = time.time() - individual_start_time
                print(f"  ⭐ 个体 {i+1} 平均得分: {avg_score:.2f} (用时: {individual_time:.1f}s)")
            
            # 计算本代统计信息
            generation_time = time.time() - generation_start_time
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # 检查是否有改进
            improved = best_fitness > ga.best_fitness
            if improved:
                training_stats['improvement_count'] += 1
            
            # 记录统计信息
            training_stats['generation_times'].append(generation_time)
            training_stats['best_fitness_history'].append(best_fitness)
            training_stats['avg_fitness_history'].append(avg_fitness)
            
            # 记录到遗传算法的训练历史中
            generation_record = {
                'generation': ga.generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'generation_time': generation_time,
                'improved': improved,
                'fitness_distribution': {
                    'max': max(fitness_scores),
                    'min': min(fitness_scores),
                    'std': np.std(fitness_scores)
                }
            }
            ga.training_history.append(generation_record)
            
            # 进化到下一代
            ga.evolve(fitness_scores)
            
            # 显示详细的代结果
            print(f"\n{'='*60}")
            print(f"📈 第 {ga.generation} 代训练完成")
            print(f"{'='*60}")
            print(f"⏱️  训练时间: {generation_time:.2f} 秒")
            print(f"🏆 最佳适应度: {best_fitness:.2f} {'🆕' if improved else ''}")
            print(f"📊 平均适应度: {avg_fitness:.2f}")
            print(f"🎯 历史最佳: {ga.best_fitness:.2f}")
            
            # 显示适应度分布
            sorted_fitness = sorted(fitness_scores, reverse=True)
            print(f"📋 适应度分布: 前5名 {[f'{f:.1f}' for f in sorted_fitness[:5]]}")
            
            # 显示改进统计
            improvement_rate = training_stats['improvement_count'] / (ga.generation) * 100
            print(f"📈 改进率: {improvement_rate:.1f}% ({training_stats['improvement_count']}/{ga.generation} 代有改进)")
            
            # 预估剩余时间
            if len(training_stats['generation_times']) > 0:
                avg_gen_time = sum(training_stats['generation_times']) / len(training_stats['generation_times'])
                remaining_time = avg_gen_time * (generations - generation - 1)
                print(f"⏳ 预估剩余时间: {remaining_time/60:.1f} 分钟")
            
            # 保存种群
            ga.save_population()
            print(f"💾 种群已保存")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    finally:
        # 保存最终种群
        ga.save_population()
        
        # 生成训练统计报告
        ga.generate_training_report()
        
        # 使用最佳个体进行演示
        if ga.best_individual:
            print("\n使用历史最佳个体进行演示...")
            game.restart()
            
            while not game.is_game_over():
                game_state = game.get_game_state()
                action = ga.best_individual.predict(game_state)
                
                if action['jump']:
                    game.jump()
                
                # 持续下蹲逻辑
                if action['duck']:
                    game.start_duck()
                else:
                    # 检查是否需要停止下蹲
                    obstacles = game_state.get('obstacles', [])
                    should_stop_duck = True
                    
                    # 如果还有高空翼龙在附近，继续下蹲
                    for obstacle in obstacles:
                        if obstacle.get('type') == 'PTERODACTYL_HIGH':
                            distance = obstacle.get('x', 0) - (game_state['dino'].get('x', 0) + game_state['dino'].get('width', 40))
                            if distance > -50 and distance < 150:  # 障碍物在附近
                                should_stop_duck = False
                                break
                    
                    if should_stop_duck:
                        game.stop_duck()
                
                time.sleep(game.delay)
            
            final_score = game.get_score()
            print(f"演示结束，最终得分: {final_score}")
        
        # 关闭游戏
        game.close()

if __name__ == "__main__":
    main()