
import os


def add_to_user_shell_config(variable_name, variable_value):
    """
    添加/覆盖用户的shell配置文件中的环境变量
    :param variable_name: 环境变量名（如PATH、MY_APP_PATH）
    :param variable_value: 环境变量新值
    """
    # 确定用户的 shell 配置文件
    shell = os.environ.get('SHELL', '/bin/bash')
    config_file = ''
    
    if 'bash' in shell:
        config_file = os.path.expanduser('~/.bashrc')
    elif 'zsh' in shell:
        config_file = os.path.expanduser('~/.zshrc')
    else:
        config_file = os.path.expanduser('~/.profile')
    
    # 定义标准的export行（统一格式，避免空格/引号问题）
    new_export_line = f'export {variable_name}="{variable_value}"\n'
    # 匹配变量名的正则式（简化为字符串匹配，适配新手）
    var_pattern = f'export {variable_name}='

    try:
        # 读取配置文件全部内容
        with open(config_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 按行读取，便于逐行修改
        
        # 标记是否找到并修改了原有变量
        updated = False
        # 存储修改后的所有行
        new_lines = []
        
        for line in lines:
            # 检查当前行是否是该变量的export语句
            if line.strip().startswith(var_pattern):
                # 找到原有变量行，替换为新值
                new_lines.append(new_export_line)
                updated = True
                print(f"找到原有{variable_name}变量，已替换为新值")
            else:
                # 非目标变量行，保留原样
                new_lines.append(line)
        
        # 如果未找到原有变量，追加新行到末尾
        if not updated:
            new_lines.append(new_export_line)
            print(f"未找到{variable_name}变量，已追加到{config_file}")
        
        # 将修改后的内容写回文件（覆盖原文件）
        with open(config_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"操作完成：{config_file}已更新")

    except FileNotFoundError:
        # 配置文件不存在时，创建并写入
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_export_line)
        print(f"配置文件不存在，创建{config_file}并添加{variable_name}变量")
