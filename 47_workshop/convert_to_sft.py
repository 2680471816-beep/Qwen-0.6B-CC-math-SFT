#!/usr/bin/env python3
"""将 problem.jsonl 文件转换为 SFT 训练格式"""

import json
import glob
import os
import re

def fix_json_line(line: str) -> str:
    """修复 source_index 没有引号的问题"""
    # 修复 "source_index": xxx} 或 "source_index": xxx, 格式
    return re.sub(
        r'"source_index":\s*([a-f0-9-]+)([},])',
        r'"source_index": "\1"\2',
        line
    )

def convert_file(input_path: str) -> None:
    """转换单个文件"""
    # 生成输出文件名
    base = os.path.basename(input_path)
    output_path = input_path.replace("_problem.jsonl", "_sft.jsonl")

    converted = 0
    skipped = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # 尝试修复 source_index 没有引号的问题
                try:
                    fixed_line = fix_json_line(line)
                    data = json.loads(fixed_line)
                except:
                    skipped += 1
                    continue

            question = data.get('question', '')
            solution = data.get('solution', '')
            answer = data.get('answer', '')

            if not question:
                skipped += 1
                continue

            # 构建新格式
            new_data = {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"<think>\n{solution}\n</think>\n\\boxed{{{answer}}}"}
                ]
            }

            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            converted += 1

    print(f"{base}: 转换 {converted} 条, 跳过 {skipped} 条 -> {os.path.basename(output_path)}")

def main():
    work_dir = "/home/ubuntu/Midterm_Project/47_workshop"
    pattern = os.path.join(work_dir, "*problem.jsonl")
    files = glob.glob(pattern)

    print(f"找到 {len(files)} 个文件待处理\n")

    for f in sorted(files):
        # 跳过空文件
        if os.path.getsize(f) == 0:
            print(f"{os.path.basename(f)}: 空文件，跳过")
            continue
        convert_file(f)

    print("\n转换完成!")

if __name__ == "__main__":
    main()
