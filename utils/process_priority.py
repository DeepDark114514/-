#  南京信息工程大学22级信安1班 202283290014
# 2026.5.13
# Windows 进程优先级设置
# 解决 CMD 窗口失去焦点后训练降速的问题

import os
import sys


def set_high_priority():
    # 将当前进程优先级设为 HIGH（仅 Windows）。
    # 防止 CMD/Terminal 窗口失去焦点后被系统降速。
    if sys.platform != 'win32':
        return

    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("[Priority] Process priority set to HIGH")
    except Exception as e:
        print(f"[Priority] Failed to set priority: {e}")


def disable_quick_edit_tip():
    # 打印提示：建议关闭 CMD 快速编辑模式，这是 Windows 控制台降速的主因
    if sys.platform != 'win32':
        return
    print("[Priority] Tip: 如果切窗口后训练仍变慢，请在 CMD 标题栏右键 -> 属性 -> 取消勾选'快速编辑模式'")
