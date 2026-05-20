import os
import sys


def set_high_priority():
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
    if sys.platform != 'win32':
        return
    print("[Priority] Tip: 如果切窗口后训练变慢，关闭CMD快速编辑模式")
