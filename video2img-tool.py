from pathlib import Path
import cv2
import numpy as np
import os
import sys
import time
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "進捗表示のために tqdm をインストールしてください: pip install tqdm"
    )

load_dotenv(override=True)  # .env ファイルの環境変数を読み込む

# OS別の非ブロッキングキー入力ヘルパー
if os.name == "nt":
    import msvcrt

    class KeyPoller:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def poll(self):
            if msvcrt.kbhit():
                try:
                    return msvcrt.getwch().lower()
                except Exception:
                    return None
            return None

else:
    import select
    import termios
    import tty

    class KeyPoller:
        def __enter__(self):
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

        def poll(self):
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                return sys.stdin.read(1).lower()
            return None


def wait_while_paused(poller, video_pbar, frame_pbar=None):
    """一時停止中の待機ループ"""
    tqdm.write("\n[一時停止中] 'p' キーで再開 / 'q' キーで安全に終了します...")
    while True:
        time.sleep(0.1)
        key = poller.poll()
        if key == "p":
            tqdm.write("[再開] 処理を再開します。")
            video_pbar.refresh()
            if frame_pbar is not None:
                frame_pbar.refresh()
            return False  # 継続
        elif key == "q":
            return True  # 終了要求


def extract_last_10sec_to_300frames(
    video_dir, output_dir, target_sec=6, num_frames=300
):
    if not video_dir:
        raise ValueError("環境変数 VIDEO_DIR が設定されていません。")

    input_root = Path(video_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    target_exts = {".mp4", ".mov", ".avi", ".mkv"}

    video_paths = sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in target_exts
    )

    print(f"対象動画数: {len(video_paths)}本")
    print(
        "操作方法: [P] 一時停止 / 再開  |  [Q] または [Ctrl+C] 安全に中断（次回続きから再開可）\n"
    )

    video_pbar = tqdm(video_paths, desc="全体進捗", unit="本")
    stop_requested = False

    with KeyPoller() as poller:
        try:
            for video_path in video_pbar:
                # 動画切り替え時のキー入力チェック
                key = poller.poll()
                if key == "p":
                    stop_requested = wait_while_paused(poller, video_pbar)
                elif key == "q":
                    stop_requested = True

                if stop_requested:
                    tqdm.write("\n[中断] ユーザー操作により処理を安全に停止しました。")
                    break

                rel_display = video_path.relative_to(input_root)
                video_pbar.set_postfix_str(str(rel_display))

                # 保存先パスの決定
                rel_parent = video_path.parent.relative_to(input_root)
                base_name = video_path.stem
                save_dir = output_root / rel_parent / base_name

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    tqdm.write(f"[スキップ] 開けませんでした: {rel_display}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames <= 0:
                    cap.release()
                    tqdm.write(f"[スキップ] フレーム数が0です: {rel_display}")
                    continue

                start_frame = max(0, total_frames - int(round(fps * target_sec)))
                available_frames = total_frames - start_frame
                extract_count = min(num_frames, available_frames)

                # レジューム機能：すでに既定枚数が保存済みの動画はスキップ
                if save_dir.exists():
                    existing_imgs = list(save_dir.glob("*.jpg"))
                    if len(existing_imgs) >= extract_count:
                        cap.release()
                        continue

                save_dir.mkdir(parents=True, exist_ok=True)
                target_indices = set(
                    np.linspace(start_frame, total_frames - 1, extract_count, dtype=int)
                )

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
                saved_idx = 0

                with tqdm(
                    total=extract_count,
                    desc=f"  └ {base_name[:20]}",
                    unit="枚",
                    leave=False,
                ) as frame_pbar:
                    while current_frame < total_frames:
                        # フレーム処理中のキー入力チェック
                        key = poller.poll()
                        if key == "p":
                            stop_requested = wait_while_paused(
                                poller, video_pbar, frame_pbar
                            )
                        elif key == "q":
                            stop_requested = True

                        if stop_requested:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        if current_frame in target_indices:
                            out_path = save_dir / f"{base_name}_{saved_idx:03d}.jpg"
                            success, buf = cv2.imencode(".jpg", frame)
                            if success:
                                out_path.write_bytes(buf.tobytes())
                                saved_idx += 1
                                frame_pbar.update(1)

                        current_frame += 1

                cap.release()

                if stop_requested:
                    tqdm.write(
                        f"\n[中断] '{rel_display}' の途中で停止しました（次回実行時にこの動画から再開します）。"
                    )
                    break

        except KeyboardInterrupt:
            tqdm.write(
                "\n[中断] Ctrl+C を検知しました。安全に終了します（次回続きから再開できます）。"
            )

    if not stop_requested:
        print("すべての処理が完了しました。")


if __name__ == "__main__":
    INPUT_VIDEO_DIR = os.getenv("VIDEO_DIR")
    OUTPUT_IMAGE_DIR = os.getenv("OUTPUT_DIR")

    extract_last_10sec_to_300frames(
        video_dir=INPUT_VIDEO_DIR,
        output_dir=OUTPUT_IMAGE_DIR,
        target_sec=6,
        num_frames=300,
    )
