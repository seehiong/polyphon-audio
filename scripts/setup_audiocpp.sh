#!/usr/bin/env bash
# Clones, configures, and builds audio.cpp (AMD/ROCm/Ubuntu workaround
# flags - see README "Quickstart" step 1; on NVIDIA/Apple Silicon/CPU,
# follow audio.cpp's own README instead), then installs the five models
# this project's server.json expects. Safe to re-run: skips the clone if
# ~/github/audio.cpp already exists, and cmake/model_manager_v2.py are both
# already idempotent on their own.
#
# audio.cpp is cloned to ~/github/audio.cpp (kept alongside this project
# rather than inside it, same as llama.cpp), with vendor/audio.cpp as a
# symlink to it so the rest of this repo's paths keep working. This
# project's own server.json (tracked at the repo root) is symlinked into
# vendor/audio.cpp too, so its "path" entries resolve against audio.cpp's
# models/ directory.
#
# Usage: scripts/setup_audiocpp.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audiocpp_dir="$HOME/github/audio.cpp"
vendor_dir="$repo_root/vendor/audio.cpp"

if [ -d "$audiocpp_dir" ]; then
    echo "$audiocpp_dir already exists, skipping clone." >&2
else
    echo "Cloning audio.cpp..." >&2
    git clone https://github.com/0xShug0/audio.cpp.git "$audiocpp_dir"
fi

if [ ! -e "$vendor_dir" ]; then
    mkdir -p "$repo_root/vendor"
    ln -s "$audiocpp_dir" "$vendor_dir"
fi

if [ ! -e "$vendor_dir/server.json" ]; then
    ln -s "$repo_root/server.json" "$vendor_dir/server.json"
fi

cd "$vendor_dir"

echo "Configuring (AMD/ROCm/gfx1151 flags)..." >&2
cmake -S . -B build_hip \
    -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_CUDA_GRAPHS=OFF -DGPU_TARGETS=gfx1151 \
    -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" \
    -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom \
    -DAUDIOCPP_MODELS="chatterbox,qwen3_asr,nemotron_asr,personaplex,qwen3_tts,stable_audio" \
    -DOpenMP_C_FLAGS="-fopenmp=libomp -I/usr/lib/llvm-20/lib/clang/20/include" -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp -I/usr/lib/llvm-20/lib/clang/20/include" -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/usr/lib/llvm-20/lib/libomp.so \
    -DAMDDeviceLibs_DIR=/usr/lib/x86_64-linux-gnu/cmake/AMDDeviceLibs

echo "Building (audiocpp_cli, audiocpp_server)..." >&2
cmake --build build_hip -j"$(nproc)" --target audiocpp_cli --target audiocpp_server

echo "Installing models..." >&2
python3 tools/model_manager_v2.py install chatterbox_q8_0                  # ~2.1GB - voice cloning
python3 tools/model_manager_v2.py install nemotron_asr_q8_0                # ~0.9GB - speech-to-text
python3 tools/model_manager_v2.py install qwen3_tts_1_7b_voicedesign_q8_0  # ~2.8GB - optional, Voice Library tab only
python3 tools/model_manager_v2.py install stable_audio_3_small_music_q8_0  # ~1.6GB - optional, Sound & Music tab only
python3 tools/model_manager_v2.py install stable_audio_3_small_sfx_q8_0    # ~1.6GB - optional, Sound & Music tab only

echo "Done. Next: configure and start audiocpp_server - see README 'Quickstart' step 2." >&2
