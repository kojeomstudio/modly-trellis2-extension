"""
Build the vendor/ directory for the TRELLIS.2 extension.

Run this script once (with the app's venv active) to populate vendor/.
The resulting vendor/ folder is committed to the extension repository
so end users never need to install anything at runtime.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)
    - PyTorch + CUDA (must be available at inference time anyway)
    - MSVC on Windows / gcc on Linux (for compiling CUDA extensions)
"""

import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

VENDOR       = Path(__file__).parent / "vendor"
TRELLIS2_ZIP = "https://github.com/microsoft/TRELLIS.2/archive/refs/heads/main.zip"

# Pure-Python packages to vendor (no compilation needed)
PURE_PACKAGES = [
    "easydict",       # configuration dict used internally by trellis2
    "plyfile",        # PLY mesh format I/O
    "einops",         # tensor reshaping helpers
    "utils3d",        # 3D math utilities
    "lpips",          # perceptual loss metric
    "trimesh",        # mesh processing
    "tqdm",           # progress bars
    # opencv-python and spconv are too large to vendor in git — installed at runtime via pip
]

# Compiled CUDA extensions to vendor (require --no-build-isolation to find torch)
# Note: flex_gemm is not on PyPI — spconv is used instead (set via SPARSE_CONV_BACKEND env var)
# Note: spconv and nvdiffrast use custom sources (see main())
COMPILED_PACKAGES = [
    "cumesh",         # CUDA mesh utilities
]

# Packages not on PyPI — installed from GitHub
GITHUB_PACKAGES = [
    "git+https://github.com/NVlabs/nvdiffrast",   # NVlabs differentiable rasterizer
]

# spconv fallback versions (newest to oldest) — tried in order until one works
SPCONV_FALLBACK_VERSIONS = ["cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target."""
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_compiled_package(package: str, dest: Path) -> None:
    """Install a compiled package into vendor/ via pip --target --no-build-isolation.

    --no-build-isolation lets the build process find torch in the current
    environment, which is required by CUDA extensions that depend on PyTorch.
    CUDAFLAGS is set to allow unsupported MSVC versions (e.g. VS 2025).
    """
    import os
    env = os.environ.copy()
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["CMAKE_CUDA_FLAGS"] = "-allow-unsupported-compiler"
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--no-build-isolation",
         "--target", str(dest),
         "--upgrade",
         package], env=env)
    print(f"  Vendored {package}.")


def build_nvdiffrast(dest: Path) -> None:
    """Clone nvdiffrast, patch setup.py to allow unsupported MSVC, build and extract to vendor/."""
    pkg_dest = dest / "nvdiffrast"
    if pkg_dest.exists() and any(pkg_dest.iterdir()):
        print("  nvdiffrast already present, skipping.")
        return

    _NVCC_PATCH = (
        "# --- injected by build_vendor.py ---\n"
        "try:\n"
        "    import torch.utils.cpp_extension as _ext\n"
        "    _orig_CUDA = _ext.CUDAExtension\n"
        "    def _patched_CUDA(*_a, **_kw):\n"
        "        eca = _kw.setdefault('extra_compile_args', {})\n"
        "        if isinstance(eca, dict):\n"
        "            eca.setdefault('nvcc', []).append('-allow-unsupported-compiler')\n"
        "        elif isinstance(eca, list):\n"
        "            eca.append('-allow-unsupported-compiler')\n"
        "        return _orig_CUDA(*_a, **_kw)\n"
        "    _ext.CUDAExtension = _patched_CUDA\n"
        "except Exception:\n"
        "    pass\n"
        "# --- end injection ---\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run(["git", "clone", "--depth=1",
             "https://github.com/NVlabs/nvdiffrast.git",
             str(tmp / "nvdiffrast_src")])

        src = tmp / "nvdiffrast_src"
        setup_py = src / "setup.py"
        if setup_py.exists():
            original = setup_py.read_text(encoding="utf-8")
            setup_py.write_text(_NVCC_PATCH + original, encoding="utf-8")
            print("  Patched setup.py with -allow-unsupported-compiler.")

        wheel_dir = tmp / "wheels"
        wheel_dir.mkdir()

        build_env = os.environ.copy()
        build_env["CUDAFLAGS"]        = "-allow-unsupported-compiler"
        build_env["CMAKE_CUDA_FLAGS"] = "-allow-unsupported-compiler"

        run([sys.executable, "-m", "pip", "wheel",
             "--no-deps", "--no-build-isolation",
             "-w", str(wheel_dir), "."],
            cwd=src, env=build_env)

        wheels = list(wheel_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError("pip wheel produced no output for nvdiffrast.")

        pkg_dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheels[0]) as zf:
            for member in zf.namelist():
                if ".dist-info" in member:
                    continue
                if member.startswith("nvdiffrast/"):
                    # Python package files → vendor/nvdiffrast/
                    rel    = member[len("nvdiffrast/"):]
                    target = pkg_dest / rel
                elif "/" not in member and (member.endswith(".pyd") or member.endswith(".so")):
                    # Root-level compiled extension (e.g. _nvdiffrast_c.cp311-win_amd64.pyd)
                    # → vendor/nvdiffrast/ so it's next to the Python package
                    target = pkg_dest / member
                else:
                    continue
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
                    print(f"  Extracted {member} -> vendor/nvdiffrast/")


def vendor_trellis2(dest: Path) -> None:
    """Download TRELLIS.2 source and extract only the trellis2/ package into vendor/."""
    import urllib.request

    trellis2_dest = dest / "trellis2"
    if trellis2_dest.exists():
        print("  trellis2/ already present, skipping.")
        return

    print("  Downloading TRELLIS.2 source from GitHub...")
    with urllib.request.urlopen(TRELLIS2_ZIP, timeout=180) as resp:
        data = resp.read()

    # The ZIP root folder is "TRELLIS.2-main/" (GitHub archive naming)
    prefix = "TRELLIS.2-main/trellis2/"
    strip  = "TRELLIS.2-main/"

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError(
            f"No files were extracted from the ZIP. "
            f"The expected prefix '{prefix}' was not found.\n"
            "Check that the GitHub archive structure matches and update the "
            "'prefix' variable in vendor_trellis2() if needed."
        )

    print(f"  trellis2/ extracted to {dest} ({extracted} files).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Guard: torch must be importable — ensures we're in the right venv.
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torch is not importable from this Python environment.\n"
            "Run build_vendor.py using the app's venv Python (the one with PyTorch),\n"
            f"not the system Python.\nCurrent interpreter: {sys.executable}"
        )

    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    print("\n[1] Vendoring pure-Python packages...")
    for pkg in PURE_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_pure_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            print("  Skipping — it may already be available in the venv.")

    # 2. TRELLIS.2 source
    print("\n[2] Vendoring trellis2 source...")
    vendor_trellis2(VENDOR)

    # 3. Compiled CUDA extensions
    print("\n[3] Vendoring compiled CUDA extensions...")
    import torch

    failed = []

    # Standard compiled packages
    for pkg in COMPILED_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            failed.append(pkg)

    # nvdiffrast — cloned and patched to allow unsupported MSVC
    print("\n  -> nvdiffrast (from GitHub, patched)")
    try:
        build_nvdiffrast(VENDOR)
    except Exception as exc:
        print(f"  WARNING: failed to build nvdiffrast: {exc}")
        failed.append("nvdiffrast")

    # spconv — try versions from newest to oldest until one works
    cuda_ver = torch.version.cuda  # e.g. "12.8"
    cuda_tag = "cu" + cuda_ver.replace(".", "")
    versions_to_try = [cuda_tag] + [v for v in SPCONV_FALLBACK_VERSIONS if v != cuda_tag]
    spconv_ok = False
    for ver in versions_to_try:
        pkg = f"spconv-{ver}"
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
            spconv_ok = True
            break
        except Exception:
            print(f"  Not available, trying next version...")
    if not spconv_ok:
        print("  WARNING: could not vendor any spconv version.")
        failed.append("spconv")

    if failed:
        print(f"\n  The following packages could not be vendored: {failed}")
        print("  Generation may not work without them.")

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("End users will never need to install anything.")


if __name__ == "__main__":
    main()
