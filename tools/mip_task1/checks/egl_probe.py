"""EGL diagnosis for habitat_sim: why can't it create a windowless context, and which
environment makes it work? Stdlib + ctypes only; run with the MIP interpreter.

Writes the first working combination to $WORKDIR/egl.env (sourced by run_task1.sh).
Env: WORKDIR, RENDER_TIMEOUT (s, default 180). Exit 0 if some combination renders, 1 otherwise.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = "/usr/share/glvnd/egl_vendor.d"
NVIDIA_JSON = f"{VENDOR_DIR}/10_nvidia.json"
MESA_JSON = f"{VENDOR_DIR}/50_mesa.json"
EGL_EXTENSIONS = 0x3055
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_NO_DISPLAY = ctypes.c_void_p(0)

RENDER_SRC = """
import habitat_sim, sys
gpu = int(sys.argv[1])
backend = habitat_sim.SimulatorConfiguration()
backend.scene_id = "NONE"
backend.gpu_device_id = gpu
sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "rgb"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [64, 64]
agent = habitat_sim.agent.AgentConfiguration()
agent.sensor_specifications = [sensor]
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
obs = sim.get_sensor_observations()
print("RENDER_OK", getattr(obs["rgb"], "shape", None))
sim.close()
"""


def section(title: str) -> None:
    print(f"\n== {title}")


def host_facts() -> None:
    section("host")
    for var in (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__EGL_VENDOR_LIBRARY_DIRS",
        "EGL_PLATFORM",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",
    ):
        print(f"{var}={os.environ.get(var, '<unset>')}")
    print(
        "container:",
        "docker"
        if Path("/.dockerenv").exists()
        else (
            "cgroup:" + Path("/proc/1/cgroup").read_text().strip().splitlines()[-1][:60]
            if Path("/proc/1/cgroup").exists()
            else "?"
        ),
    )
    print("vendor jsons:", sorted(glob.glob(f"{VENDOR_DIR}/*.json")) or "none")
    for j in glob.glob(f"{VENDOR_DIR}/*.json"):
        print(f"  {j}: {Path(j).read_text().strip()}")
    for name in (
        "EGL",
        "EGL_nvidia",
        "EGL_mesa",
        "GLdispatch",
        "nvidia-eglcore",
        "cuda",
    ):
        print(f"find_library({name}):", ctypes.util.find_library(name))
    print(
        "nvidia egl libs:",
        sorted(
            glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia*")
            + glob.glob("/usr/lib*/libEGL_nvidia*")
            + glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-eglcore*")
        )[:6]
        or "none found in the usual dirs",
    )


def bundled_libegl() -> str | None:
    try:
        import habitat_sim  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    libs = Path(habitat_sim.__file__).resolve().parent.parent / "habitat_sim.libs"
    hits = sorted(glob.glob(str(libs / "libEGL*")))
    return hits[0] if hits else None


def compiled_vendor_paths(lib: str) -> list[str]:
    """Where a glvnd libEGL looks for vendor JSONs (compiled in): grep the binary for 'egl_vendor.d'."""
    try:
        data = Path(lib).read_bytes()
    except OSError:
        return []
    out = []
    idx = 0
    while True:
        idx = data.find(b"egl_vendor.d", idx)
        if idx < 0:
            break
        start = data.rfind(b"\x00", 0, idx) + 1
        end = data.find(b"\x00", idx)
        out.append(data[start:end].decode(errors="replace"))
        idx = end
    return sorted(set(out))


def egl_devices(libpath: str) -> None:
    """Load one libEGL, list client extensions and EGL devices, initialise a display per device."""
    section(f"EGL via {libpath}")
    try:
        egl = ctypes.CDLL(libpath)
    except OSError as ex:
        print("cannot load:", ex)
        return
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetError.restype = ctypes.c_int
    exts = egl.eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)
    exts_s = exts.decode() if exts else ""
    print(
        "client extensions:", exts_s or f"<none> (eglGetError=0x{egl.eglGetError():x})"
    )
    has_dev = "EGL_EXT_device_enumeration" in exts_s or "EGL_EXT_device_base" in exts_s
    print(
        "device enumeration supported:",
        has_dev,
        "| platform_device:",
        "EGL_EXT_platform_device" in exts_s,
    )
    egl.eglGetDisplay.restype = ctypes.c_void_p
    egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
    egl.eglInitialize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglTerminate.argtypes = [ctypes.c_void_p]
    major, minor = ctypes.c_int(), ctypes.c_int()
    dflt = egl.eglGetDisplay(None)
    if dflt:
        ok = egl.eglInitialize(dflt, ctypes.byref(major), ctypes.byref(minor))
        print(
            f"default display: init={'ok' if ok else 'FAIL'} vendor={egl.eglQueryString(dflt, EGL_VENDOR)} version={egl.eglQueryString(dflt, EGL_VERSION)}"
        )
        if ok:
            egl.eglTerminate(dflt)
    else:
        print(
            f"default display: EGL_NO_DISPLAY (eglGetError=0x{egl.eglGetError():x})  <- this is the error habitat reports"
        )
    if not has_dev:
        return
    q = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    gpd = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    qds = egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
    if not (q and gpd):
        print("eglQueryDevicesEXT / eglGetPlatformDisplayEXT not resolvable")
        return
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(q)
    getdpy = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p
    )(gpd)
    qdstr = (
        ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(qds)
        if qds
        else None
    )
    devs = (ctypes.c_void_p * 16)()
    n = ctypes.c_int(0)
    if not query(16, devs, ctypes.byref(n)):
        print(f"eglQueryDevicesEXT failed (0x{egl.eglGetError():x})")
        return
    print("EGL devices:", n.value)
    for i in range(n.value):
        dev_ext = qdstr(devs[i], EGL_EXTENSIONS) if qdstr else b""
        drm = (
            qdstr(devs[i], 0x3233)
            if qdstr and b"EGL_EXT_device_drm" in (dev_ext or b"")
            else None
        )
        dpy = getdpy(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
        if not dpy:
            print(f"  [{i}] no display (0x{egl.eglGetError():x}) exts={dev_ext}")
            continue
        ok = egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
        print(
            f"  [{i}] init={'ok' if ok else 'FAIL'} vendor={egl.eglQueryString(dpy, EGL_VENDOR)} version={egl.eglQueryString(dpy, EGL_VERSION)} drm={drm} exts={dev_ext}"
        )
        if ok:
            egl.eglTerminate(dpy)


def render(env_over: dict[str, str | None], gpu: int, timeout: int) -> tuple[bool, str]:
    env = dict(os.environ)
    env.update({"MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet"})
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    try:
        p = subprocess.run(
            [sys.executable, "-c", RENDER_SRC, str(gpu)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = (p.stdout + p.stderr).strip()
    if p.returncode == 0 and "RENDER_OK" in out:
        return True, out.split("RENDER_OK", 1)[1].strip()
    lines = [line for line in out.splitlines() if line.strip()]
    return False, " | ".join(lines[-2:])[:300]


def main() -> int:
    workdir = Path(os.environ.get("WORKDIR", ".")).expanduser()
    timeout = int(os.environ.get("RENDER_TIMEOUT", "180"))
    host_facts()
    bundled = bundled_libegl()
    section("libEGL variants")
    sys_egl = ctypes.util.find_library("EGL")
    print("system libEGL:", sys_egl)
    print("habitat-bundled libEGL:", bundled)
    if bundled:
        print(
            "  compiled-in vendor search paths:",
            compiled_vendor_paths(bundled) or "<none found in binary>",
        )
    if sys_egl:
        egl_devices(sys_egl)
    if bundled:
        egl_devices(bundled)

    section("habitat_sim render attempts (empty scene, 64x64)")
    nvidia = NVIDIA_JSON if Path(NVIDIA_JSON).exists() else None
    mesa = MESA_JSON if Path(MESA_JSON).exists() else None
    variants: list[tuple[str, dict[str, str | None]]] = [("as-is", {})]
    if nvidia:
        variants.append((
            "nvidia vendor json",
            {"__EGL_VENDOR_LIBRARY_FILENAMES": nvidia},
        ))
        variants.append((
            "nvidia vendor json, DISPLAY unset",
            {"__EGL_VENDOR_LIBRARY_FILENAMES": nvidia, "DISPLAY": None},
        ))
    variants.append((
        "vendor dir, DISPLAY unset",
        {"__EGL_VENDOR_LIBRARY_DIRS": VENDOR_DIR, "DISPLAY": None},
    ))
    variants.append(("DISPLAY unset", {"DISPLAY": None}))
    if sys_egl:
        # bypass the wheel's bundled glvnd: preload the system libEGL (and its dispatch)
        preload = ":".join(
            p for p in (sys_egl, ctypes.util.find_library("GLdispatch")) if p
        )
        variants.append((
            "system libEGL preloaded",
            {"LD_PRELOAD": preload, "DISPLAY": None},
        ))
        if nvidia:
            variants.append((
                "system libEGL preloaded + nvidia json",
                {
                    "LD_PRELOAD": preload,
                    "__EGL_VENDOR_LIBRARY_FILENAMES": nvidia,
                    "DISPLAY": None,
                },
            ))
    if mesa:
        variants.append((
            "mesa surfaceless (software, slow)",
            {
                "__EGL_VENDOR_LIBRARY_FILENAMES": mesa,
                "EGL_PLATFORM": "surfaceless",
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "DISPLAY": None,
            },
        ))
    winner: tuple[str, dict[str, str | None], int] | None = None
    for name, over in variants:
        for gpu in (0, 1, 2, 3):
            ok, msg = render(over, gpu, timeout)
            print(
                f"  {name:38s} gpu_device_id={gpu}: {'OK ' + msg if ok else 'FAIL ' + msg}"
            )
            if ok:
                winner = (name, over, gpu)
                break
            if gpu == 0 and "EGL" not in msg and "context" not in msg.lower():
                break  # not an EGL problem: other gpu ids will not help
        if winner:
            break
    print()
    if not winner:
        print("RESULT: no combination rendered. Send this whole log back.")
        return 1
    name, over, gpu = winner
    lines = [
        f"# written by egl_probe.py — first working combination: {name}, gpu_device_id={gpu}"
    ]
    for k, v in over.items():
        lines.append(f"unset {k}" if v is None else f"export {k}={v}")
    lines.append(f"export MIP_GPU_DEVICE_ID={gpu}")
    path = workdir / "egl.env"
    path.write_text("\n".join(lines) + "\n")
    print(
        f"RESULT: '{name}' renders (gpu_device_id={gpu}). Wrote {path}; run_task1.sh sources it automatically."
    )
    if gpu != 0:
        print(
            "  note: gpu_device_id != 0 — MIP's env config uses the default device; set CUDA_VISIBLE_DEVICES / EGL device accordingly (see report)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
