#!/bin/bash
set -euo pipefail

# Build a TinyNav map on the Jetson devcontainer from a copied rosbag.
# Usage:
#   bash scripts/run_jetson_rosbag_build_map.sh bag_2025_08_27_00_57_16
#   bash scripts/run_jetson_rosbag_build_map.sh /tinynav/tinynav_db/rosbags/bag_2025_08_27_00_57_16

tinynav_root="${TINYNAV_ROOT:-/tinynav}"
tinynav_db_path="${TINYNAV_DB_PATH:-${tinynav_root}/tinynav_db}"
ros_domain_id="${ROS_DOMAIN_ID:-231}"
session_name=""
dry_run=0

usage() {
    cat <<EOF
Usage: $0 [--domain ROS_DOMAIN_ID] [--session NAME] [--dry-run] BAG_NAME_OR_PATH

Build a map on Jetson from a rosbag folder using the backend-equivalent looper flow.

Arguments:
  BAG_NAME_OR_PATH       Bag folder name under ${tinynav_db_path}/rosbags, a bag folder path,
                         or a direct bag_0.db3 path.

Options:
  --domain ID            ROS_DOMAIN_ID for isolated map building. Default: ${ros_domain_id}
  --session NAME         tmux session name. Default: build_map_<bag_name>
  --dry-run              Validate paths and print the resolved build settings without starting tmux.
  -h, --help             Show this help.

Examples:
  $0 bag_2025_08_27_00_57_16
  $0 /tinynav/tinynav_db/rosbags/bag_2025_08_27_00_57_16
EOF
}

bag_arg=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --domain" >&2
                exit 2
            fi
            ros_domain_id="$2"
            shift 2
            ;;
        --session)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --session" >&2
                exit 2
            fi
            session_name="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "${bag_arg}" ]]; then
                echo "Only one bag argument is allowed" >&2
                usage >&2
                exit 2
            fi
            bag_arg="$1"
            shift
            ;;
    esac
done

if [[ -z "${bag_arg}" ]]; then
    usage >&2
    exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required" >&2
    exit 3
fi

if [[ ! -d "${tinynav_root}" ]]; then
    echo "TinyNav root not found: ${tinynav_root}" >&2
    echo "Run this inside the Jetson devcontainer, or set TINYNAV_ROOT." >&2
    exit 3
fi

if [[ "${bag_arg}" == *.db3 ]]; then
    bag_file="${bag_arg}"
    bag_dir="$(dirname "${bag_file}")"
else
    if [[ "${bag_arg}" = /* || "${bag_arg}" == ./* || "${bag_arg}" == ../* || "${bag_arg}" == *"/"* ]]; then
        bag_dir="${bag_arg}"
    else
        bag_dir="${tinynav_db_path}/rosbags/${bag_arg}"
    fi
    bag_file="${bag_dir}/bag_0.db3"
fi

if [[ ! -d "${bag_dir}" ]]; then
    echo "Bag directory not found: ${bag_dir}" >&2
    exit 4
fi

if [[ ! -f "${bag_file}" ]]; then
    echo "bag_0.db3 not found: ${bag_file}" >&2
    exit 5
fi

vocab_path="${tinynav_root}/docs/Vocabulary/ORBvoc.txt"
if [[ ! -f "${vocab_path}" ]]; then
    echo "DBoW3 vocabulary not found: ${vocab_path}" >&2
    exit 6
fi

if ! python3 - <<'PY'
try:
    import pydbow3  # noqa: F401
except Exception:
    try:
        import pyDBoW3  # noqa: F401
    except Exception:
        raise
PY
then
    echo "pydbow3/pyDBoW3 is not importable in this Python environment" >&2
    echo "Install the DBoW3 Python binding before running backend-equivalent BOW map build." >&2
    exit 7
fi

map_path="${tinynav_db_path}/map"
maps_dir="${tinynav_db_path}/maps"
logs_dir="${tinynav_db_path}/logs"

bag_name="$(basename "${bag_dir}")"
safe_bag_name="$(printf '%s' "${bag_name}" | tr -c 'A-Za-z0-9_-' '_')"
if [[ -z "${session_name}" ]]; then
    session_name="build_map_${safe_bag_name}"
fi

if [[ "${dry_run}" == "1" ]]; then
    cat <<EOF
Jetson map build dry run passed.
  session: ${session_name}
  bag: ${bag_file}
  ROS_DOMAIN_ID: ${ros_domain_id}
  map output: ${map_path}
  maps dir: ${maps_dir}
  vocabulary: ${vocab_path}
EOF
    exit 0
fi

mkdir -p "${maps_dir}" "${logs_dir}"
build_log="${logs_dir}/build_map_${safe_bag_name}_$(date +%Y_%m_%d_%H_%M_%S).log"
bridge_log="${logs_dir}/looper_bridge_${safe_bag_name}_$(date +%Y_%m_%d_%H_%M_%S).log"

runner_script="$(mktemp "/tmp/tinynav_build_map_${safe_bag_name}.XXXXXX.sh")"
cat > "${runner_script}" <<EOF
#!/bin/bash
set -euo pipefail
cd "${tinynav_root}"

exec > >(tee -a "${build_log}") 2>&1

export TINYNAV_DB_PATH="${tinynav_db_path}"
export ROS_DOMAIN_ID="${ros_domain_id}"
export TMPDIR="\${TMPDIR:-/tmp}"
export TEMP="\${TEMP:-\${TMPDIR}}"
export TMP="\${TMP:-\${TMPDIR}}"

cleanup_bridge() {
    tmux send-keys -t "${session_name}:0.0" C-c >/dev/null 2>&1 || true
}
trap cleanup_bridge EXIT

map_path="${map_path}"
maps_dir="${maps_dir}"

echo "LOG_FILE:${build_log}"
echo "ROS_DOMAIN_ID:\${ROS_DOMAIN_ID}"
echo "Building map from ${bag_file}"
echo "Temporary output: \${map_path}"

if [[ -L "\${map_path}" || -f "\${map_path}" ]]; then
    rm -f "\${map_path}"
elif [[ -d "\${map_path}" ]]; then
    rm -rf "\${map_path}"
fi

uv run python "${tinynav_root}/tinynav/core/build_map_node.py" \\
    --map_save_path "\${map_path}" \\
    --bag_file "${bag_file}" \\
    --loop-closure-mode bow \\
    --loop-closure-use-bow \\
    --dbow3-vocabulary-path "${vocab_path}"

if [[ ! -d "\${map_path}" ]]; then
    echo "map output not found: \${map_path}" >&2
    exit 7
fi

ts="\$(date +map_%Y_%m_%d_%H_%M_%S)"
dest="\${maps_dir}/\${ts}"
if [[ -e "\${dest}" ]]; then
    echo "destination already exists: \${dest}" >&2
    exit 8
fi

mv "\${map_path}" "\${dest}"
ln -sfn "maps/\${ts}" "\${map_path}"

pois_path="\${dest}/pois.json"
if [[ ! -f "\${pois_path}" ]]; then
    cat > "\${pois_path}" <<'JSON'
{
  "0": {
    "id": 0,
    "name": "home",
    "position": [
      0.0,
      0.0,
      0.0
    ]
  }
}
JSON
    echo "Auto-created home POI at (0,0,0)"
fi

echo "MAP_BUILD_DONE:\${dest}"
echo "ACTIVE_MAP:\$(readlink -f "\${map_path}")"
EOF
chmod +x "${runner_script}"

tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
tmux new-session -d -s "${session_name}" \
    "cd '${tinynav_root}' && ROS_DOMAIN_ID='${ros_domain_id}' TINYNAV_DB_PATH='${tinynav_db_path}' uv run python '${tinynav_root}/tool/looper_bridge_node.py' 2>&1 | tee -a '${bridge_log}'"
tmux split-window -t "${session_name}:0" -h "${runner_script}"
tmux select-pane -t "${session_name}:0.1"

cat <<EOF
Started Jetson map build.
  session: ${session_name}
  bag: ${bag_file}
  ROS_DOMAIN_ID: ${ros_domain_id}
  build log: ${build_log}
  bridge log: ${bridge_log}

Monitor:
  tmux attach -t ${session_name}
  tail -f ${build_log}
EOF
