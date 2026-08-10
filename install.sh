#!/usr/bin/env bash
#
# One-shot installer for AlphaPulldownSnakemake.
#
# Creates the conda environment, deploys the workflow into a working directory,
# and pre-fetches the Singularity/Apptainer containers into a *shared* image
# directory so that subsequent deployments never re-download them.
#
#   ./install.sh                       # deploy into ./AlphaPulldownSnakemake
#   ./install.sh -d my_project         # deploy into ./my_project
#   ./install.sh -v 2.5.1              # pin a workflow version
#   ./install.sh -i /g/shared/images   # shared image directory
#   ./install.sh --no-pull             # skip container pre-fetch
#
set -euo pipefail

VERSION="2.5.1"
DEST="AlphaPulldownSnakemake"
ENV_NAME="snake"
IMAGE_DIR_DEFAULT="$HOME/.apptainer/snakemake-images"
IMAGE_DIR="$IMAGE_DIR_DEFAULT"
DO_PULL=1
REPO="https://github.com/KosinskiLab/AlphaPulldownSnakemake"

usage() {
    sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--version)   VERSION="$2"; shift 2 ;;
        -d|--dest)      DEST="$2"; shift 2 ;;
        -i|--image-dir) IMAGE_DIR="$2"; shift 2 ;;
        -n|--env-name)  ENV_NAME="$2"; shift 2 ;;
        --no-pull)      DO_PULL=0; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==> WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m==> ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. Prerequisites
# --------------------------------------------------------------------------
command -v conda >/dev/null 2>&1 || die "conda not found in PATH. Install Miniforge/Miniconda first."

CONTAINER_CMD=""
for candidate in apptainer singularity; do
    if command -v "$candidate" >/dev/null 2>&1; then CONTAINER_CMD="$candidate"; break; fi
done
if [ -z "$CONTAINER_CMD" ] && [ "$DO_PULL" -eq 1 ]; then
    warn "Neither apptainer nor singularity found; skipping container pre-fetch."
    DO_PULL=0
fi

# --------------------------------------------------------------------------
# 2. Conda environment
# --------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' already exists; leaving it untouched."
else
    log "Creating conda environment '$ENV_NAME' (workflow $VERSION)..."
    conda env create -n "$ENV_NAME" \
        -f "https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/$VERSION/workflow/envs/alphapulldown.yaml"
fi

conda activate "$ENV_NAME"
command -v snakedeploy >/dev/null 2>&1 || die "snakedeploy missing from environment '$ENV_NAME'."

# --------------------------------------------------------------------------
# 3. Shared container image directory
# --------------------------------------------------------------------------
# Snakemake names cached images md5(<container-url>).simg and skips the pull
# when the file is already present. Keeping this directory outside the working
# directory is what makes containers download exactly once per machine.
mkdir -p "$IMAGE_DIR"
log "Shared container images: $IMAGE_DIR"

# --------------------------------------------------------------------------
# 4. Deploy the workflow
# --------------------------------------------------------------------------
if [ -e "$DEST" ]; then
    log "Directory '$DEST' already exists; skipping deployment."
else
    log "Deploying workflow $VERSION into '$DEST'..."
    snakedeploy deploy-workflow "$REPO" "$DEST" --tag "$VERSION"
fi

# Keep the deployed profiles pointing at the image directory chosen above.
# Done unconditionally so that deployments of older tags - whose profiles
# predate the apptainer-prefix key - also get a shared image directory.
for profile in "$DEST"/config/profiles/*/config.yaml; do
    [ -f "$profile" ] || continue
    if grep -q '^apptainer-prefix:' "$profile"; then
        sed -i.bak "s|^apptainer-prefix:.*|apptainer-prefix: \"$IMAGE_DIR\"|" "$profile"
    else
        printf '\napptainer-prefix: "%s"\n' "$IMAGE_DIR" >> "$profile"
    fi
    rm -f "$profile.bak"
done
log "Profiles pinned to $IMAGE_DIR"

# --------------------------------------------------------------------------
# 5. Pre-fetch containers referenced by the deployed config
# --------------------------------------------------------------------------
if [ "$DO_PULL" -eq 1 ]; then
    CONFIG="$DEST/config/config.yaml"
    [ -f "$CONFIG" ] || die "Deployed config not found at $CONFIG"

    # Only docker:// URLs need fetching; local .sif paths are used in place.
    urls=$(grep -E '^[[:space:]]*(prediction|analysis)_container:' "$CONFIG" \
           | sed -E 's/^[^:]*:[[:space:]]*//; s/^["'\'']//; s/["'\'']?[[:space:]]*(#.*)?$//' \
           | grep '^docker://' || true)

    if [ -z "$urls" ]; then
        log "No docker:// containers in config; nothing to pre-fetch."
    fi

    for url in $urls; do
        name=$(python -c 'import hashlib,sys; print(hashlib.md5(sys.argv[1].encode()).hexdigest())' "$url")
        target="$IMAGE_DIR/$name.simg"
        if [ -f "$target" ]; then
            log "Already cached: $url"
        else
            log "Pulling $url ..."
            # Pull to a temporary name so an interrupted transfer is never
            # mistaken for a complete image on the next run.
            tmp="$IMAGE_DIR/.$name.simg.partial"
            rm -f "$tmp"
            if "$CONTAINER_CMD" pull "$tmp" "$url"; then
                mv "$tmp" "$target"
            else
                rm -f "$tmp"
                warn "Failed to pull $url - Snakemake will retry on first run."
            fi
        fi
    done
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
cat <<EOF

Installation complete.

  conda activate $ENV_NAME
  cd $DEST
  # edit config/config.yaml, then:
  snakemake --profile config/profiles/desktop --cores 8

Containers are shared from $IMAGE_DIR and will not be downloaded again.
EOF
