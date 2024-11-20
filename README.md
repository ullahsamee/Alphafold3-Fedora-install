## Accurate structure prediction of biomolecular interactions with AlphaFold 3

AlphaFold 3 model with a substantially updated diffusion-based architecture that is capable of predicting the joint structure of complexes including proteins, nucleic acids, small molecules, ions and modified residues. The new AlphaFold model demonstrates substantially improved accuracy over many previous specialized tools: far greater accuracy for protein–ligand interactions compared with state-of-the-art docking tools, much higher accuracy for protein–nucleic acid interactions compared with nucleic-acid-specific predictors and substantially higher antibody–antigen prediction accuracy compared with AlphaFold-Multimer v.2.3.

see paper: <https://www.nature.com/articles/s41586-024-07487-w>

code: <https://github.com/google-deepmind/alphafold3/tree/main>

Installation: <https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md>

## First Obtain Model Parameters (1Gb file size)

<https://docs.google.com/forms/d/e/1FAIpQLSfWZAgo1aYk0O4MuAXZj8xRQ8DafeFJnldNOnh_13qAx2ceZw/viewform>

## Lets install AlphaFold3

**Prerequisites Hardware:**

-OS: Fedora39

-CUDA 12.4 or 12.6 enabled with NVIDIA Driver

\-*I have; 2xRTX 3090Ti GPU, 128GB Memory , 12TB SSD*

-Python3.11 or higher

***Important:*** make sure your GPU have 8.6 or higher compute capability. Check a nice discussion ***anything with GPU capability \< 8.0 produces bad results. here:*** <https://github.com/google-deepmind/alphafold3/issues/59> [![Nice info shared by Augustin-Zidek](images/Screenshot 2024-11-19 143427.png)](https://github.com/google-deepmind/alphafold3/issues/59#:~:text=rtx_2080_ti%20%20%20%20%20%207.5%20%20(bad)%0Artx_3090%20%20%20%20%20%20%20%20%208.6%0Artx_4090%20%20%20%20%20%20%20%20%208.9%0Atitan_rtx%20%20%20%20%20%20%20%207.5%20%20(bad)%0Aquadro_rtx_6000%20%207.5%20%20(bad)%0Av100%20%20%20%20%20%20%20%20%20%20%20%20%207.0%20%20(bad)%0Aa100_pcie_40gb%20%20%208.0%0Aa100_80gb%20%20%20%20%20%20%20%208.0)

**Prerequisites packages-system Setup:**

-   sudo dnf groupinstall "Development Tools"

-   sudo dnf install cmake-data

-   sudo dnf install gcc-c++ cmake

-   sudo dnf install boost-devel

-   sudo dnf install zstd

-   python -m pip install --upgrade pip

-   python -m pip install matplotlib

-   python -m pip install pandas

-   sudo dnf install python3.11-devel

-   sudo dnf install python3-numpy-devel

-   python -m pip install numpy --upgrade

-   sudo dnf install boost-numpy3 python3-numpy python3-numpy-f2py python3-numpy-doc

-   sudo dnf install python3.11

## **STEP1 - Install HMMER**

Install HMMER and change the path according to yours linux user name

Creat a directory(called biotools) in /home/your_username/biotools

``` bash
HMMER_DIR="/home/your_username/biotools/hmmer"

wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz

tar zxvf hmmer-3.4.tar.gz

cd hmmer-3.4

./configure --prefix=${HMMER_DIR}

make -j8

make -j8 install

cd easel

make install
```

## **STEP2 - Setup AlphaFold3**

Clone AF3 repo to biotools directory and download databases

``` bash
APPDIR="/home/your_username/biotools"

mkdir -p $APPDIR cd $APPDIR
git clone https://github.com/google-deepmind/alphafold3.git
ALPHAFOLD3DIR="$APPDIR/alphafold3"

cd ${ALPHAFOLD3DIR} 
mkdir public_databases 
chmod +x fetch_databases.sh 
./fetch_databases.sh 
```

It will fetch and unzip total 09-databases, make sure to check and unzip the ***"pdb_2022_09_28_mmcif_files.tar"*** in the directory(public_databases) if you want to predict small molecules and It takes alot of time unzipping.

## **STEP3 - Model Setup**

Decompress model file: zstd -d af3.bin.zst and move files; "af3.bin and af3.bin.zst" to /home/your_username/biotools/alphafold/models

``` bash
zstd -d af3.bin.zst
```
## **STEP4 - Environment Setup**

Create and activate virtual environment

``` bash
cd ${ALPHAFOLD3DIR}

python -m venv .venv

. .venv/bin/activate
```

Just check that you're going good by seeing something; /usr/bin/python when you type in terminal.

``` bash
which python
```

Install required packages using pip

``` bash
python -m pip install absl-py==2.1.0 chex==0.1.87 dm-haiku==0.0.13 dm-tree==0.1.8 \
    filelock==3.16.1 "jax[cuda12]==0.4.34" jax-cuda12-pjrt==0.4.34 \
    jax-triton==0.2.0 jaxlib==0.4.34 jaxtyping==0.2.34 jmp==0.0.4 \
    ml-dtypes==0.5.0 numpy==2.1.3 nvidia-cublas-cu12==12.6.3.3 \
    nvidia-cuda-cupti-cu12==12.6.80 nvidia-cuda-nvcc-cu12==12.6.77 \
    nvidia-cuda-runtime-cu12==12.6.77 nvidia-cudnn-cu12==9.5.1.17 \
    nvidia-cufft-cu12==11.3.0.4 nvidia-cusolver-cu12==11.7.1.2 \
    nvidia-cusparse-cu12==12.5.4.2 nvidia-nccl-cu12==2.23.4 \
    nvidia-nvjitlink-cu12==12.6.77 opt-einsum==3.4.0 pillow==11.0.0 \
    rdkit==2024.3.5 scipy==1.14.1 tabulate==0.9.0 toolz==1.0.0 \
    tqdm==4.67.0 triton==3.1.0 typeguard==2.13.3 \
    typing-extensions==4.12.2 zstandard==0.23.0
```

After this step you'll see ***"Successfully installed alphafold3-3.0.0" on your Terminal.***

``` bash
python -m pip install --no-deps .
```

Run build_data

``` bash
.venv/bin/build_data
```

Verify installation of AF3 works and check help message is displayed

``` bash
python run_alphafold.py --help
```

***Congratulations thats it!!!!***


**Before Running Predictions**

Check models directory outputs displayed

``` bash
ls -l /home/your_username/models
```

Check public databases outputs displayed

``` bash
ls -l /home/your_username/biotools/alphafold3/public_databases
```

Check HMMER binaries outputs displayed

``` bash
ls -l /home/your_username/biotools/hmmer/bin/hmm*
```

We're good now...

**Lets Run AF3 Predictions**
#### Time to run AF3, so two thing we need; input json file and bash script. You can read more on input here <https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>

Create input JSON file with protein sequence in a folder anywhere with any name. We'll both place json and script together and run it from there. Simply paste below code in terminal and press ENTER which will create a json file for a protein structure predictions.

``` bash
cat > test.json << 'EOL'
[
  {
    "name": "ALK2_prediction",
    "modelSeeds": [],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MVDGVMILPVLIMIALPSPSMEDEKPKVNPKLYMCVCEGLSCGNEDHCEGQQCFSSLSINDGFHVYQKGCFQVYEQGKMTCK",
          "count": 1
        }
      }
    ]
  }
]
EOL
```
Here's the bash Script to run AF3
Now copy the code and save into a file name **AF3_script.sh**
Update paths in run script: 
***PATHS AND DIRECTORIES*** according to your ***username***
*ALPHAFOLD3DIR="/home/your_username/biotools/alphafold3" HMMER3_BINDIR="/home/your_username/biotools/hmmer/bin" DB_DIR="/home/your_username/biotools/alphafold3/public_databases" MODEL_DIR="/home/your_username/biotools/alphafold3/models"*


``` bash
#!/bin/bash

#============= FANCY PROGRESS DISPLAY =============
function format_gpu_info() {
    nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd "+" - | cut -c -50
}

function format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

function monitor_progress() {
    local log_file=$1
    local start_time=$(date +%s)
    local dots=""
    local estimated_total=3600  # Initial estimate: 60 minutes
    local last_stage=""
    local stage_times=(
        "MSA:1200"           # ~10-20 minutes for MSA
        "Structure:1800"     
        "Processing:600"     
    )
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((estimated_total - elapsed))
        local percent=$((elapsed * 100 / estimated_total))
        
        # Ensure percent doesn't exceed 100
        if [ $percent -gt 100 ]; then
            percent=100
        fi
        
        # Format times
        local elapsed_fmt=$(format_time $elapsed)
        local remaining_fmt=$(format_time $remaining)
        
        # Check for different stages in the log file
        local current_stage=""
        if grep -q "Getting protein MSAs" "$log_file" 2>/dev/null; then
            current_stage="🧬 Generating MSA sequences"
            if [ "$last_stage" != "MSA" ]; then
                estimated_total=$((elapsed + 1200))  # Adjust total time based on MSA stage
                last_stage="MSA"
            fi
        elif grep -q "Running model" "$log_file" 2>/dev/null; then
            current_stage="🔮 Running structure prediction"
            if [ "$last_stage" != "Structure" ]; then
                estimated_total=$((elapsed + 1800))  # Adjust for structure prediction
                last_stage="Structure"
            fi
        elif grep -q "Processing chain" "$log_file" 2>/dev/null; then
            current_stage="🔄 Processing protein chains"
            if [ "$last_stage" != "Processing" ]; then
                estimated_total=$((elapsed + 600))   # Adjust for processing
                last_stage="Processing"
            fi
        else
            current_stage="⚙️  Initializing"
        fi
        
        # Create progress bar
        local bar_size=30
        local filled_size=$((percent * bar_size / 100))
        local unfilled_size=$((bar_size - filled_size))
        local progress_bar=$(printf "%${filled_size}s" | tr ' ' '█')
        local empty_bar=$(printf "%${unfilled_size}s" | tr ' ' '░')
        
        # Print progress
        echo -ne "\r\033[K${current_stage}"
        echo -ne " [${progress_bar}${empty_bar}] ${percent}%"
        echo -ne " | Elapsed: ${elapsed_fmt} | Est. remaining: ${remaining_fmt}${dots}"
        
        # Update dots animation
        dots=".$dots"
        if [ ${#dots} -gt 3 ]; then
            dots=""
        fi
        
        sleep 1
        
        # Check if parent process still exists
        if ! kill -0 $2 2>/dev/null; then
            echo -ne "\n"
            break
        fi
    done
}

#============= PATHS AND DIRECTORIES =============
ALPHAFOLD3DIR="/home/your_username/biotools/alphafold3"
HMMER3_BINDIR="/home/your_username/biotools/hmmer/bin"
DB_DIR="/home/your_username/biotools/alphafold3/public_databases"
MODEL_DIR="/home/your_username/biotools/alphafold3/models"

#============= GPU SETUP =============
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"

#============= SETUP AND CHECKS =============
WORK_DIR=$(pwd)
JSON_FILE=$(ls -1 *.json 2>/dev/null | head -n 1)

if [ -z "$JSON_FILE" ]; then
    echo "❌ Error: No JSON file found!"
    exit 1
fi

BASE_NAME=$(basename "$JSON_FILE" .json)
OUTPUT_DIR="${WORK_DIR}/output/${BASE_NAME}"
LOG_FILE="${OUTPUT_DIR}/af3_run.log"

# Create directories
mkdir -p "${OUTPUT_DIR}" 2>/dev/null

# Print initial status
echo "╔═══════════════════════════════════════════════════╗"
echo "║ 🚀 AlphaFold3 Prediction Script | ullahsamee      ║"
echo "╠═══════════════════════════════════════════════════╣"
printf "║ Input: %-43s ║\n" "$JSON_FILE"
printf "║ Output: %-42s ║\n" "$(basename ${OUTPUT_DIR})"
printf "║ GPUs: %-44s ║\n" "$(format_gpu_info)"
echo "╚═══════════════════════════════════════════════════╝"

#============= ACTIVATE VIRTUAL ENVIRONMENT =============
source "${ALPHAFOLD3DIR}/.venv/bin/activate" 2>/dev/null

#============= RUN ALPHAFOLD3 =============
cd "${ALPHAFOLD3DIR}"

echo "📋 Progress:"
echo "──────────────────────"

# Run AlphaFold3 with output redirected
{
    python run_alphafold.py \
        --jackhmmer_binary_path="${HMMER3_BINDIR}/jackhmmer" \
        --nhmmer_binary_path="${HMMER3_BINDIR}/nhmmer" \
        --hmmalign_binary_path="${HMMER3_BINDIR}/hmmalign" \
        --hmmsearch_binary_path="${HMMER3_BINDIR}/hmmsearch" \
        --hmmbuild_binary_path="${HMMER3_BINDIR}/hmmbuild" \
        --db_dir="${DB_DIR}" \
        --model_dir="${MODEL_DIR}" \
        --json_path="${WORK_DIR}/${JSON_FILE}" \
        --output_dir="${OUTPUT_DIR}" \
        --buckets="256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120" \
        2>&1 
} >> "${LOG_FILE}" &

PID=$!

# Monitor progress
monitor_progress "$LOG_FILE" "$PID"

# Wait for completion
wait $PID
EXIT_STATUS=$?

cd "${WORK_DIR}"

#============= COMPLETION STATUS =============
echo "──────────────────────"
if [ $EXIT_STATUS -eq 0 ]; then
    TOTAL_TIME=$(( $(date +%s) - start_time ))
    TOTAL_TIME_FMT=$(format_time $TOTAL_TIME)
    echo "✅ Prediction completed successfully in ${TOTAL_TIME_FMT}!"
    echo
    echo "📊 GPU Statistics:"
    echo "──────────────────"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
        --format=csv,noheader | \
        awk -F', ' '{printf "GPU %s: %s utilization, %s memory used, %s\n", $1, $2, $3, $4}'
    echo
    echo "📝 Log file: ${LOG_FILE}"
else
    echo "❌ Prediction failed! Check log file for details:"
    echo "📝 ${LOG_FILE}"
fi
```


**Execute AF3_script.sh**
``` bash
chmod +x AF3_script.sh

./AF3_script.sh
```

**Results**
You will get an output folder named based on the json input file that you feeded to AF3.

You can run this script from anywhere as long as you have JSON file placed in a folder together.

reach me out on [Linkedin](https://www.linkedin.com/in/samee-ullah-structural-biologist/), if you want me to model something for you
