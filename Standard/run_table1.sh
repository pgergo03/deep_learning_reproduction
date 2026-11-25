#!/usr/bin/env bash
# Reproduces Table 1 (EfficientNet-B0, CIFAR-10). Run from Standard/. Logs go to logs/.
set -e
cd "$(dirname "$0")"
mkdir -p Models Results mult_adv CW_adv logs
P="../.venv/bin/python -u"
J=${JOBS:-3}

# name | train flags | eval flags (bil_m = bil used in the model file name)
VARIANTS=(
  "std|--preproc 0|--preproc 0 --bil_m 10 --time std"
  "bil|--preproc 1 --bil 10 --noise 0|--preproc 1 --bil 10 --noise 0 --bil_m 10 --time bil"
  "noise|--preproc 1 --bil 0 --noise 0.032|--preproc 1 --bil 0 --noise 0.032 --bil_m 0 --time noise"
  "nb|--preproc 1|--preproc 1 --bil_m 10 --time nb"
  "bn|--preproc 1 --noise_first 0|--preproc 1 --noise_first 0 --bil_m 10 --time bn"
)

train=(); evals=()
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name tflags eflags <<< "$v"
  ls Models/model_*_"$name" >/dev/null 2>&1 || train+=("$P train.py $tflags --tag $name > logs/train_$name.log 2>&1")
  evals+=("$P mult_adv.py $eflags --eps 0 0.03 > logs/mult_$name.log 2>&1")
  evals+=("$P CW_adv.py $eflags > logs/cw_$name.log 2>&1")
done

printf '%s\n' "${train[@]}" | xargs -P "$J" -I{} bash -c {}
printf '%s\n' "${evals[@]}" | xargs -P "$J" -I{} bash -c {}
$P summarize_table1.py
