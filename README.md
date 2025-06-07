# deepCDG-eval
This repository contains the evaluation of deepCDG.
##  Robustness evaluation on pan-cancer datasets
Performance evaluation can be run by calling

``python robust.py --perburbation ["Network", "Feature", "Rewired"] -- perburbation_ratio [0, 0.25, 0.5, 0.75, 0.9]``

## Time overhead analysis
Time overhead analysis can be run by calling

``python time&cuda.py``
