repeats=500
n=500

for dataset in Sensor BA
do
    python experiment.py --dataset $dataset --n $n --proportion_remove 0.05 --repeats $repeats
    for K in {1..3}
    do
        python experiment.py --dataset $dataset --k $K --n $n --proportion_remove 0.05 --repeats $repeats
    done 
done


