docker rm -f $(docker ps -aq) && docker container prune -f && docker network prune -f && docker builder prune -f
docker container prune -f && docker network prune -f && docker builder prune -f
python experiments/run_replicationbench.py --run-id T-SE_run2 --num-proc 20
docker rm -f $(docker ps -aq) && docker container prune -f && docker network prune -f && docker builder prune -f
docker container prune -f && docker network prune -f && docker builder prune -f
python experiments/run_replicationbench.py --run-id T-SE_run3 --num-proc 20