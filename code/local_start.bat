
Rem configs set

Rem Datasets: movielens census_income churn fraud_detection
set dataset=churn
set experiment=experiment_small_model_based_big_model

Rem 1 2 3 4(End)
set task_start_num=1
set task_end_num=4

Rem ################# Run local #################
:run_exp
    mkdir "local/console"
    echo. > local/local_task_num.txt

    set /a i=%task_start_num%
    :for_loop_run_task
    if %i% gtr %task_end_num% (goto for_loop_run_task_exit)
        python -u main.py --experiment %experiment% --dataset %dataset% --task_num %i% > local/console/%dataset%_%i%.txt 2>&1
        echo %i% >> local/local_task_num.txt
        set /a i+=1
        goto for_loop_run_task
    :for_loop_run_task_exit
    exit
