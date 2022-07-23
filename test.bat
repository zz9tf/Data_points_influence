for /L %%i in (1,1,5) do (
    if %%i == 3 (
        echo %%i
    ) else (
        echo %%i Here I am
    )
)